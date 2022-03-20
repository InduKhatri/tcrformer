import os
import sys
import shutil
import comet_ml
import torch
import pandas as pd
import re
import math
from tqdm.auto import tqdm 
import numpy as np
from comet_ml import get_global_experiment
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib import colors as mcolors
from sklearn import preprocessing
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, balanced_accuracy_score,roc_curve, auc, precision_recall_curve
from dataclasses import dataclass
from typing import List, Optional, Tuple
from torch.utils.data import Dataset
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from sklearn.model_selection import train_test_split
import optuna
from packaging import version

from transformers.models.xlnet import XLNetPreTrainedModel
from transformers.modeling_utils import SequenceSummary
from transformers.models.xlnet.modeling_xlnet import XLNetLayer, XLNetModelOutput
from transformers import Trainer, TrainingArguments, TrainerCallback, EarlyStoppingCallback, XLNetForSequenceClassification, XLNetTokenizer, XLNetModel
from transformers.file_utils import ModelOutput
from transformers.optimization import get_linear_schedule_with_warmup, AdamW
from transformers.trainer_pt_utils import get_parameter_names
from transformers.trainer_utils import ShardedDDPOption

e = 0
ep = 1
model_name = 'prot_xlnet/'

class XLNetTCRModel(XLNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.mem_len = config.mem_len
        self.reuse_len = config.reuse_len
        self.d_model = config.d_model
        self.same_length = config.same_length
        self.attn_type = config.attn_type
        self.bi_data = config.bi_data
        self.clamp_len = config.clamp_len
        self.n_layer = config.n_layer

        self.word_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.v_gene_embeddings = nn.Embedding(65, config.d_model, padding_idx=64)
        self.j_gene_embeddings = nn.Embedding(15, config.d_model, padding_idx=14)
        self.mask_emb = nn.Parameter(torch.FloatTensor(1, 1, config.d_model))
        self.layer = nn.ModuleList([XLNetLayer(config) for _ in range(config.n_layer)])
        self.dropout = nn.Dropout(config.dropout)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.word_embedding

    def set_input_embeddings(self, new_embeddings):
        self.word_embedding = new_embeddings

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError

    def create_mask(self, qlen, mlen):
        """
        Creates causal attention mask. Float mask where 1.0 indicates masked, 0.0 indicates not-masked.
        Args:
            qlen: Sequence length
            mlen: Mask length
        ::
                  same_length=False: same_length=True: <mlen > < qlen > <mlen > < qlen >
               ^ [0 0 0 0 0 1 1 1 1] [0 0 0 0 0 1 1 1 1]
                 [0 0 0 0 0 0 1 1 1] [1 0 0 0 0 0 1 1 1]
            qlen [0 0 0 0 0 0 0 1 1] [1 1 0 0 0 0 0 1 1]
                 [0 0 0 0 0 0 0 0 1] [1 1 1 0 0 0 0 0 1]
               v [0 0 0 0 0 0 0 0 0] [1 1 1 1 0 0 0 0 0]
        """
        attn_mask = torch.ones([qlen, qlen])
        mask_up = torch.triu(attn_mask, diagonal=1)
        attn_mask_pad = torch.zeros([qlen, mlen])
        ret = torch.cat([attn_mask_pad, mask_up], dim=1)
        if self.same_length:
            mask_lo = torch.tril(attn_mask, diagonal=-1)
            ret = torch.cat([ret[:, :qlen] + mask_lo, ret[:, qlen:]], dim=1)

        ret = ret.to(self.device)
        return ret

    def cache_mem(self, curr_out, prev_mem):
        # cache hidden states into memory.
        if self.reuse_len is not None and self.reuse_len > 0:
            curr_out = curr_out[: self.reuse_len]

        if self.mem_len is None or self.mem_len == 0:
            # If `use_mems` is active but no `mem_len` is defined, the model behaves like GPT-2 at inference time
            # and returns all of the past and current hidden states.
            cutoff = 0
        else:
            # If `use_mems` is active and `mem_len` is defined, the model returns the last `mem_len` hidden
            # states. This is the preferred setting for training and long-form generation.
            cutoff = -self.mem_len
        if prev_mem is None:
            # if `use_mems` is active and `mem_len` is defined, the model
            new_mem = curr_out[cutoff:]
        else:
            new_mem = torch.cat([prev_mem, curr_out], dim=0)[cutoff:]

        return new_mem.detach()

    @staticmethod
    def positional_embedding(pos_seq, inv_freq, bsz=None):
        sinusoid_inp = torch.einsum("i,d->id", pos_seq, inv_freq)
        pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        pos_emb = pos_emb[:, None, :]

        if bsz is not None:
            pos_emb = pos_emb.expand(-1, bsz, -1)

        return pos_emb

    def relative_positional_encoding(self, qlen, klen, bsz=None):
        # create relative positional encoding.
        freq_seq = torch.arange(0, self.d_model, 2.0, dtype=torch.float)
        inv_freq = 1 / torch.pow(10000, (freq_seq / self.d_model))

        if self.attn_type == "bi":
            # beg, end = klen - 1, -qlen
            beg, end = klen, -qlen
        elif self.attn_type == "uni":
            # beg, end = klen - 1, -1
            beg, end = klen, -1
        else:
            raise ValueError(f"Unknown `attn_type` {self.attn_type}.")

        if self.bi_data:
            fwd_pos_seq = torch.arange(beg, end, -1.0, dtype=torch.float)
            bwd_pos_seq = torch.arange(-beg, -end, 1.0, dtype=torch.float)

            if self.clamp_len > 0:
                fwd_pos_seq = fwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)
                bwd_pos_seq = bwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)

            if bsz is not None:
                fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz // 2)
                bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq, bsz // 2)
            else:
                fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq)
                bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq)

            pos_emb = torch.cat([fwd_pos_emb, bwd_pos_emb], dim=1)
        else:
            fwd_pos_seq = torch.arange(beg, end, -1.0)
            if self.clamp_len > 0:
                fwd_pos_seq = fwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)
            pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz)

        pos_emb = pos_emb.to(self.device)
        return pos_emb
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mems=None,
        perm_mask=None,
        target_mapping=None,
        token_type_ids=None,
        input_mask=None,
        head_mask=None,
        inputs_embeds=None,
        v_gene_ids=None, 
        j_gene_ids=None, 
        use_mems=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,  # delete after depreciation warning is removed
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if "use_cache" in kwargs:
            warnings.warn(
                "The `use_cache` argument is deprecated and will be removed in a future version, use `use_mems` instead.",
                FutureWarning,
            )
            use_mems = kwargs["use_cache"]

        if self.training:
            use_mems = use_mems if use_mems is not None else self.config.use_mems_train
        else:
            use_mems = use_mems if use_mems is not None else self.config.use_mems_eval

        # the original code for XLNet uses shapes [len, bsz] with the batch dimension at the end
        # but we want a unified interface in the library with the batch size on the first dimension
        # so we move here the first dimension (batch) to the end
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_ids = input_ids.transpose(0, 1).contiguous()
            qlen, bsz = input_ids.shape[0], input_ids.shape[1]
        elif inputs_embeds is not None:
            inputs_embeds = inputs_embeds.transpose(0, 1).contiguous()
            qlen, bsz = inputs_embeds.shape[0], inputs_embeds.shape[1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        token_type_ids = token_type_ids.transpose(0, 1).contiguous() if token_type_ids is not None else None
        v_gene_ids = v_gene_ids.transpose(0, 1).contiguous() if v_gene_ids is not None else None
        j_gene_ids = j_gene_ids.transpose(0, 1).contiguous() if j_gene_ids is not None else None
        input_mask = input_mask.transpose(0, 1).contiguous() if input_mask is not None else None
        attention_mask = attention_mask.transpose(0, 1).contiguous() if attention_mask is not None else None
        perm_mask = perm_mask.permute(1, 2, 0).contiguous() if perm_mask is not None else None
        target_mapping = target_mapping.permute(1, 2, 0).contiguous() if target_mapping is not None else None

        mlen = mems[0].shape[0] if mems is not None and mems[0] is not None else 0
        klen = mlen + qlen

        dtype_float = self.dtype
        device = self.device

        # Attention mask
        # causal attention mask
        if self.attn_type == "uni":
            attn_mask = self.create_mask(qlen, mlen)
            attn_mask = attn_mask[:, :, None, None]
        elif self.attn_type == "bi":
            attn_mask = None
        else:
            raise ValueError(f"Unsupported attention type: {self.attn_type}")

        # data mask: input mask & perm mask
        assert input_mask is None or attention_mask is None, "You can only use one of input_mask (uses 1 for padding) "
        "or attention_mask (uses 0 for padding, added for compatibility with BERT). Please choose one."
        if input_mask is None and attention_mask is not None:
            input_mask = 1.0 - attention_mask
        if input_mask is not None and perm_mask is not None:
            data_mask = input_mask[None] + perm_mask
        elif input_mask is not None and perm_mask is None:
            data_mask = input_mask[None]
        elif input_mask is None and perm_mask is not None:
            data_mask = perm_mask
        else:
            data_mask = None

        if data_mask is not None:
            # all mems can be attended to
            if mlen > 0:
                mems_mask = torch.zeros([data_mask.shape[0], mlen, bsz]).to(data_mask)
                data_mask = torch.cat([mems_mask, data_mask], dim=1)
            if attn_mask is None:
                attn_mask = data_mask[:, :, :, None]
            else:
                attn_mask += data_mask[:, :, :, None]

        if attn_mask is not None:
            attn_mask = (attn_mask > 0).to(dtype_float)

        if attn_mask is not None:
            non_tgt_mask = -torch.eye(qlen).to(attn_mask)
            if mlen > 0:
                non_tgt_mask = torch.cat([torch.zeros([qlen, mlen]).to(attn_mask), non_tgt_mask], dim=-1)
            non_tgt_mask = ((attn_mask + non_tgt_mask[:, :, None, None]) > 0).to(attn_mask)
        else:
            non_tgt_mask = None

        # Word embeddings and prepare h & g hidden states
        if inputs_embeds is not None:
            word_emb_k = inputs_embeds
            v_gene_embeddings = self.v_gene_embeddings(v_gene_ids)
            j_gene_embeddings = self.j_gene_embeddings(j_gene_ids)
        else:
            word_emb_k = self.word_embedding(input_ids)
            v_gene_embeddings = self.v_gene_embeddings(v_gene_ids)
            j_gene_embeddings = self.j_gene_embeddings(j_gene_ids)
        
        output_h = self.dropout(word_emb_k + v_gene_embeddings + j_gene_embeddings)
        if target_mapping is not None:
            word_emb_q = self.mask_emb.expand(target_mapping.shape[0], bsz, -1)
            # else:  # We removed the inp_q input which was same as target mapping
            #     inp_q_ext = inp_q[:, :, None]
            #     word_emb_q = inp_q_ext * self.mask_emb + (1 - inp_q_ext) * word_emb_k
            output_g = self.dropout(word_emb_q)
        else:
            output_g = None

        # Segment embedding
        if token_type_ids is not None:
            # Convert `token_type_ids` to one-hot `seg_mat`
            if mlen > 0:
                mem_pad = torch.zeros([mlen, bsz], dtype=torch.long, device=device)
                cat_ids = torch.cat([mem_pad, token_type_ids], dim=0)
            else:
                cat_ids = token_type_ids

            # `1` indicates not in the same segment [qlen x klen x bsz]
            seg_mat = (token_type_ids[:, None] != cat_ids[None, :]).long()
            seg_mat = nn.functional.one_hot(seg_mat, num_classes=2).to(dtype_float)
        else:
            seg_mat = None

        # Positional encoding
        pos_emb = self.relative_positional_encoding(qlen, klen, bsz=bsz)
        pos_emb = self.dropout(pos_emb)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads] (a head_mask for each layer)
        # and head_mask is converted to shape [num_hidden_layers x qlen x klen x bsz x n_head]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                head_mask = head_mask.expand(self.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to float if need + fp16 compatibility
        else:
            head_mask = [None] * self.n_layer

        new_mems = ()
        if mems is None:
            mems = [None] * len(self.layer)

        attentions = [] if output_attentions else None
        hidden_states = [] if output_hidden_states else None
        for i, layer_module in enumerate(self.layer):
            if use_mems:
                # cache new mems
                new_mems = new_mems + (self.cache_mem(output_h, mems[i]),)
            if output_hidden_states:
                hidden_states.append((output_h, output_g) if output_g is not None else output_h)

            outputs = layer_module(
                output_h,
                output_g,
                attn_mask_h=non_tgt_mask,
                attn_mask_g=attn_mask,
                r=pos_emb,
                seg_mat=seg_mat,
                mems=mems[i],
                target_mapping=target_mapping,
                head_mask=head_mask[i],
                output_attentions=output_attentions,
            )
            output_h, output_g = outputs[:2]
            if output_attentions:
                attentions.append(outputs[2])

        # Add last hidden state
        if output_hidden_states:
            hidden_states.append((output_h, output_g) if output_g is not None else output_h)

        output = self.dropout(output_g if output_g is not None else output_h)

        # Prepare outputs, we transpose back here to shape [bsz, len, hidden_dim] (cf. beginning of forward() method)
        output = output.permute(1, 0, 2).contiguous()

        if not use_mems:
            new_mems = None

        if output_hidden_states:
            if output_g is not None:
                hidden_states = tuple(h.permute(1, 0, 2).contiguous() for hs in hidden_states for h in hs)
            else:
                hidden_states = tuple(hs.permute(1, 0, 2).contiguous() for hs in hidden_states)

        if output_attentions:
            if target_mapping is not None:
                # when target_mapping is provided, there are 2-tuple of attentions
                attentions = tuple(
                    tuple(att_stream.permute(2, 3, 0, 1).contiguous() for att_stream in t) for t in attentions
                )
            else:
                attentions = tuple(t.permute(2, 3, 0, 1).contiguous() for t in attentions)

        if not return_dict:
            return tuple(v for v in [output, new_mems, hidden_states, attentions] if v is not None)

        return XLNetModelOutput(
            last_hidden_state=output, mems=new_mems, hidden_states=hidden_states, attentions=attentions
        )

@dataclass
class XLNetForSequenceClassificationOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    mems: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class XLNetForTCRClassification(XLNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.alpha = 1
        self.method = config.task_specific_params['method']
        self.gamma = config.task_specific_params['gamma']

        self.num_labels = config.num_labels
        self.config = config

        if self.method == 1:
            self.transformer = XLNetModel(config)
            self.sequence_summary = SequenceSummary(config)
            self.logits_proj = nn.Bilinear(config.hidden_size, 2 , config.num_labels)
        elif self.method == 2:
            self.transformer = XLNetTCRModel(config)
            self.sequence_summary = SequenceSummary(config)
            self.logits_proj = nn.Linear(config.hidden_size, config.num_labels)
        else:
            self.transformer = XLNetModel(config)
            self.sequence_summary = SequenceSummary(config)
            self.logits_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mems=None,
        perm_mask=None,
        target_mapping=None,
        token_type_ids=None,
        input_mask=None,
        head_mask=None,
        vgenes = None,
        jgenes = None,
        inputs_embeds=None,
        labels=None,
        use_mems=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,  # delete when `use_cache` is removed in XLNetModel
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if self.method == 2: 
            transformer_outputs = self.transformer(
                input_ids,
                attention_mask=attention_mask,
                mems=mems,
                perm_mask=perm_mask,
                target_mapping=target_mapping,
                token_type_ids=token_type_ids,
                input_mask=input_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                v_gene_ids=vgenes, 
                j_gene_ids=jgenes,
                use_mems=use_mems,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )
        else:
            transformer_outputs = self.transformer(
                input_ids,
                attention_mask=attention_mask,
                mems=mems,
                perm_mask=perm_mask,
                target_mapping=target_mapping,
                token_type_ids=token_type_ids,
                input_mask=input_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                use_mems=use_mems,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        output = transformer_outputs[0]
        
        output = self.sequence_summary(output)
        
        # Size([32, 1024])
        if self.method == 1:
        # Additional Features gets concatenated to Pooled Output from Transformers
            if vgenes is not None:
                vgene = vgenes
                # Size ([32])
                vgene = vgene.unsqueeze(dim=1)
                # Size ([32,1])
            if jgenes is not None:
                jgene = jgenes
                # Size ([32])
                jgene = jgene.unsqueeze(dim=1)
                # Size ([32,1])
            genes = torch.cat((vgene, jgene), dim=1)
            genes = genes.type(torch.cuda.FloatTensor)
            logits = self.logits_proj(output, genes)
        else:
            logits = self.logits_proj(output)            
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss = torch.nn.functional.cross_entropy(input=logits.view(-1, self.num_labels), target=labels.view(-1), reduction='none')
                pt = torch.exp(-loss)
                loss = (self.alpha * (1-pt)**self.gamma * loss).mean()
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return XLNetForSequenceClassificationOutput(
            loss=loss,
            logits=logits,
            mems=transformer_outputs.mems,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

class ProtTrans(Dataset):
    """
    Setup for ProtTrans finetune with VDJdb dataset
    """

    def __init__(self, split="train", tokenizer_name='Rostlab/prot_bert', max_length=100):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.datasetFolderPath = '/tudelft.net/staff-umbrella/tcr/dataset/'
        self.FilePath = os.path.join(self.datasetFolderPath, 'beta_only.csv')
        self.tokenizer = XLNetTokenizer.from_pretrained(tokenizer_name, do_lower_case=False)
        self.max_length = max_length
        df = pd.read_csv(self.FilePath,names=['complex.id', 'Gene', 'CDR3', 'V', 'J', 'Species', 'MHC A', 'MHC B',
       'MHC class', 'Epitope', 'Epitope gene', 'Epitope species', 'Score', 'labels'],skiprows=1)
        label_enc = preprocessing.LabelEncoder()
        df['labels'] = label_enc.fit_transform(df['labels'].astype(str))
        self.mapping = dict(zip(label_enc.classes_, range(len(label_enc.classes_))))
        self.labels_dic = {y:x for x,y in self.mapping.items()}
    
        cate = [['TRBV1', 'TRBV10-1', 'TRBV10-2', 'TRBV10-3', 'TRBV11-1', 'TRBV11-2', 'TRBV11-3', 'TRBV12-1', 'TRBV12-2', 'TRBV12-3', 'TRBV12-4', 'TRBV12-5', 'TRBV13', 'TRBV13-1', 'TRBV13-2', 'TRBV13-3', 'TRBV14', 'TRBV15', 'TRBV16', 'TRBV17', 'TRBV18', 'TRBV19', 'TRBV2', 'TRBV20', 'TRBV20-1', 'TRBV23', 'TRBV24', 'TRBV24-1', 'TRBV25-1', 'TRBV26', 'TRBV27', 'TRBV28', 'TRBV29', 'TRBV29-1', 'TRBV3', 'TRBV3-1', 'TRBV30', 'TRBV31', 'TRBV4', 'TRBV4-1', 'TRBV4-2', 'TRBV4-3', 'TRBV5', 'TRBV5-1', 'TRBV5-4', 'TRBV5-5', 'TRBV5-6', 'TRBV5-8', 'TRBV6-1', 'TRBV6-2', 'TRBV6-3', 'TRBV6-4', 'TRBV6-5', 'TRBV6-6', 'TRBV6-9', 'TRBV7-2', 'TRBV7-3', 'TRBV7-4', 'TRBV7-6', 'TRBV7-7', 'TRBV7-8', 'TRBV7-9', 'TRBV9'], 
                ['TRBJ1-1', 'TRBJ1-2', 'TRBJ1-3', 'TRBJ1-4', 'TRBJ1-5', 'TRBJ1-6', 'TRBJ2-1', 'TRBJ2-2', 'TRBJ2-3', 'TRBJ2-4', 'TRBJ2-5', 'TRBJ2-6', 'TRBJ2-7']]
        enc = preprocessing.OrdinalEncoder(categories=cate)
        enc.fit(df[["V","J"]])
        df[["V","J"]] = enc.transform(df[["V","J"]])

        # train, test = train_test_split(df, stratify=df['labels'], test_size=0.20, random_state=44)
        train_ratio = 0.70
        validation_ratio = 0.15
        test_ratio = 0.15
        
        # train is now 75% of the entire data set
        # the _junk suffix means that we drop that variable completely
        train, test = train_test_split(df, stratify=df['labels'], test_size=1 - train_ratio, random_state=44)

        # test is now 10% of the initial data set
        # validation is now 15% of the initial data set
        val, test = train_test_split(test, stratify=test['labels'], test_size=test_ratio/(test_ratio + validation_ratio), random_state=44) 
        self.num_labels = self.labels_dic.__len__()

        if split=="train":
            seqs = list(train['CDR3'])
            vgenes = list(train['V'])
            jgenes = list(train['J'])
            labels = list(train['labels'])

            assert len(seqs) == len(labels)
            assert len(vgenes) == len(labels)
            assert len(jgenes) == len(labels)

            self.seqs, self.vgenes, self.jgenes, self.labels = seqs, vgenes, jgenes, labels
        if split == "test":
            seqs = list(test['CDR3'])
            vgenes = list(test['V'])
            jgenes = list(test['J'])
            labels = list(test['labels'])

            assert len(seqs) == len(labels)
            assert len(vgenes) == len(labels)
            assert len(jgenes) == len(labels)
            self.seqs, self.vgenes, self.jgenes, self.labels = seqs, vgenes, jgenes, labels
        
        if split == "valid":
            seqs = list(val['CDR3'])
            vgenes = list(val['V'])
            jgenes = list(val['J'])
            labels = list(val['labels'])

            assert len(seqs) == len(labels)
            assert len(vgenes) == len(labels)
            assert len(jgenes) == len(labels)
            self.seqs, self.vgenes, self.jgenes, self.labels = seqs, vgenes, jgenes, labels 

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if ep == 0:
            if torch.is_tensor(idx):
                idx = idx.tolist()
            seq = " ".join("".join(self.seqs[idx].split()))
            seq = re.sub(r"[UZOB]", "X", seq)
            seq_ids = self.tokenizer(seq, truncation=True, padding='max_length', max_length=self.max_length)
            sample = {key: torch.tensor(val) for key, val in seq_ids.items()}
            sample['labels'] = torch.tensor(self.labels[idx])
        elif ep == 1:
            if torch.is_tensor(idx):
                idx = idx.tolist()
            seq = " ".join("".join(self.seqs[idx].split()))
            seq = re.sub(r"[UZOB]", "X", seq)
            seq_ids = self.tokenizer(seq, truncation=True, padding='max_length', max_length=self.max_length)
            sample = {key: torch.tensor(val) for key, val in seq_ids.items()}
            sample['labels'] = torch.tensor(self.labels[idx])
            sample['vgenes'] = torch.tensor(self.vgenes[idx])
            sample['jgenes'] = torch.tensor(self.jgenes[idx])
        else:
            if torch.is_tensor(idx):
                idx = idx.tolist()
            seq = " ".join("".join(self.seqs[idx].split()))
            seq = re.sub(r"[UZOB]", "X", seq)
            length = len(self.seqs[idx])
            seq_ids = self.tokenizer(seq, truncation=True, padding='max_length', max_length=self.max_length)
            sample = {key: torch.tensor(val) for key, val in seq_ids.items()}
            sample['labels'] = torch.tensor(self.labels[idx])
            
            v = [64] * (self.max_length - (length+2))
            v += [int(self.vgenes[idx])] * length
            v += [64] * 2
            sample['vgenes'] = torch.tensor(v)

            j = [14] * (self.max_length - (length+2))
            j += [int(self.jgenes[idx])] * length
            j += [14] * 2
            sample['jgenes'] = torch.tensor(j)
        return sample            

class MyCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if state.is_world_process_zero:
            experiment = comet_ml.get_global_experiment()
            if experiment is not None:
                experiment._set_model_graph(model, framework="transformers")
                experiment._log_parameters(args, prefix="args/", framework="transformers")
                experiment._log_parameters(model.config, prefix="config/", framework="transformers")

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        def to_numpy(x):
            return x.cpu().detach().numpy()
        if state.is_world_process_zero:
            experiment = comet_ml.get_global_experiment()
            if experiment is not None:
                experiment._log_metrics(metrics, step=state.global_step, epoch=math.ceil(state.epoch), framework="transformers")
                for name, layer in zip(model._modules, model.children()):
                    if "transformer" in name:
                        wname = "%s.%s" % (name, "weight")
                        try:
                            experiment.log_histogram_3d(to_numpy(layer.word_embedding.weight), name="%s_%s.%s" % (name, "word_embedding", "weight"), step=state.global_step, epoch=math.ceil(state.epoch))
                            experiment.log_histogram_3d(to_numpy(layer.v_gene_embeddings.weight), name="%s_%s.%s" % (name, "v_gene_embedding", "weight"), step=state.global_step, epoch=math.ceil(state.epoch))
                            experiment.log_histogram_3d(to_numpy(layer.j_gene_embeddings.weight), name="%s_%s.%s" % (name, "j_gene_embedding", "weight"), step=state.global_step, epoch=math.ceil(state.epoch))
                        except:
                            experiment.log_histogram_3d(to_numpy(layer.word_embedding.weight), name=wname, step=state.global_step, epoch=math.ceil(state.epoch))
                    if "sequence_summary" in name:
                        wname = "%s.%s" % (name, "weight")
                        bname = "%s.%s" % (name, "bias")
                        experiment.log_histogram_3d(to_numpy(layer.summary.weight), name=wname, step=state.global_step, epoch=math.ceil(state.epoch))
                        experiment.log_histogram_3d(to_numpy(layer.summary.bias), name=bname, step=state.global_step, epoch=math.ceil(state.epoch))
                    if "activ" in name:
                        continue
                    if not hasattr(layer, "weight"):
                        continue
                    wname = "%s.%s" % (name, "weight")
                    bname = "%s.%s" % (name, "bias")
                    experiment.log_histogram_3d(to_numpy(layer.weight), name=wname, step=state.global_step, epoch=math.ceil(state.epoch))
                    experiment.log_histogram_3d(to_numpy(layer.bias), name=bname, step=state.global_step, epoch=math.ceil(state.epoch))
                del(name)
                del(layer)
                         
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if state.is_world_process_zero:
            experiment = comet_ml.get_global_experiment()
            if experiment is not None:
                try:
                    experiment.log_metric('train_loss', logs['loss'], step=state.global_step, epoch=logs['epoch'], include_context=True)
                    experiment.log_metric('learning_rate', logs['learning_rate'], step=state.global_step, epoch=logs['epoch'], include_context=True)
                    experiment._log_metrics(logs, step=state.global_step, epoch=logs['epoch'], framework="transformers")
                except:
                    experiment._log_metrics(logs, step=state.global_step, epoch=math.ceil(logs['epoch']), framework="transformers")


def compute_metrics(pred):
    global e
    experiment = get_global_experiment()
    if e <= 50:
        e += 1
    else:
        e = 1
    y_true = pred.label_ids
    y_pred = pred.predictions.argmax(-1)
    labels = list(train_dataset.labels_dic.values())
    y_scores = pred.predictions
    y_scores = torch.nn.functional.softmax(torch.from_numpy(y_scores).float())

    # Confusion Matrix
    conf = experiment.create_confusion_matrix(labels=list(train_dataset.labels_dic.values()), max_categories=train_dataset.num_labels)
    conf.compute_matrix(y_true=y_true, y_predicted=y_pred)
    experiment.log_confusion_matrix(matrix=conf, file_name="confusion-matrix-%02d.json" % e)
    
    # Classification Report
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    for label, metric in report.items():
        try:
            experiment.log_metrics(metric, prefix=f'eval_{label}', epoch=e)
        except:
            experiment.log_metric(label, metric, epoch=e)
    
    # Evaluation Metrics
    acc = balanced_accuracy_score(y_true, y_pred)
    # acc_unbalanced = accuracy_score(y_true, y_pred)

    #code to get colors
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                for name, color in colors.items())
    sorted_names = [name for hsv, name in by_hsv]
    from itertools import cycle
    colors = cycle(np.random.choice(sorted_names, len(labels), replace=True))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    
    # if e == 10:
    nclasses = list(range(0, test_dataset.num_labels))
    y_t = label_binarize(y_true, classes=nclasses)
    y_p = y_scores
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    auc_precision_recall=dict()
    prec = dict()
    recc = dict()
    lw=2
    for i in range(len(nclasses)):
        prec[i], recc[i], _ = precision_recall_curve(y_t[:, i], y_p[:, i])
        fpr[i], tpr[i], _ = roc_curve(y_t[:, i], y_p[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        auc_precision_recall[i] = auc(recc[i], prec[i])
        
        experiment.log_curve(f"pr-curve-class-{i}", recc[i], prec[i], step=e)
        experiment.log_curve(f"roc-curve-class-{i}", fpr[i], tpr[i], step=e)
        experiment.log_metric(f'auc-class-{i}', roc_auc[i], epoch=e)
        experiment.log_metric(f'auprc-class-{i}', auc_precision_recall[i], epoch=e)

    for i, color in zip(range(len(nclasses)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label='{0} (area = {1:0.2f})'
            ''.format(train_dataset.labels_dic[i], roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    plt.savefig('/tudelft.net/staff-umbrella/tcr/md_xln/roc.png', bbox_inches = 'tight')
    experiment.log_image('/tudelft.net/staff-umbrella/tcr/md_xln/roc.png', name='roc', overwrite=False, image_format="png",
                    image_scale=1.0, image_shape=None, image_colormap=None,
                    image_minmax=None, image_channels="last", copy_to_tmp=True, step=e)
    plt.close()

    for i, color in zip(range(len(nclasses)), colors):
        plt.plot(prec[i], recc[i], color=color, lw=2, label='{0} (area = {1:0.2f})'
            ''.format(train_dataset.labels_dic[i], auc_precision_recall[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall')
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    plt.savefig('/tudelft.net/staff-umbrella/tcr/md_xln/prc.png', bbox_inches = 'tight')
    experiment.log_image('/tudelft.net/staff-umbrella/tcr/md_xln/prc.png', name='prc', overwrite=False, image_format="png",
                    image_scale=1.0, image_shape=None, image_colormap=None,
                    image_minmax=None, image_channels="last", copy_to_tmp=True, step=e)
    plt.close()

    return {
        'accuracy': acc,
        'f1': report['weighted avg']['f1-score']
    }

class Train3r(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        """
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters and 'gene' not in n],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters and 'gene' in n],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate*10,
                },
            ]
            optimizer_cls = Adafactor if self.args.adafactor else AdamW
            if self.args.adafactor:
                optimizer_cls = Adafactor
                optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
            else:
                optimizer_cls = AdamW
                optimizer_kwargs = {
                    "betas": (self.args.adam_beta1, self.args.adam_beta2),
                    "eps": self.args.adam_epsilon,
                }
            # optimizer_kwargs["lr"] = self.args.learning_rate
            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        # if is_sagemaker_mp_enabled():
        #     self.optimizer = smp.DistributedOptimizer(self.optimizer)
        # self.lr_scheduler = get_linear_schedule_with_warmup(self.optimizer, num_training_steps=self.args.num_training_steps, num_warmup_steps=self.args.num_warmup_steps)   
        return self.optimizer

train_dataset = ProtTrans(split="train", tokenizer_name=model_name, max_length=40)
val_dataset = ProtTrans(split="valid", tokenizer_name=model_name, max_length=40)
test_dataset = ProtTrans(split="test", tokenizer_name=model_name, max_length=40)
label_names = list(train_dataset.labels_dic.values())

# def objective(trial: optuna.Trial):
experiment = comet_ml.Experiment(    
    api_key="",
    project_name="xlnet-tuner-1-SO",
    workspace=""
    )
hyper_parameters = {
    "seed": 6,
    "learning_rate" : 1.6593359209831075e-05,
    "gradient_acm": 27,
    "dropout": 0.0,
    "summary_last_dropout": 0.5,
    "weight_decay" :1e-06,
    "warmup_ratio" : 0.1,
    "adam_beta1":0.8,
    "adam_beta2": 0.752,
    "gamma": 5.5
    }

experiment.log_parameters(hyper_parameters)
experiment.add_tags([
                    'weight_decay_{}'.format(hyper_parameters["weight_decay"]), 
                    'le_rate_{}'.format(hyper_parameters["learning_rate"]), 
                    'gradient_accumulation_steps_{}'.format(hyper_parameters["gradient_acm"]),
                    'seed_{}'.format(hyper_parameters["seed"]),
                    'gamma_{}'.format(hyper_parameters["gamma"]),
                    'dropout_{}'.format(hyper_parameters["dropout"]),
                    'summary_last_dropout_{}'.format(hyper_parameters["summary_last_dropout"]),
                    'adam_beta1_{}'.format(hyper_parameters["adam_beta1"]),
                    'adam_beta2_{}'.format(hyper_parameters["adam_beta2"]),
                    'warmup_ratio_{}'.format(hyper_parameters["warmup_ratio"])
                        ])

training_args = TrainingArguments(  
    output_dir='/tudelft.net/staff-umbrella/tcr/md_xln/results',
    logging_dir='/tudelft.net/staff-umbrella/tcr/md_xln/logs',
    num_train_epochs=50,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=8,
    seed=hyper_parameters["seed"],
    weight_decay=hyper_parameters["weight_decay"],                                                                                                                        
    learning_rate=hyper_parameters["learning_rate"],
    gradient_accumulation_steps=hyper_parameters["gradient_acm"],
    adam_beta1=hyper_parameters["adam_beta1"],
    adam_beta2=hyper_parameters["adam_beta2"],
    warmup_ratio=hyper_parameters["warmup_ratio"],                                
    overwrite_output_dir=True,                                                                                         
    logging_steps=10,  
    do_train=True,                                        
    do_eval=True,
    load_best_model_at_end=True, 
    greater_is_better=False,  
    fp16=True,        
    metric_for_best_model="eval_loss",                                          
    evaluation_strategy="epoch",                         
    save_strategy="epoch", 
    fp16_opt_level="02",
    disable_tqdm=True,
    save_total_limit=3
)
def model_init():
    task_specific_params = {
        'gamma': hyper_parameters["gamma"],
        'method': ep
        }
    model = XLNetForTCRClassification.from_pretrained(model_name, 
                                                    num_labels=train_dataset.num_labels,
                                                    id2label=train_dataset.labels_dic, 
                                                    label2id=train_dataset.mapping, 
                                                    summary_last_dropout=hyper_parameters["summary_last_dropout"],
                                                    dropout=hyper_parameters["dropout"], 
                                                    task_specific_params = task_specific_params)
    return model
trainer = Trainer(
    model_init=model_init,                # the instantiated 🤗 Transformers model to be trained
    args=training_args,                   # training arguments, defined above
    train_dataset=train_dataset,          # training dataset
    eval_dataset=val_dataset,             # evaluation dataset
    compute_metrics = compute_metrics,     # evaluation metrics
    callbacks=[MyCallback, EarlyStoppingCallback(3, 0.0)]
    )
trainer.train()

results = trainer.evaluate()
perplexity = round(math.exp(results['eval_loss']),2)
experiment.log_metric('eval_perplexity', perplexity, include_context=True)
experiment.log_metric('eval_loss_es', results['eval_loss'], include_context=True)
experiment.log_metric('eval_acc_es', results['eval_accuracy'], include_context=True)
experiment.log_metric('eval_f1_es', results['eval_f1'], include_context=True)
test_results = trainer.predict(test_dataset)
perplexity = round(math.exp(test_results.metrics['test_loss']),2)
experiment.log_metric('test_perplexity', perplexity, include_context=True)
experiment.log_metric('test_loss_es', test_results.metrics['test_loss'], include_context=True)
experiment.log_metric('test_acc_es', test_results.metrics['test_accuracy'], include_context=True)
experiment.log_metric('test_f1_es', test_results.metrics['test_f1'], include_context=True)
experiment.end()

torch.cuda.empty_cache()
try:
    shutil.rmtree('/tudelft.net/staff-umbrella/tcr/md_xln/results')
    shutil.rmtree('/tudelft.net/staff-umbrella/tcr/md_xln/logs')
except OSError as er:
    print ("Error CLEANING!: %s - %s." % (er.filename, er.strerror))
#     global e
#     e = 0
#     return results['eval_f1']
   
# study_name = "xlnet-tuner-1-SO"  # Unique identifier of the study.
# storage_name = "sqlite:///{}.db".format(study_name)
# study = optuna.create_study(study_name=study_name, storage=storage_name,directions=["maximize"], load_if_exists=True)
# study.optimize(objective, n_trials=1)
# print("Number of finished trials: ", len(study.trials))