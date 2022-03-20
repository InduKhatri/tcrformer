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
from transformers import BertModel, BertPreTrainedModel, BertTokenizer, Trainer, TrainingArguments, BertForSequenceClassification, TrainerCallback, EarlyStoppingCallback
from torch.utils.data import Dataset
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutput 
from sklearn.model_selection import train_test_split
import optuna

from transformers.optimization import get_linear_schedule_with_warmup, AdamW

from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler
from packaging import version
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from transformers.optimization import get_linear_schedule_with_warmup, AdamW
from transformers.trainer_pt_utils import get_parameter_names
from transformers.trainer_utils import ShardedDDPOption

e = 0
model_name = 'prot_bert/'

class BertTCREmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.v_gene_embeddings = nn.Embedding(65, config.hidden_size, padding_idx=64)
        self.j_gene_embeddings = nn.Embedding(15, config.hidden_size, padding_idx=14)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "token_type_ids",
                torch.zeros(self.position_ids.size(), dtype=torch.long),
                persistent=False,
            )

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, v_gene_ids=None, j_gene_ids=None,  past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        v_gene_embeddings = self.v_gene_embeddings(v_gene_ids)
        j_gene_embeddings = self.j_gene_embeddings(j_gene_ids)

        embeddings = inputs_embeds + v_gene_embeddings + j_gene_embeddings + token_type_embeddings
        # embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertTCRModel(BertPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertTCREmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        vgenes = None,
        jgenes = None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            v_gene_ids = vgenes,
            j_gene_ids = jgenes,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

class BertForTCRClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.alpha = 1
        self.method = config.task_specific_params['method']
        self.gamma = config.task_specific_params['gamma']

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        
        if self.method == 1:
            self.bert = BertModel(config)
            self.dropout = nn.Dropout(classifier_dropout)
            self.classifier = nn.Bilinear(config.hidden_size, 2 , config.num_labels)
        elif self.method == 2:
            self.bert = BertTCRModel(config)
            self.dropout = nn.Dropout(classifier_dropout)
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        else:
            self.bert = BertModel(config)
            self.dropout = nn.Dropout(classifier_dropout)
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        self.post_init()
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        vgenes = None,
        jgenes = None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
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
            logits = self.classifier(pooled_output, genes)

        else:
            logits = self.classifier(pooled_output)
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
                # loss_fct = CrossEntropyLoss()
                # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                loss = torch.nn.functional.cross_entropy(input=logits.view(-1, self.num_labels), target=labels.view(-1), reduction='none')
                pt = torch.exp(-loss)
                loss = (self.alpha * (1-pt)**self.gamma * loss).mean()
                # loss = focal_loss(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
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
        # self.trainFilePath = os.path.join(self.datasetFolderPath, 'train_beta_large.csv')
        # self.testFilePath = os.path.join(self.datasetFolderPath, 'test_beta_large.csv')
        self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
        self.max_length = max_length
        # changes
        df = pd.read_csv(self.FilePath,names=['complex.id', 'Gene', 'CDR3', 'V', 'J', 'Species', 'MHC A', 'MHC B',
       'MHC class', 'Epitope', 'Epitope gene', 'Epitope species', 'Score', 'labels'],skiprows=1)
        
        label_enc = preprocessing.LabelEncoder()
        df['labels'] = label_enc.fit_transform(df['labels'].astype(str))
        self.mapping = dict(zip(label_enc.classes_, range(len(label_enc.classes_))))
        self.labels_dic = {y:x for x,y in self.mapping.items()}
        
        # # v_categories = ['TRBV1', 'TRBV10-1', 'TRBV10-2', 'TRBV10-3', 'TRBV11-1', 'TRBV11-2', 'TRBV11-3', 'TRBV12-1', 'TRBV12-2', 'TRBV12-3', 'TRBV12-4', 'TRBV12-5', 'TRBV13', 'TRBV13-1', 'TRBV13-2', 'TRBV13-3', 'TRBV14', 'TRBV15', 'TRBV16', 'TRBV17', 'TRBV18', 'TRBV19', 'TRBV2', 'TRBV20', 'TRBV20-1', 'TRBV23', 'TRBV24', 'TRBV24-1', 'TRBV25-1', 'TRBV26', 'TRBV27', 'TRBV28', 'TRBV29', 'TRBV29-1', 'TRBV3', 'TRBV3-1', 'TRBV30', 'TRBV31', 'TRBV4', 'TRBV4-1', 'TRBV4-2', 'TRBV4-3', 'TRBV5', 'TRBV5-1', 'TRBV5-4', 'TRBV5-5', 'TRBV5-6', 'TRBV5-8', 'TRBV6-1', 'TRBV6-2', 'TRBV6-3', 'TRBV6-4', 'TRBV6-5', 'TRBV6-6', 'TRBV6-9', 'TRBV7-2', 'TRBV7-3', 'TRBV7-4', 'TRBV7-6', 'TRBV7-7', 'TRBV7-8', 'TRBV7-9', 'TRBV9']
        # v_gene_enc = preprocessing.LabelEncoder()
        # df['V'] = v_gene_enc.fit_transform(df['V'].astype(str))
        # self.v_gene_mapping = dict(zip(v_gene_enc.classes_, range(len(v_gene_enc.classes_))))
        # self.v_gene_dic = {y:x for x,y in self.v_gene_mapping.items()}

        # # j_categories = ['TRBJ1-1', 'TRBJ1-2', 'TRBJ1-3', 'TRBJ1-4', 'TRBJ1-5', 'TRBJ1-6', 'TRBJ2-1', 'TRBJ2-2', 'TRBJ2-3', 'TRBJ2-4', 'TRBJ2-5', 'TRBJ2-6', 'TRBJ2-7']
        # j_gene_enc = preprocessing.LabelEncoder()
        # df['J'] = j_gene_enc.fit_transform(df['J'].astype(str))
        # self.j_gene_mapping = dict(zip(j_gene_enc.classes_, range(len(j_gene_enc.classes_))))
        # self.j_gene_dic = {y:x for x,y in self.j_gene_mapping.items()}

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

    # def __getitem__(self, idx):
    #     if torch.is_tensor(idx):
    #         idx = idx.tolist()
    #     seq = " ".join("".join(self.seqs[idx].split()))
    #     seq = re.sub(r"[UZOB]", "X", seq)
    #     seq_ids = self.tokenizer(seq, truncation=True, padding='max_length', max_length=self.max_length)
    #     sample = {key: torch.tensor(val) for key, val in seq_ids.items()}
    #     sample['labels'] = torch.tensor(self.labels[idx])
    #     return sample

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        seq = " ".join("".join(self.seqs[idx].split()))
        seq = re.sub(r"[UZOB]", "X", seq)
        seq_ids = self.tokenizer(seq, truncation=True, padding='max_length', max_length=self.max_length)
        sample = {key: torch.tensor(val) for key, val in seq_ids.items()}
        sample['labels'] = torch.tensor(self.labels[idx])
        sample['vgenes'] = torch.tensor(self.vgenes[idx])
        sample['jgenes'] = torch.tensor(self.jgenes[idx])
        return sample 

    # def __getitem__(self, idx):
    #     if torch.is_tensor(idx):
    #         idx = idx.tolist()
    #     seq = " ".join("".join(self.seqs[idx].split()))
    #     seq = re.sub(r"[UZOB]", "X", seq)
    #     length = len(self.seqs[idx]) + 2
    #     plength = len(self.seqs[idx])
    #     seq_ids = self.tokenizer(seq, truncation=True, padding='max_length', max_length=self.max_length)
    #     sample = {key: torch.tensor(val) for key, val in seq_ids.items()}
    #     sample['labels'] = torch.tensor(self.labels[idx])
        
    #     v = [0]
    #     v += [int(self.vgenes[idx])] * plength
    #     v +=[0]
    #     v += [64] * (self.max_length - length)
    #     sample['vgenes'] = torch.tensor(v)
        
    #     j = [0]
    #     j += [int(self.jgenes[idx])] * plength
    #     j += [0]
    #     j += [14] * (self.max_length - length)
    #     sample['jgenes'] = torch.tensor(j)
    #     return sample

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
                    if "bert" in name:
                         wname = "%s.%s" % (name, "weight")
                         experiment.log_histogram_3d(to_numpy(layer.embeddings.word_embeddings.weight), name=wname, step=state.global_step, epoch=math.ceil(state.epoch))
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
    # precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    # precision_1, recall_1, f1_1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=1)
    # precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    # precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=0)
    # Macro average gives relation of individual metrics as compared to all the classes.
    # precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=1)
    
    # dic = {
    #     'accuracy': acc,
    #     'f1': f1,
    #     'precision': precision,
    #     'recall': recall,
    #     'epoch': e,
    # }
    
    # experiment.log_metrics(dic=dic, epoch=e, prefix='eval')

    # intermediate_dictionary = {'y_true':y_true, 'y_pred':y_pred}
    # chi = pd.DataFrame(intermediate_dictionary)
    # experiment.log_dataframe_profile(chi, 'truevspred', step=e)
    # name = experiment.get_name()
    # chi.to_csv('/content/logs/cm_{0}_{1}.csv'.format(name, e), encoding='utf-8', index = False)


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
    plt.savefig('/tudelft.net/staff-umbrella/tcr/md/roc.png', bbox_inches = 'tight')
    experiment.log_image('/tudelft.net/staff-umbrella/tcr/md/roc.png', name='roc', overwrite=False, image_format="png",
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
    plt.savefig('/tudelft.net/staff-umbrella/tcr/md/prc.png', bbox_inches = 'tight')
    experiment.log_image('/tudelft.net/staff-umbrella/tcr/md/prc.png', name='prc', overwrite=False, image_format="png",
                    image_scale=1.0, image_shape=None, image_colormap=None,
                    image_minmax=None, image_channels="last", copy_to_tmp=True, step=e)
    plt.close()

    return {
        'accuracy': acc,
        'f1': report['weighted avg']['f1-score']
    }

train_dataset = ProtTrans(split="train", tokenizer_name=model_name, max_length=40)
val_dataset = ProtTrans(split="valid", tokenizer_name=model_name, max_length=40)
test_dataset = ProtTrans(split="test", tokenizer_name=model_name, max_length=40)
label_names = list(train_dataset.labels_dic.values())

# def objective(trial: optuna.Trial):
experiment = comet_ml.Experiment(    
    api_key="",
    project_name="bert-tuner-1-SO",
    workspace=""
    )
hyper_parameters = {
    "seed": 68,
    "learning_rate" : 1.633225074779523e-05,
    "gradient_acm": 25,
    "hidden_dropout_prob": 0.0,
    "classifier_dropout": 0.30000000000000004,
    "attention_probs_dropout_prob": 0.1,
    "weight_decay" : 0.01,
    "warmup_ratio" : 0.1,
    "adam_beta1": 0.72,
    "adam_beta2": 0.573,
    "gamma": 2.0
    }


experiment.log_parameters(hyper_parameters)
experiment.add_tags([
                    'weight_decay_{}'.format(hyper_parameters["weight_decay"]), 
                    'le_rate_{}'.format(hyper_parameters["learning_rate"]), 
                    'gradient_accumulation_steps_{}'.format(hyper_parameters["gradient_acm"]),
                    'seed_{}'.format(hyper_parameters["seed"]),
                    'gamma_{}'.format(hyper_parameters["gamma"]),
                    'hidden_dropout_prob_{}'.format(hyper_parameters["hidden_dropout_prob"]),
                    'attention_probs_dropout_prob_{}'.format(hyper_parameters["attention_probs_dropout_prob"]),
                    'classifier_dropout_{}'.format(hyper_parameters["classifier_dropout"]),
                    'adam_beta1_{}'.format(hyper_parameters["adam_beta1"]),
                    'adam_beta2_{}'.format(hyper_parameters["adam_beta2"]),
                    'warmup_ratio_{}'.format(hyper_parameters["warmup_ratio"])
                        ])

training_args = TrainingArguments(  
    output_dir='/tudelft.net/staff-umbrella/tcr/md/results',
    logging_dir='/tudelft.net/staff-umbrella/tcr/md/logs',
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
    gradient_checkpointing=True,           
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
        'method': 1
        }
    model = BertForTCRClassification.from_pretrained(model_name, 
                                                        num_labels=train_dataset.num_labels,
                                                        id2label=train_dataset.labels_dic, 
                                                        label2id=train_dataset.mapping, 
                                                        hidden_dropout_prob=hyper_parameters["hidden_dropout_prob"], 
                                                        classifier_dropout=hyper_parameters["classifier_dropout"], 
                                                        attention_probs_dropout_prob=hyper_parameters["attention_probs_dropout_prob"], 
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
    shutil.rmtree('/tudelft.net/staff-umbrella/tcr/md/results')
    shutil.rmtree('/tudelft.net/staff-umbrella/tcr/md/logs')
except OSError as er:
    print ("Error CLEANING!: %s - %s." % (er.filename, er.strerror))
#     global e
#     e = 0
#     return results['eval_f1']

# study_name = "bert-tuner-1-SO"  # Unique identifier of the study.
# storage_name = "sqlite:///{}.db".format(study_name)

# study = optuna.create_study(study_name=study_name, storage=storage_name, directions=["maximize"] , load_if_exists=True)
# study.enqueue_trial(
# {
#     "seed": 40,
#     "learning_rate" : 2.054324446856054e-05,
#     "gradient_acm": 38,
#     "hidden_dropout_prob": 0.0,
#     "classifier_dropout": 0.5,
#     "attention_probs_dropout_prob":0.1,
#     "weight_decay" : 0.01,
#     "adam_beta1": 0.8500000000000001,
#     "adam_beta2": 0.515,
#     "gamma": 5,
#     "warmup_ratio": 0.20
#     }
# )
# study.optimize(objective, n_trials = 4)
# print("Number of finished trials: ", len(study.trials))