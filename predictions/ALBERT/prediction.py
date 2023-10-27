import math
import os
import re
import shutil

import comet_ml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from comet_ml import get_global_experiment
from matplotlib import colors as mcolors
from matplotlib.pyplot import cm
from packaging import version
from sklearn import preprocessing
from sklearn.metrics import classification_report, balanced_accuracy_score, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from torch import nn
from torch.nn import BCEWithLogitsLoss, MSELoss
from torch.utils.data import Dataset
from transformers import AlbertModel, AlbertPreTrainedModel, AlbertTokenizer, Trainer, TrainingArguments, \
    TrainerCallback, EarlyStoppingCallback
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.albert.configuration_albert import AlbertConfig
from transformers.models.albert.modeling_albert import AlbertTransformer
from transformers.optimization import AdamW
from transformers.trainer_pt_utils import get_parameter_names
from transformers.trainer_utils import ShardedDDPOption

# ep = 0 (Baseline) / 1 (Classification) / 2 (Embedding)
ep = 1
# Update model name and path when changing models
model_name = 'models/ALBERT/prot_albert/'
model_path = 'models/ALBERT/md_al/model/'
# path where to save results
path = 'tcrformers/predict/ALBERT'


class AlbertTCREmbeddings(nn.Module):
    """
    Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)
        self.v_gene_embeddings = nn.Embedding(65, config.embedding_size, padding_idx=64)
        self.j_gene_embeddings = nn.Embedding(15, config.embedding_size, padding_idx=14)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer("token_type_ids",
                torch.zeros(self.position_ids.size(), dtype=torch.long, device=self.position_ids.device),
                persistent=False, )

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.forward
    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, v_gene_ids=None,
            j_gene_ids=None, past_key_values_length=0):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length: seq_length + past_key_values_length]

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

        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
class AlbertTCRModel(AlbertPreTrainedModel):
    config_class = AlbertConfig
    base_model_prefix = "albert"

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)

        self.config = config
        self.embeddings = AlbertTCREmbeddings(config)
        self.encoder = AlbertTransformer(config)
        if add_pooling_layer:
            self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
            self.pooler_activation = nn.Tanh()
        else:
            self.pooler = None
            self.pooler_activation = None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} ALBERT has
        a different architecture in that its layers are shared across groups, which then has inner groups. If an ALBERT
        model has 12 hidden layers and 2 hidden groups, with two inner groups, there is a total of 4 different layers.
        These layers are flattened: the indices [0,1] correspond to the two inner groups of the first hidden layer,
        while [2,3] correspond to the two inner groups of the second hidden layer.
        Any layer with in index other than [0,1,2,3] will result in an error. See base class PreTrainedModel for more
        information about head pruning
        """
        for layer, heads in heads_to_prune.items():
            group_idx = int(layer / self.config.inner_group_num)
            inner_group_idx = int(layer - group_idx * self.config.inner_group_num)
            self.encoder.albert_layer_groups[group_idx].albert_layers[inner_group_idx].attention.prune_heads(heads)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
            vgenes=None, jgenes=None, inputs_embeds=None, output_attentions=None, output_hidden_states=None,
            return_dict=None, ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
            v_gene_ids=vgenes, j_gene_ids=jgenes, inputs_embeds=inputs_embeds)

        encoder_outputs = self.encoder(embedding_output, extended_attention_mask, head_mask=head_mask,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, )

        sequence_output = encoder_outputs[0]

        pooled_output = self.pooler_activation(self.pooler(sequence_output[:, 0])) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(last_hidden_state=sequence_output, pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions, )

class AlbertForTCRClassification(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.alpha = 1
        self.method = config.task_specific_params['method']
        self.gamma = config.task_specific_params['gamma']

        if self.method == 1:
            # add two more size to include two additional gene features
            self.albert = AlbertModel(config)
            self.dropout = nn.Dropout(config.classifier_dropout_prob)
            self.classifier = nn.Bilinear(config.hidden_size, 2, config.num_labels)
        elif self.method == 2:
            self.albert = AlbertTCRModel(config)
            self.dropout = nn.Dropout(config.classifier_dropout_prob)
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        else:
            self.albert = AlbertModel(config)
            self.dropout = nn.Dropout(config.classifier_dropout_prob)
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # self.classifier2 = nn.Linear(config.num_labels+2, config.num_labels)
        # self.init_weights()
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
            vgenes=None, jgenes=None, inputs_embeds=None, labels=None, output_attentions=None,
            output_hidden_states=None, return_dict=None, ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.method == 2:
            outputs = self.albert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                position_ids=position_ids, head_mask=head_mask, vgenes=vgenes, jgenes=jgenes,
                inputs_embeds=inputs_embeds, output_attentions=output_attentions,
                output_hidden_states=output_hidden_states, return_dict=return_dict, )
        else:
            outputs = self.albert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,
                output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                return_dict=return_dict, )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        # Size([32, 1024])
        if self.method == 1:
            # Additional Features gets concatenated to Pooled Output from Transformers
            if vgenes is not None:
                vgene = vgenes
                # Size ([32])
                vgene = vgene.unsqueeze(dim=1)  # Size ([32,1])
            if jgenes is not None:
                jgene = jgenes
                # Size ([32])
                jgene = jgene.unsqueeze(dim=1)  # Size ([32,1])
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
                loss = torch.nn.functional.cross_entropy(input=logits.view(-1, self.num_labels), target=labels.view(-1),
                                                         reduction='none')
                pt = torch.exp(-loss)
                loss = (self.alpha * (
                            1 - pt) ** self.gamma * loss).mean()  # loss = focal_loss(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,
            attentions=outputs.attentions, )

class ProtTrans(Dataset):
    """
    Setup for ProtTrans finetune with VDJdb dataset
    """

    def __init__(self, split="train", tokenizer_name='Rostlab/prot_albert', max_length=100):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.datasetFolderPath = 'data/'
        self.FilePath = os.path.join(self.datasetFolderPath, 'beta_only.csv')
        # self.trainFilePath = os.path.join(self.datasetFolderPath, 'train_beta_large.csv')
        # self.testFilePath = os.path.join(self.datasetFolderPath, 'test_beta_large.csv')
        self.tokenizer = AlbertTokenizer.from_pretrained(model_name, do_lower_case=False)
        self.max_length = max_length
        # changes
        df = pd.read_csv(self.FilePath,
                         names=['complex.id', 'Gene', 'CDR3', 'V', 'J', 'Species', 'MHC A', 'MHC B', 'MHC class',
                                'Epitope', 'Epitope gene', 'Epitope species', 'Score', 'labels'], skiprows=1)

        label_enc = preprocessing.LabelEncoder()
        df['labels'] = label_enc.fit_transform(df['labels'].astype(str))
        self.mapping = dict(zip(label_enc.classes_, range(len(label_enc.classes_))))
        self.labels_dic = {y: x for x, y in self.mapping.items()}

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

        cate = [
            ['NA','TRBV1', 'TRBV10-1', 'TRBV10-2', 'TRBV10-3', 'TRBV11-1', 'TRBV11-2', 'TRBV11-3', 'TRBV12-1', 'TRBV12-2', 'TRBV12-3', 'TRBV12-4', 'TRBV12-5', 'TRBV13', 'TRBV13-1', 'TRBV13-2', 'TRBV13-3', 'TRBV14', 'TRBV15', 'TRBV16', 'TRBV17', 'TRBV18', 'TRBV19', 'TRBV2', 'TRBV20', 'TRBV20-1', 'TRBV23', 'TRBV24', 'TRBV24-1', 'TRBV25-1', 'TRBV26', 'TRBV27', 'TRBV28', 'TRBV29', 'TRBV29-1', 'TRBV3', 'TRBV3-1', 'TRBV30', 'TRBV31', 'TRBV4', 'TRBV4-1', 'TRBV4-2', 'TRBV4-3', 'TRBV5', 'TRBV5-1', 'TRBV5-4', 'TRBV5-5', 'TRBV5-6', 'TRBV5-8', 'TRBV6-1', 'TRBV6-2', 'TRBV6-3', 'TRBV6-4', 'TRBV6-5', 'TRBV6-6', 'TRBV6-9', 'TRBV7-2', 'TRBV7-3', 'TRBV7-4', 'TRBV7-6', 'TRBV7-7', 'TRBV7-8', 'TRBV7-9', 'TRBV9'],
            ['NA','TRBJ1-1', 'TRBJ1-2', 'TRBJ1-3', 'TRBJ1-4', 'TRBJ1-5', 'TRBJ1-6', 'TRBJ2-1', 'TRBJ2-2', 'TRBJ2-3', 'TRBJ2-4', 'TRBJ2-5', 'TRBJ2-6', 'TRBJ2-7'],
        ]
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
        val, test = train_test_split(test, stratify=test['labels'],
                                     test_size=test_ratio / (test_ratio + validation_ratio), random_state=44)
        true = pd.read_csv('data/true_data_IMMREP.csv', names=['CDR3', 'V', 'J', 'labels'], skiprows=1)
        true['labels'] = label_enc.transform(true['labels'].astype(str))
        true[["V", "J"]] = enc.transform(true[["V", "J"]])
        self.num_labels = self.labels_dic.__len__()

        if split == "train":
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

        if split == "true":
            self.labels_dic = {0: 'CMV IE2 NEGVKAAW', 1: 'CMV pp65 IPSINVHHY', 2: 'CMV pp65 NLVPMVATV', 3: 'CMV pp65 TPRVTGGGAM', 4: 'DENV1 NS3 GTSGSPIVNR', 5: 'EBV BMLF1 GLCTLVAML', 6: 'EBV BRLF1 YVLDHLIVV', 7: 'EBV BZLF1 RAKFKQLL', 8: 'HCV NS3 ATDALMTGY', 9: 'HCV NS3 KLVALGINAV', 10: 'HIV-1 Gag EIYKRWII', 11: 'HIV-1 Gag FRDYVDRFYKTLRAEQASQE', 12: 'HIV-1 Gag GPGHKARVL', 13: 'HIV-1 Gag KAFSPEVIPMF', 14: 'HIV-1 Gag KRWIILGLNK', 15: 'HIV-1 Nef FLKEKGGL', 16: 'InfluenzaA HA PKYVKQNTLKLAT', 17: 'InfluenzaA M GILGFVFTL', 18: 'InfluenzaA NP ASNENMETM', 19: 'InfluenzaA PA SSLENFRAYV', 20: 'InfluenzaA PB1 LSLRNPILV', 21: 'InfluenzaA PB1 SSYRRPVGI', 22: 'MCMV M38 SSPPMFRV', 23: 'MCMV M45 HGIRNASFI', 24: 'MCMV m139 TVYGFCLL'}
            seqs = list(true['CDR3'])
            vgenes = list(true['V'])
            jgenes = list(true['J'])
            labels = list(true['labels'])

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

def compute_metrics(test_results):
    experiment = get_global_experiment()

    y_true = test_results.label_ids
    y_pred = test_results.predictions.argmax(-1)
    labels = list(true_dataset.labels_dic.values())
    y_scores = test_results.predictions
    y_scores = torch.nn.functional.softmax(torch.from_numpy(y_scores).float())

    # Confusion Matrix
    conf = experiment.create_confusion_matrix(labels=list(true_dataset.labels_dic.values()),
                                              max_categories=true_dataset.num_labels)
    conf.compute_matrix(y_true=y_true, y_predicted=y_pred)
    experiment.log_confusion_matrix(matrix=conf, file_name="confusion-matrix-0.json")

    # Evaluation Metrics
    acc = balanced_accuracy_score(y_true, y_pred)
    # acc_unbalanced = accuracy_score(y_true, y_pred)

    # code to get colors
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                    for name, color in colors.items())
    sorted_names = [name for hsv, name in by_hsv]
    from itertools import cycle
    colors = cycle(np.random.choice(sorted_names, len(labels), replace=True))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))

    # if e == 10:
    nclasses = list(range(0, true_dataset.num_labels))
    y_t = label_binarize(y_true, classes=nclasses)
    y_p = y_scores
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    auc_precision_recall = dict()
    prec = dict()
    recc = dict()
    lw = 2
    for i in range(len(nclasses)):
        prec[i], recc[i], _ = precision_recall_curve(y_t[:, i], y_p[:, i])
        fpr[i], tpr[i], _ = roc_curve(y_t[:, i], y_p[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        auc_precision_recall[i] = auc(recc[i], prec[i])


    for i, color in zip(range(len(nclasses)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label='{0} (area = {1:0.2f})'
                                                          ''.format(true_dataset.labels_dic[i], roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.savefig('tcrformers/predict/ALBERT/roc_1.png', bbox_inches='tight')
    experiment.log_image('tcrformers/predict/ALBERT/roc_1.png', name='roc', overwrite=False,
                         image_format="png",
                         image_scale=1.0, image_shape=None, image_colormap=None,
                         image_minmax=None, image_channels="last", copy_to_tmp=True, step=1)
    plt.close()

    for i, color in zip(range(len(nclasses)), colors):
        plt.plot(prec[i], recc[i], color=color, lw=2, label='{0} (area = {1:0.2f})'
                                                            ''.format(true_dataset.labels_dic[i],
                                                                      auc_precision_recall[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.savefig('tcrformers/predict/ALBERT/prc_1.png', bbox_inches='tight')
    experiment.log_image('tcrformers/predict/ALBERT/prc_1.png', name='prc', overwrite=False,
                         image_format="png",
                         image_scale=1.0, image_shape=None, image_colormap=None,
                         image_minmax=None, image_channels="last", copy_to_tmp=True, step=1)
    plt.close()

true_dataset = ProtTrans(split="true", tokenizer_name=model_name, max_length=40)
label_names = list(true_dataset.labels_dic.values())

# def objective(trial: optuna.Trial):
# def objective(trial: optuna.Trial):
experiment = comet_ml.Experiment(
    api_key="",
    project_name="",
    workspace=""
)

experiment.add_tags([
    'model_ALBERT_1'
])

model = AlbertForTCRClassification.from_pretrained(model_path)

trainer = Trainer(model)

test_results = trainer.predict(true_dataset)
compute_metrics(test_results)
experiment.end()

torch.cuda.empty_cache()