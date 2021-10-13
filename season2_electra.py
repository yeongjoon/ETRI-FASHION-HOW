import logging
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.activations import get_activation
from transformers.models.electra.configuration_electra import ElectraConfig
from transformers.file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import SequenceSummary

from transformers.models.electra.modeling_electra import ElectraPreTrainedModel, ElectraEmbeddings, ElectraClassificationHead, ElectraModel

from image_model import ImageModel

from season2_resnet_model import season2_resnet101

logger = logging.getLogger(__name__)

_CONFIG_FOR_DOC = "ElectraConfig"
_TOKENIZER_FOR_DOC = "ElectraTokenizer"

ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/electra-small-generator",
    "google/electra-base-generator",
    "google/electra-large-generator",
    "google/electra-small-discriminator",
    "google/electra-base-discriminator",
    "google/electra-large-discriminator",
    # See all ELECTRA models at https://huggingface.co/models?filter=electra
]

class ExtractedFeatElectraForSequenceClassification(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.electra = ElectraModel(config)
        # self.classifier = ElectraClassificationHead(config)

        self.extracted_feat_to_hidden = nn.Linear(4096, config.hidden_size)
        self.text_and_extracted_feat_classifier = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size * 2),
            nn.Dropout(config.hidden_dropout_prob),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size * 2, 2)
        )

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        image=None, # 여기 수정함
        extracted_feat=None, # 여기도 수정함
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict,
        )

        sequence_output = discriminator_hidden_states[0]

        cls_vector = sequence_output[:, 0, :]  # take <s> token (equiv. to [CLS])
        mean_extracted_feat = self.extracted_feat_to_hidden(
            torch.mean(extracted_feat, dim=1)# [batch_size, 3, 4096] => [batch_size, 4096] => [batch_size, 768])
        )

        logits = self.text_and_extracted_feat_classifier(torch.cat((cls_vector, mean_extracted_feat), dim=-1)) # [batch_size, 768*2]

        # logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )


class AttentionFeatElectraForSequenceClassification(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.electra = ElectraModel(config)
        # self.classifier = ElectraClassificationHead(config)

        self.query = nn.Linear(config.hidden_size, config.hidden_size) # kqv attention할때 cls token 쪽에 query.
        self.key = nn.Linear(4096, config.hidden_size) # kqv attention 할 때 feature들 쪽에 Key.
        self.value = nn.Linear(4096, config.hidden_size) #kqv attention 할 때 feature들 쪽에 value.

        self.attention_feat_cls_classifier = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size * 4),
            nn.Dropout(config.hidden_dropout_prob),
            nn.GELU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, 2)
        )

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        image=None, # 여기 수정함
        extracted_feat=None, # 여기도 수정함
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict,
        )

        sequence_output = discriminator_hidden_states[0]

        # take <s> token (equiv. to [CLS])
        cls_vector = sequence_output[:, 0, :]

        cls_tmp = torch.unsqueeze(sequence_output[:, 0, :], dim=2)  # reshape it into [batch_size, 768, 1]

        cls_query = self.query(cls_vector.unsqueeze(1)) #[batch_size, 1, 768]
        feat_key = self.key(extracted_feat) # [batch_size, 3, 768]
        feat_value = self.value(extracted_feat) # [batch_size, 3, 768]

        e = torch.bmm(feat_key, cls_query.transpose(1, 2)) # [batch_size, 3, 1]
        alpha = nn.Softmax(dim=1)(e) # softmax 함수를 거친다.
        feat_final = torch.bmm(feat_value.transpose(1, 2), alpha).squeeze(2) # [batch_size, 768, 3] * [batch_size, 3, 1] => [batch_size, 768]

        logits = self.attention_feat_cls_classifier(torch.cat((cls_vector, feat_final), dim=-1)) # [batch_size, 768*2] -> [batch_size, 2]

        # logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )


class ResnetAttentionElectraForSequenceClassification(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.electra = ElectraModel(config) # 여기 수정
        self.classifier = ElectraClassificationHead(config)
        self.resnet = season2_resnet101()


        self.query = nn.Linear(config.hidden_size, config.hidden_size) # kqv attention할때 cls token 쪽에 query.
        self.key = nn.Linear(2048, config.hidden_size) # kqv attention 할 때 feature들 쪽에 Key.
        self.value = nn.Linear(2048, config.hidden_size) #kqv attention 할 때 feature들 쪽에 value.

        self.attention_feat_cls_classifier = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size * 4),
            nn.Dropout(config.hidden_dropout_prob),
            nn.GELU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, 2)
        )

        # # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        # self.concat_to_hidden = nn.Linear(2048+config.hidden_size, config.hidden_size) # 2048+768 -> 768
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        image=None, # 여기 수정함
        extracted_feat=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 이 부분에서 image와 embedding을 합침
        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict,
        )

        sequence_output = discriminator_hidden_states[0]

        # take <s> token (equiv. to [CLS])
        cls_vector = sequence_output[:, 0, :]

        cls_tmp = torch.unsqueeze(sequence_output[:, 0, :], dim=2)  # reshape it into [batch_size, 768, 1]

        image_feat = self.resnet(image) # [batch_size, 49, 2048]

        cls_query = self.query(cls_vector.unsqueeze(1)) # [batch_size, 1, 768]
        feat_key = self.key(image_feat) # [batch_size, 49, 768]
        feat_value = self.value(image_feat) # [batch_size, 49, 768]

        e = torch.bmm(feat_key, cls_query.transpose(1, 2)) # [batch_size, 49, 1]
        alpha = nn.Softmax(dim=1)(e) # softmax 함수를 거친다.
        feat_final = torch.bmm(feat_value.transpose(1, 2), alpha).squeeze(2) # [batch_size, 768, 3] * [batch_size, 3, 1] => [batch_size, 768]

        logits = self.attention_feat_cls_classifier(torch.cat((cls_vector, feat_final), dim=-1)) # [batch_size, 768*2] -> [batch_size, 2]


        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )
