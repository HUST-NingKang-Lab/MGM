import torch
import torch.nn as nn
from transformers import (
    BertForSequenceClassification,
    PreTrainedTokenizer,
    BertConfig,
    TrainingArguments,
    Trainer,
)
from transformers.modeling_outputs import SequenceClassifierOutput

import numpy as np
from typing import Optional, List, Dict, Union, Tuple

class MultiLabelBertForSequenceClassification(BertForSequenceClassification):
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss_fct = nn.BCEWithLogitsLoss()
        if labels is not None:
            labels = labels.float()
            loss = loss_fct(logits, labels)
            outputs.loss = loss
        return outputs
    
class MultiLabelTrainer(Trainer):   
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        labels = labels.float()
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss