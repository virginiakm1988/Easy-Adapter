import torch
from torch import nn
from torch.nn import Embedding
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertOutput

from .adapters import (
    ConvAdapter,
    HoulsbyAdapter,
    AdapterBias,
    
)


PEFT_TYPE_MAPPING = {
    'houlsby': Houlsby_Adapter,
    'AdapterBias': AdapterBias,
    'conv_adapter': Conv_Adapter,
}

class AdaptBERTBase(BertOutput):
    """Implementation of Adapter-BERT"""

    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.config = config
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.adapter = PEFT_TYPE_MAPPING[config.adapter]

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        adapter_output = self.adapter(hidden_states)
        hidden_states = self.dropout(hidden_states) + adapter_output
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
