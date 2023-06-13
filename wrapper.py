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
