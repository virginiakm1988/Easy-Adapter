from transformers import AutoModelForSequenceClassification
from adapters import HoulsbyAdapter, AdapterBias, ConvAdapter, LoRA
import loralib as lora
from torch import nn
import torch

BertLayerNorm = torch.nn.LayerNorm

##vanilla houlsby residual adapter, custom layers
class adapted_bert_output(nn.Module):
  def __init__(self, BertOutput, config):
    super().__init__()
    self.config = config
    self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)

    if config.adapter == "houlsby":
      self.adapter = HoulsbyAdapter(config.hidden_size)
    elif config.adapter == "conv_adapter":
      self.adapter = ConvAdapter(config.max_position_embeddings)
    elif self.adapter == "AdapterBias":
      self.adapter = AdapterBias(config.hidden_size)
    elif self.adapter == "lora":
      self.adapter = LoRA(config.hidden_size)
    else:
      raise NotImplementedError

  def forward(self,  hidden_states, input_tensor):

    hidden_states = self.dense(hidden_states)
    if self.config.adapter != None:
      adapter_output = self.adapter(hidden_states)
      hidden_states = self.dropout(hidden_states) + adapter_output
    else:
      hidden_states = self.dropout(hidden_states)
    hidden_states = self.LayerNorm(hidden_states + input_tensor)
  
    return hidden_states