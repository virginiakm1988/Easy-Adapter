# -*- coding: utf-8 -*-
import torch
from typing import Dict


def mark_only_adapter_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'adapter' not in n:
          p.requires_grad = False
        else:
          p.requires_grad = True
    if bias == "none":
      return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    else:
        raise NotImplementedError


def adapter_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'adapter' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'adapter_' in k or 'bias' in k}
    else:
        raise NotImplementedError
