import os
import whisper
import torch
from whisper.model import ResidualAttentionBlock
from typing import List, Optional, Union
from utils import mark_only_adapter_as_trainable
from adapters import HoulsbyAdapter
from torch import Tensor, nn

class AdapterResidualAttentionBlock(ResidualAttentionBlock):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__(n_state, n_head)
        self.adapter = HoulsbyAdapter(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        adapter_output = self.adapter(x)
        x = x + self.mlp(self.mlp_ln(x)) + adapter_output
        return x


def load_adapter_model(
    name: str,
    device: Optional[Union[str, torch.device]] = None,
    download_root: str = None,
    in_memory: bool = False,
) -> whisper.Whisper:
    """
    Load a Whisper ASR model

    Parameters
    ----------Yt
    name : str
        one of the official model names listed by `whisper.available_models()`, or
        path to a model checkpoint containing the model dimensions and the model state_dict.
    device : Union[str, torch.device]
        the PyTorch device to put the model into
    download_root: str
        path to download the model files; by default, it uses "~/.cache/whisper"
    in_memory: bool
        whether to preload the model weights into host memory

    Returns
    -------
    model : Whisper
        The Whisper ASR model instance
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if download_root is None:
        default = os.path.join(os.path.expanduser("~"), ".cache")
        download_root = os.path.join(os.getenv("XDG_CACHE_HOME", default), "whisper")

    if name in whisper._MODELS:
        checkpoint_file = whisper._download(whisper._MODELS[name], download_root, in_memory)
        alignment_heads = whisper._ALIGNMENT_HEADS[name]
    elif os.path.isfile(name):
        checkpoint_file = open(name, "rb").read() if in_memory else name
        alignment_heads = None
    else:
        raise RuntimeError(
            f"Model {name} not found; available models = {available_models()}"
        )

    with (
        io.BytesIO(checkpoint_file) if in_memory else open(checkpoint_file, "rb")
    ) as fp:
        checkpoint = torch.load(fp, map_location=device)
    del checkpoint_file

    dims = whisper.ModelDimensions(**checkpoint["dims"])
    model = whisper.Whisper(dims)
    #print(model.dims)
   
    for id in range(len(model.encoder.blocks)):
      model.encoder.blocks[id] = AdapterResidualAttentionBlock(n_state = dims.n_audio_state, n_head = dims.n_audio_head)
    model.load_state_dict(checkpoint["model_state_dict"], strict = False)

    if alignment_heads is not None:
        model.set_alignment_heads(alignment_heads)
    mark_only_adapter_as_trainable(model)

    return model.to(device)
