#!/usr/bin/env python3
"""
reference: https://github.com/ziqipang/LM4VisualEncoding/blob/main/image_classification/models/llama.py
modified from https://github.com/Lightning-AI/lit-llama/blob/main/lit_llama/model.py
We can easily load and modify lit format LLM.
1. remove RoPE for attention query and key 
2. remove Causal self-attention, because we will combine them the speech pretrain encoder, no need to causal.
3. remove input postion operation

"""


import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing_extensions import Self

#MaskCache = torch.Tensor
#RoPECache = torch.Tensor
#KVCache = Tuple[torch.Tensor, torch.Tensor]



def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


@dataclass
class LLaMAConfig:
    n_layer: int = 32
    n_head: int = 32
    n_embd: int = 4096
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2


    n_layers: int =32 ## 7B  model layers
    first_layer: int = 31 ## select layer
    #norm_eps: float = 1e-5
"""
class LLaMAConfig:
    block_size: int = 2048
    vocab_size: int = 32000
    padded_vocab_size: Optional[int] = None
    n_layer: int = 32
    n_head: int = 32
    n_embd: int = 4096
    dim: int = 512 ## RMSnorm dim
    n_layers: int =32 ## 7B  model layers
    first_layer: int = 31 ## select layer
    def __post_init__(self):
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(self.vocab_size, 64)

    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls(**llama_configs[name])


llama_configs = {
    "7B": dict(n_layer=32, n_head=32, n_embd=4096),
    "13B": dict(n_layer=40, n_head=40, n_embd=5120),
    "30B": dict(n_layer=60, n_head=52, n_embd=6656),
    "65B": dict(n_layer=80, n_head=64, n_embd=8192),
}
"""
class LitRMSNorm(nn.Module): # it is from https://github.com/Lightning-AI/lit-llama/blob/main/lit_llama/model.py
    """Root Mean Square Layer Normalization.

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """

    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: the original RMSNorm paper implementation is not equivalent
        # norm_x = x.norm(2, dim=self.dim, keepdim=True)
        # rms_x = norm_x * d_x ** (-1. / 2)
        # x_normed = x / (rms_x + self.eps)
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.scale * x_normed


class RMSNorm(torch.nn.Module): ## it is from https://github.com/facebookresearch/llama/blob/llama_v1/llama/model.py
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class MLP(nn.Module):
    def __init__(self, config: LLaMAConfig) -> None:
        super().__init__()
        hidden_dim = 4 * config.n_embd
        n_hidden = int(2 * hidden_dim / 3)
        n_hidden = find_multiple(n_hidden, config.multiple_of)

        self.c_fc1 = nn.Linear(config.n_embd, n_hidden, bias=False)
        self.c_fc2 = nn.Linear(config.n_embd, n_hidden, bias=False)
        self.c_proj = nn.Linear(n_hidden, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        x = self.c_proj(x)
        return x


class Attention(nn.Module):
    def __init__(self, config: LLaMAConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        #self.block_size = config.block_size
    def forward(
        self,
        x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size)
        q = q.view(B, T, self.n_head, head_size)
        v = v.view(B, T, self.n_head, head_size)

        k = k.transpose(1, 2)  # (B, nh, T, hs)
        q = q.transpose(1, 2)  # (B, nh, T, hs)
        v = v.transpose(1, 2)  # (B, nh, T, hs)


         # efficient attention using Flash Attention CUDA kernels
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)

        return y


class Block(nn.Module):
    def __init__(self, config: LLaMAConfig) -> None:
        super().__init__()
        self.rms_1 = LitRMSNorm(config.n_embd)
        self.attn = Attention(config)
        self.rms_2 = LitRMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        
        h= self.attn(self.rms_1(x))
        x = x + h
        x = x + self.mlp(self.rms_2(x))
        return x

class LLaMATransformer(nn.Module):
    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.config = config
        self.n_layers = config['n_layers']
        self.first_layer = config['first_layer']
        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.first_layer, self.n_layers):
            self.layers.append(Block(config))

        #self.norm = RMSNorm(config['dim'], eps=config['norm_eps'])
        self.norm = LitRMSNorm(config.n_embd)
        # work-around for PEFT, Huggingface
        self.prepare_inputs_for_generation = None
    #@classmethod
    #def from_name(cls, name: str) -> Self:
    #    return cls(LLaMAConfig.from_name(name)) 
   
    def forward(self, tokens: torch.Tensor):
        bsz, token_num, hidden_dim = tokens.shape
        h = tokens
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)

        #return h.float()
        return h

    def custom_load_state_dict(self, checkpoint, tail=False, strict=False):
        """
        I use OpenLLaMA version, it is from https://github.com/Lightning-AI/lit-llama/blob/main/howto/download_weights.md
        final get lit version weight format , it is as follows:
        >>> a = torch.load("model_hub/OPT-LLM/lit-llama/7b/7B/lit-llama.pth", map_location={'cuda:3': 'cuda:4'})
        >>> for k in a.keys():
        ...     print(k)
        ...
        transformer.wte.weight
        transformer.h.0.attn.c_attn.weight
        transformer.h.0.attn.c_proj.weight
        transformer.h.0.mlp.c_fc1.weight
        transformer.h.0.mlp.c_proj.weight
        transformer.h.0.mlp.c_fc2.weight
        transformer.h.0.rms_1.scale
        transformer.h.0.rms_2.scale
        transformer.h.1.attn.c_attn.weight
        transformer.h.1.attn.c_proj.weight
        transformer.h.1.mlp.c_fc1.weight
        transformer.h.1.mlp.c_proj.weight
        transformer.h.1.mlp.c_fc2.weight
        transformer.h.1.rms_1.scale
        ...
        transformer.h.31.attn.c_attn.weight
        transformer.h.31.attn.c_proj.weight
        transformer.h.31.mlp.c_fc1.weight
        transformer.h.31.mlp.c_proj.weight
        transformer.h.31.mlp.c_fc2.weight
        transformer.h.31.rms_1.scale
        transformer.h.31.rms_2.scale
        transformer.ln_f.scale
        lm_head.weight
        """
        if tail:
            for i in range(self.first_layer,self.n_layers): ##(TODO) md modify more correct name i.e. select layer?
                layer_checkpoint_keys = [k for k in checkpoint.keys() if f'transformer.h.{i}.' in k] ## full name weight key
                layer_checkpoint_keys = [k.replace(f'transformer.h.{i}.', '') for k in layer_checkpoint_keys] # weight key
                layer_checkpoint = {k: checkpoint[f'transformer.h.{i}.{k}'] for k in layer_checkpoint_keys} ## weight
                self.layers[i - self.first_layer].load_state_dict(
                    layer_checkpoint, strict=strict)
        return

if __name__ == "__main__":
   checkpoint_path="model_hub/OPT-LLM/lit-llama/7b/7B/lit-llama.pth"
   checkpoint = torch.load(checkpoint_path,map_location={'cuda:3': 'cuda:4'}) ## two gpus
   name = "7B"
   model = LLaMATransformer.from_name(name)
   print(f"model: {model}")
   model.custom_load_state_dict(checkpoint, tail=True, strict=False)
  
