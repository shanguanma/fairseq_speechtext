#!/usr/bin/env python3
import torch.nn as nn
from functools import partial
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
from fairseq.models.wav2vec import ConvFeatureExtractionModel
from fairseq.modules import (
    LayerNorm,
    SamePad,
    TransposeLast,
)
from fairseq.modules import PositionalEmbedding, FairseqDropout, LayerNorm
from .modules import BlockEncoder, TextLocalEncoder,AltBlock
from .contextualized_features import contextualized_features 
from fairseq.tasks import FairseqTask


@dataclass
class AudioTextConfig:
    #type: Modality = Modality.AUDIO
    ## audio frontend
    extractor_mode: str = "layer_norm"
    feature_encoder_spec: str = field(
        default="[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]",
        metadata={
            "help": "string describing convolutional feature extraction layers in form of a python list that contains "
            "[(dim, kernel_size, stride), ...]"
        },
    )
    ### conv style position encoder for audio
    conv_pos_width: int = field(
        default=95,
        metadata={"help": "number of filters for convolutional positional embeddings"},
    )
    conv_pos_groups: int = field(
        default=16,
        metadata={"help": "number of groups for convolutional positional embedding"},
    )
    conv_pos_depth: int = field(
        default=5,
        metadata={"help": "depth of positional encoder network"},
    )
    conv_pos_pre_ln: bool = False,
    ### text frontend
    max_source_positions: int = 512
    learned_pos: bool = True
    dropout: float = 0.1  # used for both local_encoder and contextualized encoder. tied with global transformer in data2vec_text

    no_scale_embedding: bool = True
    layernorm_embedding: bool = True
    no_token_positional_embeddings: bool = False    
   
    ## transformer part 
    depth: int = 8

    ### attention black  part parameter
    num_heads: int = 12
    norm_eps: float = 1e-6
    norm_affine: bool = True
    encoder_dropout: float = 0.1
    post_mlp_drop: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.0
    end_of_block_targets: bool = False
    mlp_ratio: float = 4
    layer_norm_first: bool = False
    embed_dim: int = 768

    dropout_input: float = 0.0
    layerdrop: float = 0.0

    
    ## for prenet(transformer layer) parameter
    prenet_depth: int = 0
    prenet_layerdrop: float = 0
    prenet_dropout: float = 0
    start_drop_path_rate: float = 0
    end_drop_path_rate: float = 0
     
class AudioTextFrontend(nn.Module):

    cfg: AudioTextConfig
    def __init__(
        self,
        cfg: AudioTextConfig,
        embed_dim: int,
        #make_block: Callable[[float], nn.ModuleList],
        norm_layer: Callable[[int], nn.LayerNorm],
        layer_norm_first: bool,
        alibi_biases: Dict,
        task: Optional[FairseqTask],
    ):

        self.feature_enc_layers = eval(modality_cfg.feature_encoder_spec)
        feature_embed_dim = self.feature_enc_layers[-1][0]

        audiolocal_encoder = ConvFeatureExtractionModel(
            conv_layers=self.feature_enc_layers,
            dropout=0.0,
            mode=modality_cfg.extractor_mode,
            conv_bias=False,
        )
        audioproject_features = nn.Sequential(
            TransposeLast(),
            nn.LayerNorm(feature_embed_dim),
            nn.Linear(feature_embed_dim, embed_dim),
        )

        num_pos_layers = cfg.conv_pos_depth
        k = max(3, cfg.conv_pos_width // num_pos_layers)

        audiopositional_encoder = nn.Sequential(
            TransposeLast(),
            *[
                nn.Sequential(
                    nn.Conv1d(
                        embed_dim,
                        embed_dim,
                        kernel_size=k,
                        padding=k // 2,
                        groups=cfg.conv_pos_groups,
                    ),
                    SamePad(k),
                    TransposeLast(),
                    LayerNorm(embed_dim, elementwise_affine=False),
                    TransposeLast(),
                    nn.GELU(),
                )
                for _ in range(num_pos_layers)
            ],
            TransposeLast(),
        )

        if cfg.conv_pos_pre_ln:
            audiopositional_encoder = nn.Sequential(LayerNorm(embed_dim), audiopositional_encoder)

        ## 
        prenetdpr = np.linspace(cfg.start_drop_path_rate,cfg.end_drop_path_rate,cfg.prenet_depth,
        )
        self.prenet = BlockEncoder(
            nn.ModuleList(self.make_block(prenetdpr[i]) for i in range(cfg.prenet_depth)),
            self.make_layer_norm(embed_dim) if not layer_norm_first else None,
            layer_norm_first,
            cfg.prenet_layerdrop,
            cfg.prenet_dropout,
        )
        
               
        ### text frontend
        self.pad_idx = task.source_dictionary.pad()
        self.vocab_size = len(task.source_dictionary)

        textlocal_encoder = TextLocalEncoder(
            vocab_size=self.vocab_size,
            embed_dim=embed_dim,
            max_source_positions=cfg.max_source_positions,
            pad_idx=self.pad_idx,
            no_scale_embedding=cfg.no_scale_embedding,
            layernorm_embedding=cfg.layernorm_embedding,
            dropout=cfg.dropout,
            no_token_positional_embeddings=cfg.no_token_positional_embeddings,
            learned_pos=cfg.learned_pos,
        )
        self.make_layer_norm = partial(
            nn.LayerNorm, eps=cfg.norm_eps, elementwise_affine=cfg.norm_affine
        )
    def make_block(drop_path, dim=None, heads=None):
        return AltBlock(
            cfg.embed_dim if dim is None else dim,
            cfg.num_heads if heads is None else heads,
            cfg.mlp_ratio,
            qkv_bias=True,
            drop=cfg.encoder_dropout,
            attn_drop=cfg.attention_dropout,
            mlp_drop=cfg.activation_dropout,
            post_mlp_drop=cfg.post_mlp_drop,
            drop_path=drop_path,
            norm_layer=self.make_layer_norm,
            layer_norm_first=cfg.layer_norm_first,
            ffn_targets=not cfg.end_of_block_targets,
        )
    
    def audiolocal_features(self, features):
        if self.local_grad_mult > 0:
            if self.local_grad_mult == 1.0:
                x = self.audiolocal_encoder(features)
            else:
                x = GradMultiply.apply(
                    self.audiolocal_encoder(features), self.local_grad_mult
                )
        else:
            with torch.no_grad():
                x = self.audiolocal_encoder(features)

        x = self.audioproject_features(x)
        return x
    def textlocal_features(self, features):
        if self.local_grad_mult > 0:
            if self.local_grad_mult == 1.0:
                x = self.textlocal_encoder(features)
            else:
                x = GradMultiply.apply(
                    self.audiolocal_encoder(features), self.local_grad_mult
                )
        else:
            with torch.no_grad():
                x = self.audiolocal_encoder(features)

        x = self.audioproject_features(x)
        return x
    

    def textconvert_padding_mask(self, x, padding_mask):
        if padding_mask is None or padding_mask.size(1) == x.size(1):
            return padding_mask

        diff = self.downsample - padding_mask.size(1) % self.downsample
        if 0 < diff < self.downsample:
            padding_mask = F.pad(padding_mask, (0, diff), value=True)

        padding_mask = padding_mask.view(padding_mask.size(0), -1, self.downsample)
        padding_mask = padding_mask.all(-1)
        if padding_mask.size(1) > x.size(1):
            padding_mask = padding_mask[:, : x.size(1)]

        assert x.size(1) == padding_mask.size(
            1
        ), f"{x.size(1), padding_mask.size(1), diff, self.downsample}"

        return padding_mask
    
    def audioconvert_padding_mask(self, x, padding_mask):
        def get_feat_extract_output_lengths(input_lengths: torch.LongTensor):
            """
            Computes the output length of the convolutional layers
            """

            def _conv_out_length(input_length, kernel_size, stride):
                return torch.floor((input_length - kernel_size) / stride + 1)

            for i in range(len(self.feature_enc_layers)):
                input_lengths = _conv_out_length(
                    input_lengths,
                    self.feature_enc_layers[i][1],
                    self.feature_enc_layers[i][2],
                )

            return input_lengths.to(torch.long)
        if padding_mask is not None:
            input_lengths = (1 - padding_mask.long()).sum(-1)
            # apply conv formula to get real output_lengths
            output_lengths = get_feat_extract_output_lengths(input_lengths)

            if padding_mask.any():
                padding_mask = torch.zeros(x.shape[:2], dtype=x.dtype, device=x.device)

                # these two operations makes sure that all values
                # before the output lengths indices are attended to
                padding_mask[
                    (
                        torch.arange(padding_mask.shape[0], device=padding_mask.device),
                        output_lengths - 1,
                    )
                ] = 1
                padding_mask = (
                    1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])
                ).bool()
            else:
                padding_mask = torch.zeros(
                    x.shape[:2], dtype=torch.bool, device=x.device
                )

        return padding_mask
    def forward(self, audio, audio_padding_mask, )
    def reset_parameters(self):
        super().reset_parameters()
        for mod in self.audioproject_features.children():
            if isinstance(mod, nn.Linear):
                mod.reset_parameters()
        #if self.decoder is not None:
        #    self.decoder.reset_parameters()    


