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
from .modules import BlockEncoder, TextLocalEncoder, AltBlock
from .modules import D2vDecoderConfig, Decoder1d
from .contextualized_features import contextualized_features, get_alibi_bias, _learned_alibi_bias
from fairseq.tasks import FairseqTask

MaskSeed = namedtuple("MaskSeed", ["seed", "update", "ids"])
MaskInfo = namedtuple("MaskInfo", ["x_unmasked", "mask", "ids_restore", "ids_keep"])

@dataclass
class AudioTextFrontendConfig:
    # type: Modality = Modality.AUDIO
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
    conv_pos_pre_ln: bool = (False,)
    ### text frontend
    max_source_positions: int = 512
    learned_pos: bool = True
    dropout: float = 0.1  # used for both local_encoder and contextualized encoder. tied with global transformer in data2vec_text

    no_scale_embedding: bool = True
    layernorm_embedding: bool = True
    no_token_positional_embeddings: bool = False
 
    num_extra_tokens: int = 0
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


    ema_local_encoder: bool = False  # used in data2vec_multi
    local_grad_mult: float = 1.0

    use_alibi_encoder: bool = False
    learned_alibi: bool = False
    alibi_max_pos: Optional[int] = None
    learned_alibi_scale: bool = False
    learned_alibi_scale_per_head: bool = False
    learned_alibi_scale_per_layer: bool = False

    num_alibi_heads: int = II("model.num_heads")
    model_depth: int = II("model.depth")
    clone_batch: int = 1
    audiodecoder: Optional[D2vDecoderConfig] = D2vDecoderConfig()
    textdecoder: Optional[D2vDecoderConfig] = D2vDecoderConfig()


@dataclass
class ContextualizedFeaturesConfig:
    mask_noise_std: float = 0.01
    mask_prob_min: Optional[float] = None
    mask_prob: float = 0.7
    inverse_mask: bool = False
    #mask_prob_adjust: float = 0
    keep_masked_pct: float = 0

    mask_length: int = 5
    add_masks: bool = False
    remove_masks: bool = False
    mask_dropout: float = 0.0
    encoder_zero_mask: bool = True

    mask_channel_prob: float = 0.0
    mask_channel_length: int = 64    
    num_alibi_heads: int = II("model.num_heads")
    alibi_scale: float = 1.0
    prenet_depth: int = 0


class AudioTextFrontend(nn.Module):
    cfg: AudioTextFrontendConfig

    def __init__(
        self,
        cfg: AudioTextFrontendConfig,
        #context_cfg
        embed_dim: int,
        task: Optional[FairseqTask],
    ):
        self.cfg = cfg
        self.feature_enc_layers = eval(self.cfg.feature_encoder_spec)
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

        num_pos_layers = self.cfg.conv_pos_depth
        k = max(3, self.cfg.conv_pos_width // num_pos_layers)

        audiopositional_encoder = nn.Sequential(
            TransposeLast(),
            *[
                nn.Sequential(
                    nn.Conv1d(
                        embed_dim,
                        embed_dim,
                        kernel_size=k,
                        padding=k // 2,
                        groups=self.cfg.conv_pos_groups,
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

        if self.cfg.conv_pos_pre_ln:
            audiopositional_encoder = nn.Sequential(
                LayerNorm(embed_dim), audiopositional_encoder
            )

        ##
        prenetdpr = np.linspace(
            self.cfg.start_drop_path_rate,
            self.cfg.end_drop_path_rate,
            self.cfg.prenet_depth,
        )
        self.prenet = BlockEncoder(
            nn.ModuleList(
                self.make_block(prenetdpr[i]) for i in range(self.cfg.prenet_depth)
            ),
            self.make_layer_norm(embed_dim) if not self.cfg.layer_norm_first else None,
            layer_norm_first,
            self.cfg.prenet_layerdrop,
            self.cfg.prenet_dropout,
        )

        ### text frontend
        self.pad_idx = task.source_dictionary.pad()
        self.vocab_size = len(task.source_dictionary)

        textlocal_encoder = TextLocalEncoder(
            vocab_size=self.vocab_size,
            embed_dim=embed_dim,
            max_source_positions=self.cfg.max_source_positions,
            pad_idx=self.pad_idx,
            no_scale_embedding=self.cfg.no_scale_embedding,
            layernorm_embedding=self.cfg.layernorm_embedding,
            dropout=self.cfg.dropout,
            no_token_positional_embeddings=self.cfg.no_token_positional_embeddings,
            learned_pos=self.cfg.learned_pos,
        )
        self.make_layer_norm = partial(
            nn.LayerNorm, eps=self.cfg.norm_eps, elementwise_affine=self.cfg.norm_affine
        )

        ### decoder part of frontend network
        self.audiodecoder = Decoder1d(self.cfg.audiodecoder, embed_dim)
        self.textdecoder = Decoder1d(self.cfg.textdecoder, embed_dim)
               
        ### ablibi part
        self.get_alibi_bias = get_alibi_bias if self.cfg.use_alibi_encoder else None

        self.local_grad_mult = self.cfg.local_grad_mult

        self.alibi_scale = None
        if self.get_alibi_bias is not None:
            self.alibi_scale = nn.Parameter(
                torch.full(
                    (
                        (self.cfg.prenet_depth + self.cfg.model_depth)
                        if self.cfg.learned_alibi_scale_per_layer
                        else 1,
                        1,
                        self.self.cfg.num_alibi_heads
                        if self.cfg.learned_alibi_scale_per_head
                        else 1,
                        1,
                        1,
                    ),
                    self.cfg.alibi_scale,
                    dtype=torch.float,
                ),
                requires_grad=self.cfg.learned_alibi_scale,
            )
   
        if self.cfg.learned_alibi and self.get_alibi_bias is not None:
            assert self.cfg.alibi_max_pos is not None
            alibi_bias = self.get_alibi_bias(
                batch_size=1,
                time_steps=self.cfg.alibi_max_pos,
                heads=self.cfg.num_alibi_heads,
                scale=1.0,
                dtype=torch.float,
                device="cpu",
            )
            self.alibi_bias = nn.Parameter(alibi_bias)
            self.get_alibi_bias = partial(
                _learned_alibi_bias, alibi_bias=self.alibi_bias
            )

    def upgrade_state_dict_named(self, state_dict, name):
        k = f"{name}.alibi_scale"
        if k in state_dict and state_dict[k].dim() == 4:
            state_dict[k] = state_dict[k].unsqueeze(0)

        return state_dict
    def make_block(drop_path, dim=None, heads=None):
        return AltBlock(
            self.cfg.embed_dim if dim is None else dim,
            self.cfg.num_heads if heads is None else heads,
            self.cfg.mlp_ratio,
            qkv_bias=True,
            drop=self.cfg.encoder_dropout,
            attn_drop=self.cfg.attention_dropout,
            mlp_drop=self.cfg.activation_dropout,
            post_mlp_drop=self.cfg.post_mlp_drop,
            drop_path=drop_path,
            norm_layer=self.make_layer_norm,
            layer_norm_first=self.cfg.layer_norm_first,
            ffn_targets=not self.cfg.end_of_block_targets,
        )
    def audiodecoder_input(self, x, mask_info: MaskInfo):
        inp_drop = self.cfg.audiodecoder.input_dropout
        if inp_drop > 0:
            x = F.dropout(x, inp_drop, training=self.training, inplace=True)
        num_extra = self.cfg.num_extra_tokens
        if mask_info is not None:
            num_masked = mask_info.ids_restore.shape[1] - x.shape[1] + num_extra

            mask_tokens = x.new_empty(
                x.size(0),
                num_masked,
                x.size(-1),
            ).normal_(0, self.modality_cfg.mask_noise_std)

            x_ = torch.cat([x[:, num_extra:], mask_tokens], dim=1)
            x = torch.gather(x_, dim=1, index=mask_info.ids_restore)
            ## note: can't use positions_masked
            #if self.cfg.audiodecoder.add_positions_masked:
            #    assert self.fixed_positional_encoder is not None
            #    pos = self.fixed_positional_encoder(x, None)
            #    x = x + (pos * mask_info.mask.unsqueeze(-1))
        else:
            x = x[:, num_extra:]
            #
        #if self.cfg.decoder.add_positions_all:
        #    assert self.fixed_positional_encoder is not None
        #    x = x + self.fixed_positional_encoder(x, None)
      
        return x, mask_info

    def textdecoder_input(self, x, mask_info: MaskInfo):
        inp_drop = self.cfg.textdecoder.input_dropout
        if inp_drop > 0:
            x = F.dropout(x, inp_drop, training=self.training, inplace=True)
        num_extra = self.cfg.num_extra_tokens
        if mask_info is not None:
            num_masked = mask_info.ids_restore.shape[1] - x.shape[1] + num_extra

            mask_tokens = x.new_empty(
                x.size(0),
                num_masked,
                x.size(-1),
            ).normal_(0, self.modality_cfg.mask_noise_std)

            x_ = torch.cat([x[:, num_extra:], mask_tokens], dim=1)
            x = torch.gather(x_, dim=1, index=mask_info.ids_restore)
            ## note: can't use positions_masked
            #if self.cfg.audiodecoder.add_positions_masked:
            #    assert self.fixed_positional_encoder is not None
            #    pos = self.fixed_positional_encoder(x, None)
            #    x = x + (pos * mask_info.mask.unsqueeze(-1))
        else:
            x = x[:, num_extra:]
            #
        #if self.cfg.decoder.add_positions_all:
        #    assert self.fixed_positional_encoder is not None
        #    x = x + self.fixed_positional_encoder(x, None)

        return x, mask_info

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
    
    def forward(
        self,
        audio,
        audio_padding_mask,
        text,
        text_padding_mask,
        mask,
        remove_masked,
        clone_batch: int = 1,
        mask_seeds: Optional[torch.Tensor] = None,
        precomputed_mask=None,
    ):
        audio_feat = self.audiolocal_features(audio)
        audio_frontend = contextualized_features(
            x=audio_feat,
            padding_mask=self.audio_padding_mask,
            convert_padding_mask=self.audioconvert_padding_mask,
            fixed_positional_encoder=None,
            relative_positional_encoder=self.audiopositional_encoder,
            mask_prob=self.cfg.mask_prob,
            mask_prob_min=self.cfg.mask_prob_min,
            inverse_mask=self.cfg.inverse_mask,
            mask_dropout=self.cfg.mask_dropout,
            add_masks=self.cfg.add_masks,
            mask_length=self.cfg.mask_length,
            keep_masked_pct=self.cfg.keep_masked_pct,
            encoder_zero_mask=self.cfg.encoder_zero_mask,
            mask_noise_std=self.cfg.mask_noise_std,
            mask_channel_prob=self.cfg.mask_channel_prob,
            get_alibi_bias=get_alibi_bias,
            num_alibi_heads=self.cfg.num_alibi_heads,
            alibi_scale=self.cfg.alibi_scale,
            mask=mask,
            remove_masked=remove_masked,
            prenet_depth=self.cfg.prenet_depth,
            context_encoder=self.prenet,  ## (TODO) md check,it  maybe function
            clone_batch=clone_batch,  ## multi mask version
            mask_seeds=mask_seed,
            precomputed_mask=precomputed_mask,
        )

        text_feat = self.textlocal_features(text)
        text_frontend = contextualized_features(
            x=text_feat,
            padding_mask=text_padding_mask,  ## in the dataset, it is raw input padding mask
            convert_padding_mask=textconvert_padding_mask,  ##
            fixed_positional_encoder=None,
            relative_positional_encoder=None,
            mask_prob=self.cfg.mask_prob,
            mask_prob_min=self.cfg.mask_prob_min,
            inverse_mask=self.cfg.inverse_mask,
            mask_dropout=self.cfg.mask_dropout,
            add_masks=self.cfg.add_masks,
            mask_length=self.cfg.mask_length,
            keep_masked_pct=self.cfg.keep_masked_pct,
            encoder_zero_mask=self.cfg.encoder_zero_mask,
            mask_noise_std=self.cfg.mask_noise_std,
            mask_channel_prob=self.cfg.mask_channel_prob,
            get_alibi_bias=get_alibi_bias,
            num_alibi_heads=self.cfg.num_alibi_heads,
            alibi_scale=self.cfg.alibi_scale,
            mask=mask,
            remove_masked=remove_masked,
            prenet_depth=self.cfg.prenet_depth,
            context_encoder=self.prenet,  ## (TODO) md check,it  maybe function
            clone_batch=clone_batch,  ## multi mask version
            mask_seeds=mask_seed,
            precomputed_mask=precomputed_mask,
        )

        return audio_frontend, text_frontend

    def reset_parameters(self):
        super().reset_parameters()
        for mod in self.audioproject_features.children():
            if isinstance(mod, nn.Linear):
                mod.reset_parameters()
        # if self.decoder is not None:
        #    self.decoder.reset_parameters()
