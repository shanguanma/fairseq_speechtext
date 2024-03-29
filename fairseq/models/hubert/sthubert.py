# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from torch import Tensor
import torch.nn.functional as F

import numpy as np
import torch
import torch.nn as nn
from omegaconf import II

from fairseq import utils
from fairseq.data.data_utils import compute_mask_indices
from fairseq.data.dictionary import Dictionary
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model
from fairseq.models.wav2vec.wav2vec2 import (
    EXTRACTOR_MODE_CHOICES,
    MASKING_DISTRIBUTION_CHOICES,
    LAYER_TYPE_CHOICES,
    ConvFeatureExtractionModel,
    TransformerEncoder,
    TransformerSentenceEncoderLayer,
    make_conv_pos,
)

from fairseq.modules import GradMultiply, LayerNorm
from fairseq.modules import PositionalEmbedding
from fairseq.tasks.sthubert_pretraining import (
    StHubertPretrainingConfig,
    StHubertPretrainingTask,
)

logger = logging.getLogger(__name__)


@dataclass
class StHubertConfig(FairseqDataclass):
    label_rate: float = II("task.label_rate")

    extractor_mode: EXTRACTOR_MODE_CHOICES = field(
        default="default",
        metadata={
            "help": "mode for feature extractor. default has a single group "
            "norm with d groups in the first conv block, whereas layer_norm "
            "has layer norms in every block (meant to use with normalize=True)"
        },
    )
    encoder_layers: int = field(
        default=12, metadata={"help": "num encoder layers in the transformer"}
    )
    encoder_embed_dim: int = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "encoder embedding dimension for FFN"}
    )
    encoder_attention_heads: int = field(
        default=12, metadata={"help": "num encoder attention heads"}
    )
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="gelu", metadata={"help": "activation function to use"}
    )
    layer_type: LAYER_TYPE_CHOICES = field(
        default="transformer", metadata={"help": "layer type in encoder"}
    )

    # dropouts
    dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for the transformer"},
    )
    attention_dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for attention weights"},
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability after activation in FFN"},
    )
    encoder_layerdrop: float = field(
        default=0.0,
        metadata={"help": "probability of dropping a tarnsformer layer"},
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    dropout_features: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the features (after feat extr)"},
    )

    final_dim: int = field(
        default=0,
        metadata={
            "help": "project final representations and targets to this many "
            "dimensions. set to encoder_embed_dim is <= 0"
        },
    )
    untie_final_proj: bool = field(
        default=False,
        metadata={"help": "use separate projection for each target"},
    )
    layer_norm_first: bool = field(
        default=False,
        metadata={"help": "apply layernorm first in the transformer"},
    )
    conv_feature_layers: str = field(
        default="[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
        metadata={
            "help": "string describing convolutional feature extraction "
            "layers in form of a python list that contains "
            "[(dim, kernel_size, stride), ...]"
        },
    )
    conv_bias: bool = field(
        default=False, metadata={"help": "include bias in conv encoder"}
    )
    logit_temp: float = field(
        default=0.1, metadata={"help": "temperature to divide logits by"}
    )
    target_glu: bool = field(
        default=False, metadata={"help": "adds projection + glu to targets"}
    )
    feature_grad_mult: float = field(
        default=1.0,
        metadata={"help": "multiply feature extractor var grads by this"},
    )

    # masking
    mask_length: int = field(default=10, metadata={"help": "mask length"})
    mask_prob: float = field(
        default=0.65,
        metadata={"help": "probability of replacing a token with mask"},
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose mask length"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    mask_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # channel masking
    mask_channel_length: int = field(
        default=10,
        metadata={"help": "length of the mask for features (channels)"},
    )
    mask_channel_prob: float = field(
        default=0.0,
        metadata={"help": "probability of replacing a feature with 0"},
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False,
        metadata={"help": "whether to allow channel masks to overlap"},
    )
    mask_channel_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # positional embeddings
    conv_pos: int = field(
        default=128,
        metadata={"help": "number of filters for convolutional positional embeddings"},
    )
    conv_pos_groups: int = field(
        default=16,
        metadata={"help": "number of groups for convolutional positional embedding"},
    )

    latent_temp: Tuple[float, float, float] = field(
        default=(2, 0.5, 0.999995),
        metadata={"help": "legacy (to be removed)"},
    )

    # loss computation
    skip_masked: bool = field(
        default=False,
        metadata={"help": "skip computing losses over masked frames"},
    )
    skip_nomask: bool = field(
        default=False,
        metadata={"help": "skip computing losses over unmasked frames"},
    )

    checkpoint_activations: bool = field(
        default=False,
        metadata={"help": "recompute activations and save memory for extra compute"},
    )

    # FP16 optimization
    required_seq_len_multiple: int = field(
        default=2,
        metadata={
            "help": "pad the input to encoder such that the sequence length is divisible by multiple"
        },
    )

    # Conformer
    depthwise_conv_kernel_size: int = field(
        default=31,
        metadata={
            "help": "depthwise-conv-kernel-size for convolution in conformer layer"
        },
    )
    attn_type: str = field(
        default="",
        metadata={"help": "if espnet use ESPNET MHA"},
    )
    pos_enc_type: str = field(
        default="abs",
        metadata={"help": "Positional encoding type to use in conformer"},
    )
    fp16: bool = field(default=False, metadata={"help": "If fp16 is being used"})
    # text net part
    max_source_positions: int = field(
        default=512,
        metadata={"help": "Maximum input length supported by the transformer encoder"},
    )
    text_dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for the TextModel embedding feature"},
    )
    text_embed_dim: int = field(
        default=768, metadata={"help": "text net part embedding dimension"}
    )


@register_model("sthubert", dataclass=StHubertConfig)
class StHubertModel(BaseFairseqModel):
    def __init__(
        self,
        cfg: StHubertConfig,
        task_cfg: StHubertPretrainingConfig,
        dictionaries: List[Dictionary],
    ) -> None:
        super().__init__()
        logger.info(f"StHubertModel Config: {cfg}")
        self.cfg = cfg
        self.padding_idx = 1
        feature_enc_layers = eval(cfg.conv_feature_layers)  # noqa
        self.embed = feature_enc_layers[-1][0]
        ### speech feature net part
        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
        )
        ### text feature net part
        self.embed_tokens = torch.nn.Embedding(
            len(dictionaries[1]), cfg.text_embed_dim, padding_idx=self.padding_idx
        )
        self.embed_positions = PositionalEmbedding(
            cfg.max_source_positions,
            cfg.text_embed_dim,
            padding_idx=self.padding_idx,
            learned=True,
        )
        self.layernorm_embedding = torch.nn.LayerNorm(cfg.text_embed_dim)

        if not task_cfg.fine_tuning:
            ## pretrain case
            feature_ds_rate = np.prod([s for _, _, s in feature_enc_layers])
            self.feat2tar_ratio = (
                cfg.label_rate * feature_ds_rate / task_cfg.sample_rate
            )

        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )
        # modality embedding for distinguishing text and speech
        # if self.post_extract_proj is not None:
        #    self.text_post_extract_proj = torch.nn.Linear(
        #        self.embed, cfg.encoder_embed_dim
        #    )
        # else:
        #    self.text_post_extract_proj = None
        dim = cfg.encoder_embed_dim if self.post_extract_proj else self.embed
        ## prepared two modality bias
        self.text_modality_bias = torch.nn.Parameter(torch.FloatTensor(dim).uniform_())
        self.speech_modality_bias = torch.nn.Parameter(
            torch.FloatTensor(dim).uniform_()
        )

        # modality-specific positional encodings

        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult
        if not task_cfg.fine_tuning:
            # pretrain case
            self.logit_temp = cfg.logit_temp
            self.skip_masked = cfg.skip_masked
            self.skip_nomask = cfg.skip_nomask

        final_dim = cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )

        # modality-specific positional encodings
        ## add releated postion encoding layer for text and speech
        self.speech_pos = make_conv_pos(
            cfg.encoder_embed_dim, cfg.conv_pos, cfg.conv_pos_groups
        )
        self.text_pos = make_conv_pos(
            cfg.text_embed_dim, cfg.conv_pos, cfg.conv_pos_groups
        )  ## (TODO) add text embedding layer and its dimension

        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.embed)
        if not task_cfg.fine_tuning:
            self.target_glu = None
            if cfg.target_glu:
                self.target_glu = nn.Sequential(
                    nn.Linear(final_dim, final_dim * 2), nn.GLU()
                )

            self.untie_final_proj = cfg.untie_final_proj
            if self.untie_final_proj:
                self.final_proj = nn.Linear(
                    cfg.encoder_embed_dim, final_dim * len(dictionaries[0])
                )
            else:
                self.final_proj = nn.Linear(cfg.encoder_embed_dim, final_dim)

        # modules below are not needed during fine-tuning
        if any([d is None for d in dictionaries]):
            logger.info("cannot find dictionary. assume will be used for fine-tuning")
        else:
            self.num_classes = [len(d) for d in dictionaries]
            self.label_embs_concat = nn.Parameter(
                torch.FloatTensor(sum(self.num_classes), final_dim)
            )
            nn.init.uniform_(self.label_embs_concat)

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""

        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: StHubertConfig, task: StHubertPretrainingTask):
        """Build a new model instance."""
        pathss = "/workspace2/maduo/dataset/format/librispeech"
        import os

        logger.info(
            f"dictionary: {task.dictionaries[1].indices.items()}"
        )  # dictionary: dict_items([('<s>', 0), ('<pad>', 1), ('</s>', 2), ('<unk>', 3), ('0', 4), ('1', 5), ('2', 6), ('3', 7), ('4', 8), ('5', 9), ('6', 10), ('7', 11), ('8', 12), ('9', 13), ('10', 14), ('11', 15), ('12', 16), ('13', 17), ('14', 18), ('15', 19), ('16', 20), ('17', 21), ('18', 22), ('19', 23), ('20', 24), ('21', 25), ('22', 26), ('23', 27), ('24', 28), ('25', 29), ('26', 30), ('27', 31), ('28', 32), ('29', 33), ('30', 34), ('31', 35), ('32', 36), ('33', 37), ('34', 38), ('35', 39), ('36', 40), ('37', 41), ('38', 42), ('39', 43), ('40', 44)])
        # with open(os.path.join(pathss, "hubert_iter1_kmdict_model.txt"), "w") as f:
        #    for sym, num in task.dictionaries[0].indices.items():
        #        f.write(f"{sym} {num}\n")
        # with open(os.path.join(pathss, "hubert_iter1_textphndict_model.txt"), "w") as f:
        #    for sym, num in task.dictionaries[1].indices.items():
        #        f.write(f"{sym} {num}\n")
        ## dictionary bos index: 0
        ## dictionary pad index: 1
        ## dictionary eos index: 2
        ## dictionary unk index: 3
        logger.info(f"dictionary bos index: {task.dictionaries[1].bos_index}")
        logger.info(f"dictionary pad index: {task.dictionaries[1].pad_index}")
        logger.info(f"dictionary eos index: {task.dictionaries[1].eos_index}")
        logger.info(f"dictionary unk index: {task.dictionaries[1].unk_index}")
        model = StHubertModel(cfg, task.cfg, task.dictionaries)
        return model

    def apply_mask(self, x, padding_mask, target_list):
        B, T, C = x.shape
        if self.mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_prob,
                self.mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        return x, mask_indices

    def compute_nce(self, x, pos, negs):
        neg_is_pos = (pos == negs).all(-1)
        pos = pos.unsqueeze(0)
        targets = torch.cat([pos, negs], dim=0)

        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1).type_as(x)
        logits /= self.logit_temp
        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")
        logits = logits.transpose(0, 1)  # (num_x, num_cls+1)
        return logits

    def forward_features(self, source: torch.Tensor) -> torch.Tensor:
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)
        return features

    def forward_text(
        self, source_text, source_text_lengths: Optional[torch.Tensor] = None
    ):
        """
        Args:
            source_text (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            source_text_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
        """
        """
        # embed tokens and positions
        #logger.info(f"in the forward_text: source_text: {source_text}")
        x = source_text
        ###  remove full 1 utterance in batch
        x_list = torch.split(source_text,1)
        a = [torch.equal(i,torch.ones(i.size(0),i.size(1),device=source_text.device)) for i in x_list]
        if not torch.any(torch.tensor(a)):
            logger.info(f"in forward_text, source_text:  {source_text}, source_text shape : {source_text.shape}")
            token_embedding = self.embed_tokens(source_text)
            x = embed = token_embedding
            if self.embed_positions is not None:
                x = embed + self.embed_positions(source_text)
            if self.layernorm_embedding is not None:
                x = self.layernorm_embedding(x)
            x = F.dropout(x, self.cfg.text_dropout)
            # compute padding mask
            encoder_padding_mask = source_text.eq(self.padding_idx)
            # account for padding while computing the representation
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))
        """
        # logger.info(f"in forward_text, source_text:  {source_text}, source_text shape : {source_text.shape}")
        token_embedding = self.embed_tokens(source_text)
        x = embed = token_embedding
        # if self.embed_positions is not None:
        #    x = embed + self.embed_positions(source_text)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = F.dropout(x, self.cfg.text_dropout)

        return x

    def forward_targets(
        self,
        features: torch.Tensor,
        target_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Trim features to ensure labels exist and then get aligned labels
        feat_tsz = features.size(2)
        targ_tsz = min([t.size(1) for t in target_list])
        if self.feat2tar_ratio * feat_tsz > targ_tsz:
            feat_tsz = int(targ_tsz / self.feat2tar_ratio)
            features = features[..., :feat_tsz]
        target_inds = torch.arange(feat_tsz).float() * self.feat2tar_ratio
        target_list = [t[:, target_inds.long()] for t in target_list]
        return features, target_list

    def forward_padding_mask(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def forward_frontend(
        self,
        source: torch.Tensor,
        source_text: torch.Tensor,
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        """output layer is 1-based"""
        # logger.info(f"source shape: {source.shape}")
        # logger.info(f"source_text shape: {source_text.shape}")
        features = self.forward_features(source)
        if target_list is not None:
            features, target_list = self.forward_targets(features, target_list)

        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        # unmasked_features = features.clone()

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        # (B,T,D)
        features = self.dropout_input(features)
        # unmasked_features = self.dropout_features(unmasked_features)
        if source_text is None:
            ## only for fine-tuning without text
            return features, None, features_pen, target_list, padding_mask

        ### text part
        features_text = self.forward_text(source_text)
        # logger.info(f"features_text shape: {features_text.shape}")
        return features, features_text, features_pen, target_list, padding_mask

    def forward_transformer(
        self,
        features: torch.Tensor,
        features_text: torch.Tensor,
        features_pen: torch.Tensor,
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        if mask:
            x, mask_indices = self.apply_mask(features, padding_mask, target_list)
        else:
            x = features
            mask_indices = None

        ### text feature and mask speech feature will be cancat on time axis.
        ## (B,T, D),float
        x = self.speech_pos(x.transpose(1, 2)).transpose(1, 2)
        features_concat = x.new_zeros((x.size(0), x.size(1), x.size(2)))
        padding_mask2 = None
        if features_text is None:
            ### only for fine-tuning without text
            features_concat = x + self.speech_modality_bias
            padding_mask2 = padding_mask
        else:
            if features_text.size(1) != 0:
                # logger.info(f"features_text shape : {features_text.shape}")
                x_text = self.text_pos(features_text.transpose(1, 2)).transpose(1, 2)

                # logger.info(f"x_text shape : {x_text.shape}")
                # (B, T' + T, D), float
                x_text_all = x_text + self.text_modality_bias
                x_speech_all = x + self.speech_modality_bias
                # logger.info(f"x_text_all shape : {x_text_all.shape}")
                # logger.info(f"x_speech_all shape : {x_speech_all.shape}")
                features_concat = torch.cat([x_text_all, x_speech_all], dim=1)
                # (B, T' + T), bool
                if padding_mask is not None:
                    padding_mask2 = nn.functional.pad(padding_mask, (x_text.size(1), 0))
                else:
                    padding_mask2 = None
        # feature_concat: (B, T'+T, D), float
        # target: (B, T), long
        # padding_mask2: (B, T'+T), bool
        # mask_indices: (B,T), bool
        x, _ = self.encoder(
            features_concat,
            padding_mask=padding_mask2,
            layer=None if output_layer is None else output_layer - 1,
        )
        x = x[:, -features.size(1) :]
        # logger.info(f"before transformer speech feature shape: {features.shape}")
        # logger.info(f"after transformer output features shape: {x.shape}")
        # print(f"before transformer speech feature shape: {features.shape}")
        # print(f"after transformer output features shape: {x.shape}")
        if features_only:
            return {
                "x": x,
                "padding_mask": padding_mask,
                "features": features,
                "features_text": features_text,
            }

        def compute_pred(proj_x, target, label_embs):
            # compute logits for the i-th label set
            y = torch.index_select(label_embs, 0, target.long())
            negs = label_embs.unsqueeze(1).expand(-1, proj_x.size(0), -1)
            if self.target_glu:
                y = self.target_glu(y)
                negs = self.target_glu(negs)
            # proj_x: (S, D)
            # y: (S, D)
            # negs: (Neg, S, D)
            # print(f"in compute_pred: proj_x shape: {proj_x.shape} , y shape: {y.shape} ")
            return self.compute_nce(proj_x, y, negs)

        label_embs_list = self.label_embs_concat.split(self.num_classes, 0)

        if not self.skip_masked:
            masked_indices = torch.logical_and(~padding_mask, mask_indices)
            proj_x_m = self.final_proj(x[masked_indices])
            if self.untie_final_proj:
                proj_x_m_list = proj_x_m.chunk(len(target_list), dim=-1)
            else:
                proj_x_m_list = [proj_x_m for _ in range(len(target_list))]
            logit_m_list = [
                compute_pred(proj_x_m, t[masked_indices], label_embs_list[i])
                for i, (proj_x_m, t) in enumerate(zip(proj_x_m_list, target_list))
            ]
        else:
            logit_m_list = [None for _ in target_list]

        if not self.skip_nomask:
            nomask_indices = torch.logical_and(~padding_mask, ~mask_indices)
            proj_x_u = self.final_proj(x[nomask_indices])
            if self.untie_final_proj:
                proj_x_u_list = proj_x_u.chunk(len(target_list), dim=-1)
            else:
                proj_x_u_list = [proj_x_u for _ in range(len(target_list))]

            logit_u_list = [
                compute_pred(proj_x_u, t[nomask_indices], label_embs_list[i])
                for i, (proj_x_u, t) in enumerate(zip(proj_x_u_list, target_list))
            ]
        else:
            logit_u_list = [None for _ in target_list]

        result = {
            "logit_m_list": logit_m_list,
            "logit_u_list": logit_u_list,
            "padding_mask": padding_mask,
            "features_pen": features_pen,
        }
        return result

    def forward(
        self,
        source: torch.Tensor,
        source_text: Optional[torch.Tensor] = None,
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """output layer is 1-based"""
        if source_text is None:  ## finetune case
            (
                feats,
                feats_text,
                feats_pen,
                target_list,
                padding_mask,
            ) = self.forward_frontend(
                source,
                source_text=None,
                target_list=target_list,
                padding_mask=padding_mask,
            )
        else:
            (
                feats,
                feats_text,
                feats_pen,
                target_list,
                padding_mask,
            ) = self.forward_frontend(
                source,
                source_text=source_text[0],
                target_list=target_list,
                padding_mask=padding_mask,
            )
        return self.forward_transformer(
            feats,
            feats_text,
            feats_pen,
            target_list=target_list,
            padding_mask=padding_mask,
            mask=mask,
            features_only=features_only,
            output_layer=output_layer,
        )

    def extract_features(
        self,
        source: torch.Tensor,
        source_text: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
        ret_conv: bool = False,
        output_layer: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        res = self.forward(
            source,
            source_text=source_text,
            padding_mask=padding_mask,
            mask=mask,
            features_only=True,
            output_layer=output_layer,
        )
        feature = res["features"] if ret_conv else res["x"]
        return feature, res["padding_mask"]

    def get_logits(self, net_output, is_masked=True):
        if is_masked:
            logits_list = net_output["logit_m_list"]
        else:
            logits_list = net_output["logit_u_list"]
        logits_list = [x.float() for x in logits_list if x is not None]
        return logits_list

    def get_targets(self, net_output, is_masked=True):
        logits_list = self.get_logits(net_output, is_masked)
        targets_list = [x.new_zeros(x.size(0), dtype=torch.long) for x in logits_list]
        return targets_list

    def get_extra_losses(self, net_output):
        extra_losses = []
        names = []

        if "features_pen" in net_output:
            extra_losses.append(net_output["features_pen"])
            names.append("features_pen")

        return extra_losses, names

    def remove_pretraining_modules(self):
        self.target_glu = None
        self.final_proj = None
        self.text_modality_bias = None
        self.text_pos = None
        self.embed_tokens = None
        self.embed_positions = None
        self.layernorm_embedding = None
