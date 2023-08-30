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
from fairseq.models.wavlm.wavlm import (
    ConvFeatureExtractionModel,
    TransformerEncoder,
)

from fairseq.modules import GradMultiply, LayerNorm
from fairseq.modules import PositionalEmbedding
from fairseq.models.hubert.hubert2 import HubertConfig2, HubertModel2  ##(TODO) check
from fairseq.modules import GradMultiply, LayerNorm
from fairseq.tasks.voicelm2_pretraining import (
    Voicelm2PretrainingConfig,
    Voicelm2PretrainingTask,
)  ##(TODO) check


logger = logging.getLogger(__name__)

##1.auido branch: cnn feature(50Hz) or fbank(104dims, 25Hz)
##2.text branch: embedding + ffn
##3 fusion type: residual cross self-attention,
##4.audio branch: mask loss and unmask loss
##5.text branch: mask loss(todo)


@dataclass
class Voicelm2Config(HubertConfig2):
    # relative position embedding
    relative_position_embedding: bool = field(
        default=False,
        metadata={
            "help": "whether to use the relative position embedding, (bucket relpos embedding by default)"
        },
    )
    num_buckets: int = field(
        default=320,
        metadata={"help": "the number of buckets for relative position embedding"},
    )
    max_distance: int = field(
        default=800,
        metadata={
            "help": "the maximum distance for computing relative bias, beyond which will assign the same embedding"
        },
    )

    ## fusion related part
    audio_feature_type: str = field(
        default="cnn",
        metadata={"help": "audio feature extractor: cnn layes or fbank layer:"},
    )
    modality_fuse: str = field(
        default="attention",
        metadata={"help": "fusing two modalities: cross self-attention"},
    )
    fuse_attention_heads: int = field(
        default=12, metadata={"help": "num fuse attention heads"}
    )

    ## for audio branch loss
    sim_type: str = field(default="cosine", metadata={"help": "similarity type"})
    # mask_type: str = field(default='input', metadata={'help': 'input or feature masking for text branch'})
    ## text
    text_embed_dim: int = field(
        default=768, metadata={"help": "text net part embedding dimension"}
    )
    max_source_positions: int = field(
        default=512,
        metadata={"help": "Maximum input length supported by the transformer encoder"},
    )

    attention_type: str = field(
        default="rel_attention",
        metadata={
            "help": """now it contains two options: rel_attention, flash_attention,
                                                                   rel_attention is has relative attention baise using bucket,'
                                                                  'it is MultiheadAttention2, flash_attetion is Fast 
                                                                   and Memory-Efficient multi head attention, but require cuda>=11.4, pytorch>=1.12"""
        },
    )


class TextModel(nn.Module):
    def __init__(self, input_dim=None, cfg=None, padding_idx=None):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(
            input_dim, cfg.text_embed_dim, padding_idx=padding_idx
        )
        # self.embed_positions = PositionalEmbedding(
        #    cfg.max_source_positions,
        #    cfg.text_embed_dim,
        #    padding_idx=padding_idx,
        #    learned=True,
        # )
        self.linear = nn.Linear(cfg.text_embed_dim, cfg.encoder_embed_dim)

    def forward(self, x):
        x = self.embed_tokens(x)
        # x = self.embed_positions(x)
        x = self.linear(x)
        return x


@register_model("voicelm2", dataclass=Voicelm2Config)
class Voicelm2Model(BaseFairseqModel):
    def __init__(
        self,
        cfg: Voicelm2Config,
        task_cfg: Voicelm2PretrainingConfig,  ##(TODO) check, will add Voicelm2PretrainingConfig in voicelm2_pretraining.py
        dictionaries: List[Dictionary],
    ) -> None:
        super().__init__()
        logger.info(f"voicelm2 Config: {cfg}")
        self.padding_idx = 1
        self.sim_type = cfg.sim_type
        self.modality_fuse = cfg.modality_fuse
        self.encoder_embed_dim = cfg.encoder_embed_dim
        feature_enc_layers = eval(cfg.conv_feature_layers)  # noqa
        self.embed = feature_enc_layers[-1][0]
        if cfg.audio_feature_type == "cnn":
            self.feature_extractor_audio = ConvFeatureExtractionModel(
                conv_layers=feature_enc_layers,
                dropout=0.0,
                mode=cfg.extractor_mode,
                conv_bias=cfg.conv_bias,
            )
        elif (
            cfg.audio_feature_type == "fbank"
        ):  ### torchaudio fbank not support batch process
            pass

        self.feature_extractor_text = TextModel(
            input_dim=len(dictionaries[1]), cfg=cfg, padding_idx=self.padding_idx
        )

        if self.modality_fuse == "attention":
            self.feature_fuse = nn.MultiheadAttention(
                embed_dim=cfg.encoder_embed_dim,
                num_heads=cfg.fuse_attention_heads,
                batch_first=True,
            )
        else:
            pass

        feature_ds_rate = np.prod([s for _, _, s in feature_enc_layers])
        self.feat2tar_ratio = cfg.label_rate * feature_ds_rate / task_cfg.sample_rate

        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != self.encoder_embed_dim
            else None
        )

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
        self.logit_temp = cfg.logit_temp
        self.skip_masked = cfg.skip_masked
        self.skip_nomask = cfg.skip_nomask

        ## text
        # `self.text_mask_type = cfg.text_mask_type

        final_dim = cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )

        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(cfg.encoder_embed_dim)

        self.target_glu = None
        if cfg.target_glu:
            self.target_glu = nn.Sequential(
                nn.Linear(final_dim, final_dim * 2), nn.GLU()
            )

        self.untie_final_proj = cfg.untie_final_proj
        if self.untie_final_proj:
            self.final_proj = nn.Linear(
                cfg.encoder_embed_dim, final_dim * len(dictionaries)
            )
        else:
            self.final_proj = nn.Linear(cfg.encoder_embed_dim, final_dim)

        # modules below are not needed during fine-tuning
        if any([d is None for d in dictionaries]):
            logger.info("cannot find dictionary. assume will be used for fine-tuning")
        else:
            self.num_classes = [
                len(d) for d in dictionaries
            ]  ##  because  audio and text are  used same  dictionary,
            ##  so self.num_classes  should  be  one  element  of  list. ,this element should be 45
            self.num_classes = [self.num_classes[0]]
            # logger.info(f"self.num_classes: {self.num_classes},  its  len: {len(self.num_classes)}")
            self.label_embs_concat = nn.Parameter(
                torch.FloatTensor(sum(self.num_classes), final_dim)
            )
            nn.init.uniform_(self.label_embs_concat)

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""

        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: Voicelm2Config, task: Voicelm2PretrainingTask):
        """Build a new model instance."""
        logger.info(
            f"dictionary: {task.dictionaries[1].indices.items()}"
        )  # dictionary: dict_items([('<s>', 0),
        # ('<pad>', 1), ('</s>', 2), ('<unk>', 3), ('0', 4), ('1', 5), ('2', 6), ('3', 7), ('4', 8), ('5', 9),
        # ('6', 10), ('7', 11), ('8', 12), ('9', 13), ('10', 14), ('11', 15), ('12', 16), ('13', 17), ('14', 18),
        # ('15', 19), ('16', 20), ('17', 21), ('18', 22), ('19', 23), ('20', 24), ('21', 25), ('22', 26),
        # ('23', 27), ('24', 28), ('25', 29), ('26', 30), ('27', 31), ('28', 32), ('29', 33), ('30', 34),
        # ('31', 35), ('32', 36), ('33', 37), ('34', 38), ('35', 39), ('36', 40), ('37', 41), ('38', 42),
        # ('39', 43), ('40', 44)])
        logger.info(f"dictionary bos index: {task.dictionaries[1].bos_index}")  ## 0
        logger.info(f"dictionary pad index: {task.dictionaries[1].pad_index}")  ## 1
        logger.info(f"dictionary eos index: {task.dictionaries[1].eos_index}")  ## 2
        logger.info(f"dictionary unk index: {task.dictionaries[1].unk_index}")  ## 3
        model = Voicelm2Model(cfg, task.cfg, task.dictionaries)
        return model

    def forward(
        self,
        source: torch.Tensor,
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """output layer is 1-based"""
        src_audio, src_text = source["audio"], source["text"]
        feature_audio = self.forward_features(
            src_audio, modality="audio"
        )  # features: [B, F, T]
        if target_list is not None:
            feature_audio, target_list = self.forward_audio_targets(
                feature_audio, target_list
            )

        feature_audio = feature_audio.transpose(1, 2)  #  [B, F, T]  -> [B, T, F]
        # if self.embed != self.encoder_embed_dim:
        if self.post_extract_proj is not None:
            feature_audio = self.post_extract_proj(feature_audio)  # [B,T,F]
        feature_text = self.forward_features(
            src_text, modality="text"
        )  # features: [B,S,F],S is text seq length.

        ## feature fuse via cross attention qurey is from audio branch, key and value is from text branch
        # feature_audio = feature_audio.transpose(1, 2) ## [B,F,T]->[B,T,F]
        features = None
        if self.modality_fuse == "attention":
            # logger.info(f"feature_text shape: {feature_text.shape}") # [B,S,F]
            # logger.info(f"feature_audio shape: {feature_audio.shape}") # [B,T,F]
            features, _ = self.feature_fuse(
                query=feature_audio, key=feature_text, value=feature_text
            )  # [B,T,F]
        elif self.modality_fuse == "flash_attention":
            with torch.backends.cuda.sdp_kernel(
                enable_math=False
            ):  ## it default is enable_flash=True,
                ## enable_math=True, enable_mem_efficient=True
                features = F.scaled_dot_product_attention(
                    query=feature_audio, key=feature_text, value=feature_text
                )
        features = feature_audio + features  ## residual add

        # logger.info(f"last  features shape: {features.shape}") # [B,T,F]
        features_pen = features.float().pow(2).mean()
        features = self.layer_norm(features)  #  [B,T,F]
        if padding_mask is not None:
            # logger.info(f"in first padding_mask shape : {padding_mask.shape}") #(B, T'),T' is sample point of audio
            padding_mask = self.forward_padding_mask(features, padding_mask)  # (B, T)

        features = self.dropout_input(features)
        if mask:
            # logger.info(f"features shape: {features.shape}")
            # logger.info(f"padding_mask shape : {padding_mask.shape}")
            x, mask_indices = self.apply_feature_mask(features, padding_mask)
        else:
            x = features
            mask_indices = None

        # target: (B, T), long
        # x: (B, T, F), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool
        x, _ = self.encoder(
            x,  # (B,T,F)
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1,
        )
        # x:(B,T,F)
        if features_only:
            return {"x": x, "padding_mask": padding_mask, "features": features}

        label_embs_list = self.label_embs_concat.split(self.num_classes, 0)
        proj_x = self.final_proj(x)  # (B,T,F)
        if self.untie_final_proj:
            proj_x_list = proj_x.chunk(
                len(self.num_classes), dim=-1
            )  # len(proj_x_list) = len(self.num_classes)
        else:
            proj_x_list = [proj_x for _ in self.num_classes]
        logit_list = [
            self.compute_logits(proj, emb).view(-1, num_class)
            for proj, emb, num_class in zip(
                proj_x_list, label_embs_list, self.num_classes
            )
        ]  # [[B*T, V]]
        if not self.skip_masked:
            mask = torch.logical_and(mask_indices, ~padding_mask).view(-1)  # [B*T]
            logit_m_list = [logit[mask] for logit in logit_list]
            target_m_list = [target.view(-1)[mask].long() for target in target_list]
        else:
            logit_m_list = [None for _ in target_list]

        if not self.skip_nomask:
            unmask = torch.logical_and(~mask_indices, ~padding_mask).view(-1)  # [B*T]
            logit_u_list = [logit[unmask] for logit in logit_list]
            target_u_list = [target.view(-1)[unmask].long() for target in target_list]
        else:
            logit_u_list = [None for _ in target_list]
        # logger.info(f"logit_m_list  len: {len(logit_m_list)}") # its length is same as len(self.num_classes),
        # it should be is 1
        # logger.info(f"logit_u_list  len: {len(logit_u_list)}") # its length is same as len(self.num_classes),
        # it should be is 1
        # logger.info(f"target_m_list  len: {len(target_m_list)}") # its length is same as len(self.num_classes),
        # it should be is 1
        # logger.info(f"target_u_list  len: {len(target_u_list)}") # its length is same as len(self.num_classes),
        # it should be is 1
        result = {
            "logit_m_list": logit_m_list,
            "logit_u_list": logit_u_list,
            "target_m_list": target_m_list,
            "target_u_list": target_u_list,
            "padding_mask": padding_mask,
            "features_pen": features_pen,
        }

        return result

    ## here it don't need to target list, because target list will ocurr in criterions/ctc.py
    def extract_finetune(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """output layer is 1-based"""
        src_audio, src_text = source["audio"], source["text"]

        feature_audio = self.forward_features(
            src_audio, modality="audio"
        )  # features: [B, F, T]
        feature_audio = feature_audio.transpose(1, 2)  #  [B, F, T]  -> [B, T, F]
        # if self.embed != self.encoder_embed_dim:
        if self.post_extract_proj is not None:
            feature_audio = self.post_extract_proj(feature_audio)  # [B,T,F]

        if src_text is not None:
            feature_text = self.forward_features(
                src_text, modality="text"
            )  # features: [B,S,F],S is text seq length.
        else:
            feature_text = feature_audio.new_zero(
                feature_audio.size(0), feature_audio.size(1), feature_audio.size(2)
            )

        ## feature fuse via cross attention qurey is from audio branch, key and value is from text branch
        # feature_audio = feature_audio.transpose(1, 2) ## [B,F,T]->[B,T,F]
        features = None
        if self.modality_fuse == "attention":
            # logger.info(f"feature_text shape: {feature_text.shape}") # [B,S,F]
            # logger.info(f"feature_audio shape: {feature_audio.shape}") # [B,T,F]
            features, _ = self.feature_fuse(
                query=feature_audio, key=feature_text, value=feature_text
            )  # [B,T,F]
        elif self.modality_fuse == "flash_attention":
            with torch.backends.cuda.sdp_kernel(
                enable_math=False
            ):  ## it default is enable_flash=True,
                features = F.scaled_dot_product_attention(
                    query=feature_audio, key=feature_text, value=feature_text
                )
        features = feature_audio + features  ## residual add

        # logger.info(f"last  features shape: {features.shape}") # [B,T,F]
        features_pen = features.float().pow(2).mean()
        features = self.layer_norm(features)  #  [B,T,F]
        if padding_mask is not None:
            # logger.info(f"in first padding_mask shape : {padding_mask.shape}") #(B, T'),T' is sample point of audio
            padding_mask = self.forward_padding_mask(features, padding_mask)  # (B, T)

        features = self.dropout_input(features)
        if mask:
            # logger.info(f"features shape: {features.shape}")
            # logger.info(f"padding_mask shape : {padding_mask.shape}")
            x, mask_indices = self.apply_feature_mask(features, padding_mask)
        else:
            x = features
            mask_indices = None

        # target: (B, T), long
        # x: (B, T, F), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool
        x, _ = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1,
        )

    def apply_feature_mask(self, x, padding_mask):
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

    def forward_features(self, source: torch.Tensor, modality: str) -> torch.Tensor:
        extractor = eval(f"self.feature_extractor_{modality}")
        if self.feature_grad_mult > 0:
            features = extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = extractor(source)
        return features

    def forward_audio_targets(
        self,
        features: torch.Tensor,  # mask_indices: torch.Tensor,
        target_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Trim features to ensure labels exist and then get aligned labels
        feat_tsz = features.size(2)
        targ_tsz = min([t.size(1) for t in target_list])
        if self.feat2tar_ratio * feat_tsz > targ_tsz:
            feat_tsz = int(targ_tsz / self.feat2tar_ratio)
            features = features[..., :feat_tsz]
            # if mask_indices is not None:
            #    mask_indices = mask_indices[..., :feat_tsz]
        target_inds = torch.arange(feat_tsz).float() * self.feat2tar_ratio
        target_list = [t[:, target_inds.long()] for t in target_list]
        # return features, mask_indices, target_list
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

    def compute_logits(self, feats, emb_mat):
        # feats: [B, T, F], emb_mat: [V, F]
        if self.sim_type == "dot":
            logits = torch.matmul(feats, emb_mat.transpose(0, 1))
        elif self.sim_type == "cosine":
            batch_size, timesteps, emb_dim = feats.size()
            feats_ = feats.view(-1, emb_dim)
            nom = (feats_.unsqueeze(dim=1) * emb_mat.unsqueeze(dim=0)).sum(
                dim=-1
            )  # [B*T, V]
            denom = (feats_**2).sum(dim=-1).sqrt().unsqueeze(dim=1) * (
                emb_mat**2
            ).sum(dim=-1).sqrt().unsqueeze(
                dim=0
            )  # [B*T, V]
            logits = (nom / denom.clamp(min=1e-6)).view(batch_size, timesteps, -1)
        else:
            raise NotImplementedError
        logits = logits / self.logit_temp
        return logits

    """
    def compute_nce(self, x, pos, negs):
        neg_is_pos = (pos == negs).all(-1)
        pos = pos.unsqueeze(0)
        targets = torch.cat([pos, negs], dim=0)

        logits = torch.cosine_similarity(
            x.float(), targets.float(), dim=-1
        ).type_as(x)
        logits /= self.logit_temp
        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")
        logits = logits.transpose(0, 1)  # (num_x, num_cls+1)
        return logits
    """

    def get_extra_losses(self, net_output):
        extra_losses = []
        names = []
        if "features_pen" in net_output:
            extra_losses.append(net_output["features_pen"])
            names.append("features_pen")

        return extra_losses, names

    def get_logits(self, net_output, is_masked=True):
        raise NotImplementedError

    def get_targets(self, net_output, is_masked=True):
        raise NotImplementedError

    def remove_pretraining_modules(self):
        self.target_glu = None
        self.final_proj = None
