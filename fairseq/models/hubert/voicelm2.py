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
from fairseq.models.hubert.hubert2 import HubertConfig2, HubertModel2  
from fairseq.modules import GradMultiply, LayerNorm
from fairseq.tasks.voicelm2_pretraining import (
    Voicelm2PretrainingConfig,
    Voicelm2PretrainingTask,
)


logger = logging.getLogger(__name__)

##1.auido branch: cnn feature(50Hz) or fbank(104dims, 25Hz)(todo)
##2.text branch: embedding + ffn
##3 fusion type: residual cross self-attention(support flash_attention ),
##4.audio branch: mask loss 
##5.text branch: mask loss


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
    text_mlm_loss: bool  = field(
            default=False,
            metadata = {"help": "if true, it will  comput  unpaired text branch  masked lm loss."},)

    predict_layers: str = field(default="[12]") # set [7,12], interget voicelm 
    separate_label_embeds: bool = field(default=False) # set True
    separate_layer_targets: bool = field(default=False)
    phnkm7_km12: bool = field(default=False) # set True

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
        #logger.info(f"voicelm2 Config: {cfg}")
        self.padding_idx = 1
        self.sim_type = cfg.sim_type
        self.modality_fuse = cfg.modality_fuse
        self.encoder_embed_dim = cfg.encoder_embed_dim
        feature_enc_layers = eval(cfg.conv_feature_layers)  # noqa
        self.embed = feature_enc_layers[-1][0]
        self.predict_layers = eval(cfg.predict_layers)
        self.separate_label_embeds = cfg.separate_label_embeds
        self.separate_layer_targets = cfg.separate_layer_targets
        self.phnkm7_km12 = cfg.phnkm7_km12
        self.text_mlm_loss = cfg.text_mlm_loss
        self.fuse_attention_heads = cfg.fuse_attention_heads
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
        elif self.modality_fuse == "flash_attention":
            pass
        elif self.modality_fuse == "attention_query_text":
            self.feature_fuse = nn.MultiheadAttention(
                embed_dim=cfg.encoder_embed_dim,
                num_heads=cfg.fuse_attention_heads,
                batch_first=True,
            )
        elif self.modality_fuse == "flash_attention_query_text":
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


        final_dim = cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )
        
        self.untie_final_proj = cfg.untie_final_proj
        self.layer_norm_first = cfg.layer_norm_first

        if self.layer_norm_first:
            self.post_layer_norm = torch.nn.Sequential(
                *[
                    LayerNorm(cfg.encoder_embed_dim)
                    for _ in range(len(self.predict_layers))
                ]
            )

        if self.separate_label_embeds:
            if self.separate_layer_targets or not self.untie_final_proj:
                self.final_proj = torch.nn.Sequential(
                    *[
                        nn.Linear(cfg.encoder_embed_dim, cfg.final_dim)
                        for _ in range(len(self.predict_layers))
                    ]
                )
            else:
                self.final_proj = torch.nn.Sequential(
                    *[
                        nn.Linear(
                            cfg.encoder_embed_dim, cfg.final_dim * len(dictionaries)
                        )
                        for _ in range(len(self.predict_layers))
                    ]
                )
        else:
            if self.separate_layer_targets or not self.untie_final_proj:
                self.final_proj = nn.Linear(cfg.encoder_embed_dim, cfg.final_dim)
            else:
                self.final_proj = nn.Linear(
                    cfg.encoder_embed_dim, cfg.final_dim * len(dictionaries)
                )

        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(cfg.encoder_embed_dim)

        self.target_glu = None
        if cfg.target_glu:
            self.target_glu = nn.Sequential(
                nn.Linear(final_dim, final_dim * 2), nn.GLU()
            )
        if self.text_mlm_loss:
            self.final_text_proj = nn.Linear(cfg.encoder_embed_dim, cfg.final_dim)
            self.text_num_classes = [len(d) for d in [dictionaries[-1]]] ## unpaired text dictionary
            self.text_label_embs = nn.Parameter(
                torch.FloatTensor(sum(self.text_num_classes), cfg.final_dim)
            )
            nn.init.uniform_(self.text_label_embs)
            
        else:
            self.final_text_proj = None
            self.text_label_embs = None
            self.text_num_classes = None

        """
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
        """
        # modules below are not needed during fine-tuning
        if any([d is None for d in dictionaries]):
            logger.info("cannot find dictionary. assume will be used for fine-tuning")
        else:
            self.num_classes = [len(d) for d in dictionaries[:-1]] ## remove unpaired text dictionary
            layer_dim = (
                len(self.predict_layers)
                if self.separate_layer_targets or self.separate_label_embeds
                else 1
            )
            embed_dim = (
                sum(self.num_classes)
                if not self.separate_layer_targets
                else max(self.num_classes)
            )
            self.label_embs_concat = nn.Parameter(
                torch.FloatTensor(layer_dim, embed_dim, cfg.final_dim)
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

        # feature_audio = feature_audio.transpose(1, 2) ## [B,F,T]->[B,T,F]
        features = None
        if self.modality_fuse == "attention": ## query is from audio, key/value is from text
            # logger.info(f"feature_text shape: {feature_text.shape}") # [B,S,F]
            # logger.info(f"feature_audio shape: {feature_audio.shape}") # [B,T,F]
            features, _ = self.feature_fuse(
                query=feature_audio, key=feature_text, value=feature_text
            )  # [B,T,F]
        elif self.modality_fuse == "flash_attention": # query is from audio, key/value is from text with flash_attention
            with torch.backends.cuda.sdp_kernel(
                enable_math=False
            ):  ## it default is enable_flash=True,
                ## enable_math=True, enable_mem_efficient=True
                B, T, _ = feature_audio.shape
                S = feature_text.size(1)
                feature_audio = feature_audio.view(B,self.fuse_attention_heads, T, -1)
                feature_text = feature_text.view(B,self.fuse_attention_heads, S, -1)
                features = F.scaled_dot_product_attention( # ## its inputs require: q,k,v : (B, heads,seq,head_dim)
                    query=feature_audio, key=feature_text, value=feature_text
                ) # (B, heads, T, F//heads)
                feature_audio = feature_audio.reshape(B,T,-1)
                features = features.reshape(B,T,-1)
                feature_text = feature_text.view(B,S,-1)

        elif self.modality_fuse == "attention_query_text": ## query is from text, key/value is from audio
            # logger.info(f"feature_text shape: {feature_text.shape}") # [B,S,F]
            # logger.info(f"feature_audio shape: {feature_audio.shape}") # [B,T,F]
            features, _ = self.feature_fuse(
                query=feature_text, key=feature_audio, value=feature_audio
            )  # [B,S,F]
            #B, T, D = feature_audio.shape
            #features = features.

        elif self.modality_fuse == "flash_attention_query_text": # query is from text, key/value is from audio with flash_attention
            with torch.backends.cuda.sdp_kernel(
                enable_math=False
            ):  ## it default is enable_flash=True,
                ## enable_math=True, enable_mem_efficient=True
                B, T, _ = feature_audio.shape
                S = feature_text.size(1)
                feature_audio = feature_audio.view(B,self.fuse_attention_heads, T, -1)
                feature_text = feature_text.view(B,self.fuse_attention_heads, S, -1)
                features = F.scaled_dot_product_attention( # ## its inputs require: q,k,v : (B, heads,seq,head_dim)
                    query=feature_text, key=feature_audio, value=feature_audio
                ) # (B, heads, T, F//heads)
                feature_audio = feature_audio.reshape(B,T,-1)
                features = features.reshape(B,T,-1)
                feature_text = feature_text.view(B,S,-1)

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
        x, layer_results = self.encoder(
            x, padding_mask=padding_mask, layer=self.predict_layers
        )

        result = {
            "x": x,
            "padding_mask": padding_mask,
            "features": features,
            "layer_results": layer_results, #  [[T,B,C],[T,B,C]]
        }

        ## for  speech branch  loss
        #if features_only:
        #    if self.layer_norm_first and output_layer is not None:
        #        result["x"] = self.post_layer_norm[-1](x)
        #    return result
        if features_only:
            return {"x": x, "padding_mask": padding_mask, "features": features}


        layer_results = [
            layer_x.transpose(0, 1) for i, (layer_x, _) in enumerate(layer_results)
        ] #  [[T,B,C],[T,B,C]] -> [[B,T,C],[B,T,C]]
        if not (x == layer_results[-1]).all():
            print(
                "{} {} {} {}".format(
                    (x == layer_results[-1]).shape,
                    (x == layer_results[-1]).float().sum(),
                    (x - layer_results[-1]).float().sum(),
                    (x - layer_results[-1]).float().abs().max(),
                )
            )

        if self.layer_norm_first:
            layer_results = [
                layernorm(x)
                for x, layernorm in zip(layer_results, self.post_layer_norm)
            ]
        logit_m_list = []
        logit_u_list = []
        target_m_list = []
        target_u_list = []
        if self.separate_layer_targets:
            assert len(layer_results) == len(self.final_proj)
            assert len(layer_results) == len(self.label_embs_concat)

        for i, layer_x in enumerate(
            layer_results
        ):  # , final_proj, label_embs in zip(layer_results, self.final_proj, label_embs_concat):
            if self.separate_label_embeds:
                final_proj = self.final_proj[i]
            else:
                final_proj = self.final_proj

            if self.separate_label_embeds or self.separate_layer_targets:
                label_embs = self.label_embs_concat[i]
            else:
                label_embs = self.label_embs_concat[0]

            if not self.separate_layer_targets:
                label_embs_list = label_embs.split(self.num_classes, 0)
            else:
                label_embs_list = [label_embs[: self.num_classes[i]]]

            proj_x = final_proj(layer_x) 
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
                logit_m_list += [logit[mask] for logit in logit_list]
                target_m_list += [target.view(-1)[mask].long() for target in target_list]
            else:
                logit_m_list += [None for _ in target_list]

            if not self.skip_nomask:
                unmask = torch.logical_and(~mask_indices, ~padding_mask).view(-1)  # [B*T]
                logit_u_list += [logit[unmask] for logit in logit_list]
                target_u_list += [target.view(-1)[unmask].long() for target in target_list]
            else:
                logit_u_list += [None for _ in target_list]

        ## Assume two style target label: ["bpekm","km"] ,two specify layer: [7,12]
        ## so logit_m_list has four elements: they are [bpekm_7,km_7,bpekm_12, km_12]
        ## Assume two style target label: ["bpekm","km"] ,two specify layer: [4, 7,12]
        ## so logit_m_list has six elements: they are [bpekm_4, km_4, bpekm_7, km_7, bpekm_12, km_12]
        ## if we only  want to logit_m_list has three elements:  they are [km_4,bpekm_7, km_12], how to do it?
        logit_m_list_1 = []
        logit_u_list_1 = []
        target_m_list_1 = []
        target_u_list_1 = []
        if self.phnkm7_km12:
            logit_m_list_1.append(logit_m_list[0])
            logit_m_list_1.append(logit_m_list[3])
            logit_m_list = logit_m_list_1
            logit_u_list_1.append(logit_u_list[0])
            logit_u_list_1.append(logit_u_list[3])
            logit_u_list = logit_u_list_1
            
            target_m_list_1.append(target_m_list[0])
            target_m_list_1.append(target_m_list[3])
            target_m_list = target_m_list_1
            target_u_list_1.append(target_u_list[0])
            target_u_list_1.append(target_u_list[3])
            target_u_list = target_u_list_1

        
        if self.text_mlm_loss:
            ##(TODO)maybe add espent style mask_uniform
            text_padding_mask = torch.BoolTensor(src_text.shape).fill_(False).to(src_text.device) ## ignore padding effect. 
            #logger.info(f"text_padding_mask device: {text_padding_mask.device}")
            #logger.info(f"feature_text device: {feature_text.device}")
            feature_text_mask, text_mask_indices = self.apply_feature_text_mask(feature_text, text_padding_mask)
            #logger.info(f"feature_text_mask device: {feature_text_mask.device}")
            #logger.info(f"text_padding_mask device: {text_padding_mask.device}")
            text_x, _ = self.encoder(feature_text_mask , padding_mask=text_padding_mask, layer=None)#(B,S,F)
            proj_x = self.final_text_proj(text_x) #(B,S,F_)
            text_proj_x_list = [proj_x for _ in self.text_num_classes]#[[B,S,F_]]
            text_label_embs_list = self.text_label_embs.split(self.text_num_classes, 0)# [B,V,F_] ## (todo check)
            #logger.info(f"text_label_embs_list,its frist element shape: {text_label_embs_list[0].shape}")#(V,F_)
            text_logit_list = [self.compute_logits(proj, emb).view(-1, num_class) for proj, emb, num_class in zip(
                    text_proj_x_list, text_label_embs_list, self.text_num_classes
                )]
            text_mask = torch.logical_and(text_mask_indices, ~text_padding_mask).view(-1)  # [B*S]

            logit_m_list += [logit[text_mask] for logit in text_logit_list]
            target_m_list += [src_text.view(-1)[text_mask].long()]
        result = {
            "logit_m_list": logit_m_list,
            "logit_u_list": logit_u_list,
            "target_m_list": target_m_list,
            "target_u_list": target_u_list,
            "padding_mask": padding_mask,
            "features_pen": features_pen,
        }
        #logger.info(f"in the forward of model: ouput result: {result}")
        return result



    def extract_features( ## it is only used to get specify layer represent for clustering  for next iter  pretrain
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False, ## it is setting to false
        ret_conv: bool = False,
        output_layer: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        res = self.forward(
            source,
            padding_mask=padding_mask,
            mask=mask,
            features_only=True,
            output_layer=output_layer,
        )
        feature = res["features"] if ret_conv else res["x"]
        return feature, res["padding_mask"]

    ## here it don't need to target list, because target list will ocurr in criterions/ctc.py
    def extract_finetune(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True, ## it is setted by apply_mask config of hubert_asr.py, 
                           ## offical default is true in funetune  mode
        features_only: bool = False,
        output_layer: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """output layer is 1-based"""
        src_audio, src_text = source["audio"], source["text"]
        src_audio = src_audio.to(torch.float16)
        if  src_text is None:
            src_text = src_text

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
            feature_text = feature_audio.new_zeros(
                feature_audio.size(0), feature_audio.size(1), feature_audio.size(2)
            )


        if self.modality_fuse == "attention": ## query is from audio, key/value is from text
            # logger.info(f"feature_text shape: {feature_text.shape}") # [B,S,F]
            # logger.info(f"feature_audio shape: {feature_audio.shape}") # [B,T,F]
            features, _ = self.feature_fuse(
                query=feature_audio, key=feature_text, value=feature_text
            )  # [B,T,F]
        elif self.modality_fuse == "flash_attention": # query is from audio, key/value is from text with flash_attention
            with torch.backends.cuda.sdp_kernel(
                enable_math=False
            ):  ## it default is enable_flash=True,
                ## enable_math=True, enable_mem_efficient=True
                B, T, _ = feature_audio.shape
                S = feature_text.size(1)
                feature_audio = feature_audio.view(B,self.fuse_attention_heads, T, -1)
                feature_text = feature_text.view(B,self.fuse_attention_heads, S, -1)
                features = F.scaled_dot_product_attention( # ## its inputs require: q,k,v : (B, heads,seq,head_dim)
                    query=feature_audio, key=feature_text, value=feature_text
                ) # (B, heads, T, F//heads)
                feature_audio = feature_audio.reshape(B,T,-1)
                features = features.reshape(B,T,-1)
                feature_text = feature_text.view(B,S,-1)

        elif self.modality_fuse == "attention_query_text": ## query is from text, key/value is from audio
            # logger.info(f"feature_text shape: {feature_text.shape}") # [B,S,F]
            # logger.info(f"feature_audio shape: {feature_audio.shape}") # [B,T,F]
            features, _ = self.feature_fuse(
                query=feature_text, key=feature_audio, value=feature_audio
            )  # [B,T,F]

        elif self.modality_fuse == "flash_attention_query_text": # query is from text, key/value is from audio with flash_attention
            with torch.backends.cuda.sdp_kernel(
                enable_math=False
            ):  ## it default is enable_flash=True,
                ## enable_math=True, enable_mem_efficient=True
                B, T, _ = feature_audio.shape
                S = feature_text.size(1)
                feature_audio = feature_audio.view(B,self.fuse_attention_heads, T, -1)
                feature_text = feature_text.view(B,self.fuse_attention_heads, S, -1)
                features = F.scaled_dot_product_attention( # ## its inputs require: q,k,v : (B, heads,seq,head_dim)
                    query=feature_text, key=feature_audio, value=feature_audio
                ) # (B, heads, T, F//heads)
                feature_audio = feature_audio.reshape(B,T,-1)
                features = features.reshape(B,T,-1)
                feature_text = feature_text.view(B,S,-1)

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
        return x, padding_mask


    def apply_feature_text_mask(self,x,padding_mask):
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
            x = x.clone()
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None
        return x, mask_indices

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
        self.label_embs_concat = None
