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
    make_conv_pos,
)

from fairseq.models.wavlm.wavlm import (
    ConvFeatureExtractionModel,
    TransformerEncoder,
    WavLMConfig,    
)

from fairseq.modules import GradMultiply, LayerNorm
from fairseq.modules import PositionalEmbedding
from fairseq.tasks.sthubert_pretraining2 import (
    StHubertPretrainingConfig2,
    StHubertPretrainingTask2,
)

logger = logging.getLogger(__name__)


@dataclass
class StHubertConfig2(FairseqDataclass):
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
    # relative position embedding , it is transformer of wavlm , it has relative positon embedding
    #                               it is also used at ILS-SSL model.        
    relative_position_embedding: bool = field(
        default=True,
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
    
    # embedding mixing for speech  part
    mix_with_unit: bool = field(
        default=True,
        metadata={"help": "mix with the unit embeddings"},
    )
    use_pred_unit: bool = field(
        default=False,
        metadata={"help": "use the embeddings of predicted units"},
    )
    l2_embedding: bool = field(
        default=False,
        metadata={"help": "compute l2 loss between unit embedding and unit hidden state"},
    )

    # text  part
    #max_source_positions: int = field(
    #    default=512,
    #    metadata={"help": "Maximum input length supported by the transformer encoder"},
    #)
    #text_dropout: float = field(
    #    default=0.1,
    #    metadata={"help": "dropout probability for the TextModel embedding feature"},
    #)
    text_embed_dim: int = field(
        default=768, metadata={"help": "text net part embedding dimension"}
    )
    ### CTC loss
    add_text_ctc: bool = field(
        default=True,
        metadata={"help": "add_text_ctc head"},
    )
    text_ctc_conv_kernel: int = field(
        default=2,
        metadata={"help": "text_ctc_conv kernel size"},
    )
    ## mask text
    mask_u2t: bool = field(
        default=True,
        metadata={"help": "mask the unit input in unit-to-text task"},
    )   
    ## mask lm loss
    compute_mum: bool = field(
        default=False,
        metadata={"help": "compute MLM loss in unit-to-text task"},
    )   
    ### shared transformer config
    ### it is common network, 
    shared_transformer: WavLMConfig=WavLMConfig()  
    add_unit_encoder:  bool = field(
        default=True,
        metadata={"help": "add shared transfromer network"},
    )

@register_model("sthubert2", dataclass=StHubertConfig2)
class StHubertModel2(BaseFairseqModel):
    def __init__(
        self,
        cfg: StHubertConfig2,
        task_cfg: StHubertPretrainingConfig2,
        dictionaries: List[Dictionary],
    ) -> None:
        super().__init__()
        logger.info(f"StHubertModel Config: {cfg}")
        self.cfg = cfg
        self.mask_u2t = cfg.mask_u2t # bool, for text
        self.compute_mum = cfg.compute_mum # bool for text
        self.add_text_ctc  = cfg.add_text_ctc  # bool for text
        self.mix_with_unit = cfg.mix_with_unit # bool for speech
        self.use_pred_unit = cfg.use_pred_unit # bool for speech
        self.l2_embedding = cfg.l2_embedding # bool for speech    
        self.add_unit_encoder = cfg.add_unit_encoder # bool for 

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
        ###  it is also speech part embedding layer,
        self.embed_tokens = torch.nn.Embedding(
            len(dictionaries[1]), cfg.text_embed_dim, padding_idx=self.padding_idx
        )
        #self.embed_positions = PositionalEmbedding(
        #    cfg.max_source_positions,
        #    cfg.text_embed_dim,
        #    padding_idx=self.padding_idx,
        #    learned=True,
        #)
        #self.layernorm_embedding = torch.nn.LayerNorm(cfg.text_embed_dim)

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
        dim = cfg.encoder_embed_dim if self.post_extract_proj else self.embed

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
        
        self.text_ctc_conv_kernel = cfg.text_ctc_conv_kernel

        if not task_cfg.fine_tuning:
            # pretrain case
            self.logit_temp = cfg.logit_temp
            self.skip_masked = cfg.skip_masked
            self.skip_nomask = cfg.skip_nomask

        self.final_dim = cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )

        # modality-specific positional encodings
        ## add releated postion encoding layer for text and speech
        #self.speech_pos = make_conv_pos(
        #    cfg.encoder_embed_dim, cfg.conv_pos, cfg.conv_pos_groups
        #)
        #self.text_pos = make_conv_pos(
        #    cfg.text_embed_dim, cfg.conv_pos, cfg.conv_pos_groups
        #)  ## (TODO) add text embedding layer and its dimension

        self.encoder = TransformerEncoder(cfg)

        ### shared network,
        self.shared_encoder=TransformerEncoder(cfg.shared_transformer)
        

        ### add text branch ctc head    
        ctc_head_output_dim = len(dictionaries[1])
        self.unit_encoder_ctc_head = CTClayer(cfg,ctc_head_output_dim)
        self.layer_norm = LayerNorm(self.embed)
        if not task_cfg.fine_tuning:
            self.target_glu = None
            if cfg.target_glu:
                self.target_glu = nn.Sequential(
                    nn.Linear(self.final_dim, self.final_dim * 2), nn.GLU()
                )

            self.final_proj_list = nn.ModuleList([
            nn.Linear(cfg.encoder_embed_dim, self.final_dim) for _ in dictionaries
        ])
        # modules below are not needed during fine-tuning
        self.num_classes = [len(d) for d in dictionaries]
        self.label_embs_list = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(n, self.final_dim)) for n in self.num_classes
        ])
        for i in range(len(self.num_classes)):
            nn.init.uniform_(self.label_embs_list[i])


    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""

        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: StHubertConfig2, task: StHubertPretrainingTask2):
        """Build a new model instance."""
        pathss = "/workspace2/maduo/dataset/format/librispeech"
        import os

        logger.info(f"dictionary: {task.dictionaries[1].indices.items()}")
        # with open(os.path.join(pathss, "hubert_iter1_kmdict_model.txt"), "w") as f:
        #    for sym, num in task.dictionaries[0].indices.items():
        #        f.write(f"{sym} {num}\n")
        # with open(os.path.join(pathss, "hubert_iter1_textphndict_model.txt"), "w") as f:
        #    for sym, num in task.dictionaries[1].indices.items():
        #        f.write(f"{sym} {num}\n")
        logger.info(f"dictionary bos index: {task.dictionaries[1].bos_index}")
        logger.info(f"dictionary pad index: {task.dictionaries[1].pad_index}")
        logger.info(f"dictionary eos index: {task.dictionaries[1].eos_index}")
        logger.info(f"dictionary unk index: {task.dictionaries[1].unk_index}")
        model = StHubertModel2(cfg, task.cfg, task.dictionaries)
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


    def forward_features(self, source: torch.Tensor) -> torch.Tensor:
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)
        return features

    def forward(
        self,
        source: Optional[torch.Tensor] = None,
        source_text: Optional[torch.Tensor] = None,
        source_text_lengths: Optional[torch.Tensor] = None,
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """output layer is 1-based"""
        if source_text is None:  ## 1. finetune case,
                                 ## 2.and compute loss  for only speech branch in pretraing
            #logger.info(f"now, i am here!.it is running forward_speech")
            result_speech = self.forward_speech(
                source,
                target_list=target_list,
                padding_mask=padding_mask, # only  for speech
            )
            result = {"result_speech": result_speech}
            return result
        else:
            assert source is not None and source_text is not None
            #logger.info(f"now, i am here!.it are running forward_speech and forward_text")
            result_speech = self.forward_speech(
                source,
                target_list=target_list,
                padding_mask=padding_mask,
            )  
            result_text = self.forward_text(
                source_text=source_text[0],
                source_text_lengths=source_text_lengths,
                mask=self.mask_u2t,
            )
            result={"result_speech": result_speech,"result_text": result_text}
            return result
    def forward_text(
        self, source_text, source_text_lengths: Optional[torch.Tensor] = None,  mask=True,
    ):
        """
        Args:
            source_text (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            source_text_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
        """
        #logger.info(f"in forward_text, source_text:  {source_text}")
         
        padding_mask = source_text == self.padding_idx
        token_embedding = self.embed_tokens(source_text)
        x = embed = token_embedding
        #if self.embed_positions is not None:
        #    x = embed + self.embed_positions(source_text)
        #if self.layernorm_embedding is not None:
        #    x = self.layernorm_embedding(x)
        #x = F.dropout(x, self.cfg.text_dropout)
        if mask:
            logger.info(f"it is use mask_u2t=True  for text")
            x, mask_indices = self.apply_mask(x, padding_mask, [source_text])
        
        out,_ = self.shared_encoder(x, padding_mask) ## BXTXC
        #logger.info(f"padding_mask: {padding_mask}, padding_mask shape: {padding_mask.shape}")
        result={}   
        result["shared_encoder_out_text"] =  out 
        if self.compute_mum:
            logger.info(f"it is use comput_mum=True  for text")
            code_logit_m_list, code_logit_u_list = self.compute_hubert_logits_simple(
                out, 
                source_text, 
                self.final_proj_list[-1], 
                self.label_embs_list[-1],
                padding_mask,
                mask_indices,
            )
            result["logit_m_list"] = code_logit_m_list
            result["logit_u_list"] = code_logit_u_list
        
        if self.add_text_ctc:
            logger.info(f"it is use add_text_ctc=True  for text")
            result["shared_encoder_out_ctc"] = [self.unit_encoder_ctc_head(out)]
            result["shared_encoder_padding_mask_ctc"] = [
                self.downsample_ctc_padding_mask(padding_mask)
            ]
        return result
    def downsample_ctc_padding_mask(self, padding_mask):
        """
        padding_mask: (B, T)
        """
        stride = self.text_ctc_conv_kernel // 2
        return padding_mask[:, ::stride]
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

    def forward_speech(
        self,
        source: torch.Tensor,
        #source_text: torch.Tensor,
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
    
        """output layer is 1-based"""
        # logger.info(f"source shape: {source.shape}")
        # logger.info(f"source_text shape: {source_text.shape}")
        features = self.forward_features(source)
        if target_list is not None:
            features, target_list = self.forward_targets(features, target_list)

        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        unmasked_features = features.clone()

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        # (B,T,D)
        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)
        if mask:
            x, mask_indices = self.apply_mask(features, padding_mask, target_list)
        else:
            x = features
            mask_indices = None

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool
        x, _ = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1,
        )
        
        if features_only:
            return {"x": x, "padding_mask": padding_mask, "features": features}
        
        logit_m_list, logit_u_list = self.compute_hubert_logits_simple(
            x,
            target_list[0],
            self.final_proj_list[0],
            self.label_embs_list[0],
            padding_mask,
            mask_indices,
        )     
        #logger.info(f"logit_m_list: {logit_m_list}, its length is {len(logit_m_list)}")    
        result = {
            "logit_m_list": logit_m_list,
            "logit_u_list": logit_u_list,
            "padding_mask": padding_mask,
            "features_pen": features_pen,
        }
        if self.add_unit_encoder:
            src_tokens, x_emb, l2_loss = self.swap_embedding(
                x, 
                padding_mask, 
                target_list[0],
                mask_indices=mask_indices,
                mix_with_unit=self.mix_with_unit,
                use_pred_unit=self.use_pred_unit,
                l2_embedding=self.l2_embedding,
            )
            out,_ = self.shared_encoder(x_emb, padding_mask) ## BXTXC
            result['shared_encoder_out_speech'] = out  # [(T, B, D)]
            if self.l2_embedding:
                result['embedding_l2_loss_speech'] = l2_loss
            ### mask lm for speech part again
            code_logit_m_list, code_logit_u_list = self.compute_hubert_logits_simple(
                out, 
                target_list[-1], 
                self.final_proj_list[-1], 
                self.label_embs_list[-1],
                padding_mask,
                mask_indices,
            )
            result['logit_m_list'] += code_logit_m_list
            result['logit_u_list'] += code_logit_u_list

        return result
    def swap_embedding(self,
        x,
        padding_mask,
        target=None,
        mask_indices=None,
        mix_with_unit=False,
        use_pred_unit=False,
        l2_embedding=False,
        remask=False
    ):
        """
        1. Mix with units if needed (default: True)
        2. Prepare for unit_encoder inputs
        Inputs:
            x, (B, T, D)
        Return:
            src_tokens, (B, T)
            soft_embeddings, (B, T, D)
            l2_loss, a loss
        """
        soft_embeddings = self.final_proj_list[0](x) if x.size(-1) == self.final_dim else x
        if padding_mask is None:
            padding_mask = soft_embeddings.new_zeros(soft_embeddings.size(0), soft_embeddings.size(1), dtype=torch.long)
        if use_pred_unit:
            src_tokens = self.compute_pred(self.final_proj_list[0](x), self.label_embs_list[0]).argmax(dim=-1)
            src_tokens[padding_mask] = self.padding_idx
        if target is not None:
            src_tokens = target
        else:
            src_tokens = padding_mask.long()

        if l2_embedding | mix_with_unit:
            unit_embeddings = self.embed_tokens(src_tokens)    # (B, T, D)
        
        l2_loss = 0
        if l2_embedding:
            if mask_indices is not None:
                l2_loss = (soft_embeddings - unit_embeddings)[mask_indices].float().pow(2).mean(dim=-1)
                scale = unit_embeddings[mask_indices].float().pow(2).sum(dim=-1)
            else:
                l2_loss = (soft_embeddings - unit_embeddings).float().pow(2).mean(dim=-1)
                scale = unit_embeddings.float().pow(2).sum(dim=-1)
            l2_loss = (l2_loss / scale).mean()

        if mix_with_unit:
            B, T, D = x.shape
            selected_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_prob / 2,
                self.mask_length // 2,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            selected_indices = torch.from_numpy(selected_indices).to(x.device)
            if mask_indices is not None:
                if remask:
                    remask_indices = torch.logical_and(selected_indices, mask_indices)
                    soft_embeddings[remask_indices] = self.mask_emb
                swap_indices = torch.logical_and(selected_indices, ~mask_indices)
            else:
                swap_indices = selected_indices
            soft_embeddings[swap_indices] = unit_embeddings[swap_indices]

        soft_embeddings = soft_embeddings * (1 - padding_mask.unsqueeze(-1).type_as(x))
        return src_tokens, soft_embeddings, l2_loss   
      
    def compute_pred(self, proj_x, label_embs):
        if self.target_glu:
            label_embs = self.target_glu(label_embs)
        x = F.normalize(proj_x.float(), dim=-1)                 # (S, D)
        label_embs = F.normalize(label_embs.float(), dim=-1)    # (C, D)
        logits = torch.matmul(x, label_embs.T).type_as(proj_x)  # (S, C)
        logits /= self.logit_temp
        return logits 

    def compute_hubert_logits_simple(self, x, target, final_proj, label_embs, padding_mask, mask_indices):
        if not self.skip_masked:
            masked_indices = torch.logical_and(~padding_mask, mask_indices) ## remove padding position mask
            proj_x_m = final_proj(x[masked_indices])
            logit_m_list = [self.compute_pred_offical(proj_x=proj_x_m, target=target[masked_indices], label_embs=label_embs)]
        else:
            logit_m_list = [None]

        if not self.skip_nomask:
            nomask_indices = torch.logical_and(~padding_mask, ~mask_indices)
            proj_x_u = final_proj(x[nomask_indices])
            logit_u_list = [self.compute_pred_offical(proj_x_u, target[nomask_indices], label_embs)]
        else:
            logit_u_list = [None]

        return logit_m_list, logit_u_list
    
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

    def compute_pred_offical(self,proj_x, target, label_embs):
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
     
    ## for fintune stage
    def extract_features(
        self,
        source: torch.Tensor,
        source_text: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
        ret_conv: bool = False,
        output_layer: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        features = self.forward_features(source)
        #if target_list is not None:
        #    features, target_list = self.forward_targets(features, target_list)

        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        unmasked_features = features.clone()

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        # (B,T,D)
        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)
        x = features

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool
        x, _ = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1,
        )
        out,_ = self.shared_encoder(x, padding_mask) ## BXTXC    
        
        return out, padding_mask


 
    def get_logits(self, net_output, is_masked=True):
        if is_masked:
            logits_lists=[]
            if "result_speech" in net_output:
                logits_lists.append(net_output["result_speech"]["logit_m_list"])
            if "result_text" in net_output:
                #logger.info(f'text logit_m_list: {net_output["result_text"]["logit_m_list"]}')
                logits_lists.append(net_output["result_text"]["logit_m_list"])
        else:
            logits_lists=[]
            if "result_speech" in net_output:
                logits_lists.append(net_output["result_speech"]["logit_u_list"])
            if "result_text" in net_output:
                #logger.info(f'text logit_u_list: {net_output["result_text"]["logit_u_list"]}')
                logits_lists.append(net_output["result_text"]["logit_u_list"])
        assert len(logits_lists)<=2
        logits_list=[]
        for lists in logits_lists:
            for x in lists:
                if x is not None:
                    logits_list.append(x.float())
        #logger.info(f"logits_list: {logits_list}, its length is {len(logits_list)}")             
        
        return logits_list

    def get_targets(self, net_output, is_masked=True):
        logits_list = self.get_logits(net_output, is_masked)
        targets_list = [x.new_zeros(x.size(0), dtype=torch.long) for x in logits_list]
        return targets_list

    def get_extra_losses(self, net_output):
        extra_losses = []
        names = []

        if "features_pen" in net_output["result_speech"]:
            extra_losses.append(net_output["result_speech"]["features_pen"])
            names.append("features_pen")
        if "embedding_l2_loss_speech" in net_output["result_speech"]:
            extra_losses.append(net_output["result_speech"]["embedding_l2_loss_speech"])
            names.append("embedding_l2_loss")

        return extra_losses, names

    def remove_pretraining_modules(self):
        self.target_glu = None
        #self.final_proj_list = None
        #self.text_modality_bias = None
        #self.text_pos = None
        #self.embed_tokens = None
        #self.embed_positions = None
        #self.layernorm_embedding = None

class CTClayer(nn.Module):
    def __init__(self,cfg,output_dim):
        super().__init__()
        self.cfg = cfg 
        self.output_dim = output_dim
        self.conv = nn.Conv1d(
            self.cfg.shared_transformer.encoder_embed_dim, self.cfg.shared_transformer.encoder_embed_dim,
            2,
            stride=1,
            bias=False,
            padding=1,
        )
        self.dropout=nn.Dropout(p=0.1)
        self.layernorm = LayerNorm(self.cfg.shared_transformer.encoder_embed_dim)
        self.activation = nn.GELU()
        self.output_layer = nn.Linear(self.cfg.shared_transformer.encoder_embed_dim, self.output_dim)
    def forward(self,x):
        # input x shape: BXTXC
        x = x.permute(0,2,1) # BXTXC ->BXCXT
        x = self.conv(x) # BXCXT
        x = x.permute(0,2,1) # BXCXT -> BXTXC
        x = self.dropout(x)
        x = self.layernorm(x)
        x = self.activation(x)
        x = self.output_layer(x) # BX T X output_dim
        return x

class Rotate3D(nn.Module):
    """
    (T, B, D) --> (B, D, T) --> (D, T, B) --> (T, B, D)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(1, 2, 0)
