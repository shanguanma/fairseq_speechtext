#!/usr/bin/env python3

import torch
import torch.nn as nn
from eend.eend.pytorch_backend.eend_m2f.mask2former_matcher import HungarianMatcher as mask2formerHungarianMatcher
from eend.eend.pytorch_backend.eend_m2f.fastinst_matcher import HungarianMatcher as fastHungarianMatcher
from eend.eend.pytorch_backend.eend_m2f.criterion import SetCriterion

class EendM2F(nn.Module):
    def __init__(self,
        backbone: nn.Module,
        pixel_decoder: nn.Module,
        transformer_decoder: nn.Module,
        #criterion: nn.Module,
        num_queries: int=50,
        deep_supervision: bool=True,
        no_object_weight: float=0.1,
        class_weight: float=2.0,
        mask_weight: float=5.0,
        dice_weight: float=5.0,
        location_weight: float=1000.0,
        proposal_weight: float=20.0,
        train_num_points: int=12544,
        oversample_ratio: float=3.0,
        importance_sample_ratio: float=0.75,
    ):
        super(EendM2F,self).__init__()

        self.backbone = backbone
        self.pixel_decoder = pixel_decoder
        self.transformer_decoder = transformer_decoder
        #self.criterion = criterion
        self.num_queries = num_queries
        #self.overlap_threshold = overlap_threshold
        #self.object_mask_threshold = object_mask_threshold

        #backbone = build_backbone(cfg)
        #sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())
        
        ## loss args
        self.deep_supervision = deep_supervision
        self.no_object_weight = no_object_weight
        self.class_weight = class_weight
        self.mask_weight = mask_weight
        self.dice_weight = dice_weight
        self.location_weight = location_weight
        self.proposal_weight = proposal_weight

        self.train_num_points = train_num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        
        if self.transformer_decoder.transformer_decoder_name == "MultiScaleMaskedTransformerDecoder":
            # building criterion
            matcher = mask2formerHungarianMatcher(
                cost_class=self.class_weight,
                cost_mask=self.mask_weight,
                cost_dice=self.dice_weight,
                num_points=self.train_num_points,
            )
        elif self.transformer_decoder.transformer_decoder_name == "FastInstDecoder":
            # building criterion
            matcher = fastHungarianMatcher(
                cost_class=self.class_weight,
                cost_mask=self.mask_weight,
                cost_dice=self.dice_weight,
                cost_location=self.location_weight,
                num_points=self.train_num_points,
            )

        weight_dict = {"loss_ce": self.class_weight, "loss_mask": self.mask_weight, "loss_dice": self.dice_weight}

        if self.deep_supervision:
            dec_layers = self.transformer_decoder.dec_layer
            aux_weight_dict = {}
            for i in range(2 * dec_layers):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        weight_dict.update({"loss_proposal": self.proposal_weight})

        losses = ["labels", "masks"]
        criterion = SetCriterion(
            self.transformer_decoder.num_classes,
            matcher=matcher,
            weight_dict=self.weight_dict,
            eos_coef=self.no_object_weight,
            losses=losses,
            num_points=self.train_num_points,
            oversample_ratio=self.oversample_ratio,
            importance_sample_ratio=self.importance_sample_ratio,
        )
        
    def forward(self,)
