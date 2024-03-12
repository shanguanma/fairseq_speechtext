# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional
from omegaconf import DictConfig
from argparse import Namespace

import torch
import torch.nn as nn

from fairseq.dataclass import ChoiceEnum

from examples.speaker_diarization.ts_vad.models.modules.modules import Conv1D, Conv1DBlock
from examples.speaker_diarization.ts_vad.models.modules.batch_norm import BatchNorm1D
from examples.speaker_diarization.ts_vad.tasks.ts_vad import TSVADTaskConfig
from examples.speaker_diarization.ts_vad.models.spex_plus import SpexPlusConfig, SpexPlusModel

logger = logging.getLogger(__name__)


INPUT_TYPE = ChoiceEnum(["mask", "tcn"])

class ExtractionModule(SpexPlusModel):
    def __init__(
        self,
        cfg: SpexPlusConfig,
        task_cfg: TSVADTaskConfig,
        spk_embed,
        max_num_speaker = None,
    ) -> None:
        super().__init__(cfg, task_cfg, spk_embed)
        if max_num_speaker is None:
            max_num_speaker = task_cfg.max_num_speaker
        # empty module
        self.speech_down = nn.Sequential(
            nn.Conv1d(cfg.enc_out_size * max_num_speaker, cfg.enc_out_size, 5, stride=1, padding=2),
            BatchNorm1D(num_features=cfg.enc_out_size),
            nn.ReLU(),
        )

        self.conv_blocks_combine = nn.ModuleList([
                Conv1DBlock(
                    in_channels=cfg.enc_out_size, 
                    conv_channels=cfg.conv_out_channels, 
                    kernel_size=cfg.conv_kernel_size, 
                    norm=cfg.conv_block_norm, 
                    dilation=1
                ) for _ in range(cfg.total_conv_block_num)
            ]
        )
        self.conv_blocks_combine_other = nn.Sequential(*[
                self._build_blocks(
                    num_blocks=cfg.conv_block_num, 
                    in_channels=cfg.enc_out_size, 
                    conv_channels=cfg.conv_out_channels, 
                    kernel_size=cfg.conv_kernel_size, 
                    norm=cfg.conv_block_norm
                ) for _ in range(cfg.total_conv_block_num)
            ]
        )

        # mask
        del self.mask1, self.mask2, self.mask3 
        self.mask_module1 = nn.ModuleList([Conv1D(cfg.enc_out_size, cfg.enc_in_channels, 1) for _ in range(max_num_speaker)])
        self.mask_module2 = nn.ModuleList([Conv1D(cfg.enc_out_size, cfg.enc_in_channels, 1) for _ in range(max_num_speaker)])
        self.mask_module3 = nn.ModuleList([Conv1D(cfg.enc_out_size, cfg.enc_in_channels, 1) for _ in range(max_num_speaker)])

        # others
        self.inference = task_cfg.inference
        self.max_num_speaker = max_num_speaker
        self.add_multi_scale = cfg.add_multi_scale

    def mask_module(self, y, audio_features, max_len, speaker_idx):
        # Multi-scale Decoder
        pre_mask1 = self.mask_module1[speaker_idx](y)
        m1 = self.non_linear(pre_mask1)
        s1 = audio_features[0] * m1

        ests = self.decoder_1d_1(s1, squeeze=True)
        if max_len != ests.size(1):
            ests = torch.nn.functional.pad(ests, (0, max_len - ests.size(1), 0, 0))

        if self.add_multi_scale:
            pre_mask2 = self.mask_module2[speaker_idx](y)
            m2 = self.non_linear(pre_mask2)

            pre_mask3 = self.mask_module3[speaker_idx](y)
            m3 = self.non_linear(pre_mask3)

            s2 = audio_features[1] * m2
            s3 = audio_features[2] * m3

            ests2 = self.decoder_1d_2(s2, squeeze=True)[:, :max_len]
            ests3 = self.decoder_1d_3(s3, squeeze=True)[:, :max_len]

            if max_len != ests2.size(1):
                ests2 = torch.nn.functional.pad(ests2, (0, max_len - ests2.size(1), 0, 0))
            if max_len != ests.size(1):
                ests3 = torch.nn.functional.pad(ests3, (0, max_len - ests3.size(1), 0, 0))

            return [ests, ests2, ests3], [m1, m2, m3]
        else:
            return [ests], [m1]

    def forward(
        self,
        mix_speech: torch.Tensor,
        y: torch.Tensor,
        aux: torch.Tensor,
        audio_features: torch.Tensor,
        after_tsvad_layer: torch.Tensor = -1,
    ):
        tcns = []
        for i in range(self.max_num_speaker):
            tcns.append(self.separator(y, aux[:, i]))

        tcns = torch.stack(tcns, dim=1) # B, 4, 256, T
        B, _, _, T = tcns.size()
        tcns = tcns.reshape(B, -1, T)
        tcns = self.speech_down(tcns) # B, 256, T

        after_diar_input = None
        for i, (conv_block, conv_block_other) in enumerate(zip(self.conv_blocks_combine, self.conv_blocks_combine_other)):
            tcns = conv_block(tcns)
            tcns = conv_block_other(tcns)
            if i == after_tsvad_layer:
                after_diar_input = tcns

        if after_tsvad_layer == -1:
            after_diar_input = tcns

        ests_speech_list = []
        mask_list = []
        for i in range(self.max_num_speaker):
            ests_speech, mask = self.mask_module(tcns, audio_features, mix_speech.size(-1), i)
            ests_speech_list.append(ests_speech)
            mask_list.append(mask)

        return ests_speech_list, mask_list, after_diar_input

    def load_state_dict(
        self,
        state_dict,
        strict=True,
        model_cfg: Optional[DictConfig] = None,
        args: Optional[Namespace] = None,
    ):
        """Copies parameters and buffers from *state_dict* into this module and
        its descendants.

        Overrides the method in :class:`nn.Module`. Compared with that method
        this additionally "upgrades" *state_dicts* from old checkpoints.
        """
        try:
            return super().load_state_dict(state_dict, strict, model_cfg, args)
        except Exception as e:
            logger.warn(e)
            return super().load_state_dict(state_dict, False, model_cfg, args)
