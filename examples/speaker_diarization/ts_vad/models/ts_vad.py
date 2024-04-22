# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field
import sys
import contextlib
from typing import Optional
from collections import defaultdict
from omegaconf import DictConfig
from argparse import Namespace

import numpy as np
import torch
import torchaudio
import torch.nn as nn

from fairseq.dataclass import FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model

from examples.speaker_diarization.ts_vad.tasks.ts_vad import TSVADTaskConfig
from examples.speaker_diarization.ts_vad.models.modules.speakerEncoder import (
    ECAPA_TDNN,
    PreEmphasis,
)
from examples.speaker_diarization.ts_vad.models.modules.ecapa_tdnn_wespeaker import ECAPA_TDNN_GLOB_c1024 
from examples.speaker_diarization.ts_vad.models.modules.resnet_wespeaker import ResNet34 
from examples.speaker_diarization.ts_vad.models.modules.cam_pplus_wespeaker import CAMPPlus
from examples.speaker_diarization.ts_vad.models.modules.postional_encoding import (
    PositionalEncoding,
)
from examples.speaker_diarization.ts_vad.models.modules.WavLM import WavLM, WavLMConfig
from examples.speaker_diarization.ts_vad.models.modules.self_att import (
    CoAttention_Simple,
)
from examples.speaker_diarization.ts_vad.models.modules.joint_speaker_det import (
    JointSpeakerDet,
)
from examples.speaker_diarization.ts_vad.models.modules.batch_norm import BatchNorm1D

logger = logging.getLogger(__name__)


@dataclass
class TSVADConfig(FairseqDataclass):
    speaker_encoder_path: Optional[str] = field(
        default=None, metadata={"help": "path to pretrained speaker encoder path."}
    )
    speech_encoder_path: Optional[str] = field(
        default=None, metadata={"help": "path to pretrained speech encoder path."}
    )
    freeze_speech_encoder_updates: int = field(
        default=10000, metadata={"help": "updates to freeze speech encoder."}
    )

    num_attention_head: int = field(
        default=8, metadata={"help": "number of attention head."}
    )
    num_transformer_layer: int = field(
        default=3, metadata={"help": "number of transformer layer."}
    )
    transformer_embed_dim: int = field(
        default=384, metadata={"help": "transformer dimension."}
    )
    transformer_ffn_embed_dim: int = field(
        default=1536, metadata={"help": "transformer dimension."}
    )
    speaker_embed_dim: int = field(
        default=192, metadata={"help": "speaker embedding dimension."}
    )
    dropout: float = field(default=0.1, metadata={"help": "dropout prob"})
    use_jsd_block: bool = field(
        default=False, metadata={"help": "number of JSD block."}
    )
    use_lstm_single: bool = field(
        default=False, metadata={"help": "use lstm for single"}
    )
    use_lstm_multi: bool = field(default=False, metadata={"help": "use lstm for multi"})
    use_spk_embed: bool = field(
        default=True, metadata={"help": "whether to use speaker embedding"}
    )
    add_ind_proj: bool = field(
        default=False, metadata={"help": "whether to add projection for each speaker"}
    )

# modify from https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/zipformer/zipformer.py#L1352C1-L1372C19
class SimpleUpsample(nn.Module):
    """
    A very simple form of upsampling that mostly just repeats the input, but
    also adds a position-specific bias.
    """

    def __init__(self, num_channels: int, upsample: int):
        super(SimpleUpsample, self).__init__()
        self.upsample = upsample

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size,seq_len,num_channels)
        Returns a tensor of shape
           ( batch_size,(seq_len*upsample), num_channels)
        """
        upsample = self.upsample
        (batch_size, seq_len,num_channels) = src.shape
        src = src.unsqueeze(1).expand(batch_size, upsample,seq_len,num_channels)
        src = src.reshape(batch_size, upsample*seq_len, num_channels)
        return src

class SpeechFeatUpsample(nn.Module):
    def __init__(self,speaker_embed_dim: int, upsample: int):
        super(SpeechFeatUpsample,self).__init__()
        self.speaker_embed_dim=speaker_embed_dim
        # here 2560 means it is feature dimension  before pool layer of resnet34_wespeaker model dimension
        #nn.Conv1d(2560, cfg.speaker_embed_dim, 5, stride=stride, padding=2),
        self.linear = nn.Linear(2560, speaker_embed_dim, bias=True)
        self.up = SimpleUpsample(speaker_embed_dim, upsample )
        self.batchnorm = BatchNorm1D(num_features=speaker_embed_dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        x = x.permute(0,2,1) #(B,F,T) -> (B,T,F)
        x = self.linear(x) # (B,T,F) -> (B,T,D)
        x = self.up(x) # (B,T,D) -> (B,2T,D)
        x = x.permute(0,2,1) # (B,2T,D)->(B,D,2T)
        x = self.batchnorm(x)
        x = self.act(x)
        return x #(B,D,2T)

class SpeechFeatUpsample2(nn.Module):
      def __init__(self, speaker_embed_dim: int, upsample: int):
          super(SpeechFeatUpsample2, self).__init__()
          self.speaker_embed_dim = speaker_embed_dim
          # here 2560 means it is feature dimension  before pool layer of resnet34_wespeaker model dimension
          self.up = nn.ConvTranspose1d(2560, speaker_embed_dim, 5, stride=upsample, padding=2,output_padding=1)
          self.batchnorm = BatchNorm1D(num_features=speaker_embed_dim)
          self.act = nn.ReLU()
      def forward(self, x: torch.Tensor)-> torch.Tensor:
          x = self.up(x) # (B,F,T) -> (B,D,2T)
          x = self.batchnorm(x)
          x = self.act(x)
          return x #(B,D,2T)


@register_model("ts_vad", dataclass=TSVADConfig)
class TSVADModel(BaseFairseqModel):
    def __init__(
        self,
        cfg: TSVADConfig,
        task_cfg: TSVADTaskConfig,
    ) -> None:
        super().__init__()
        # Speaker Encoder
        self.use_spk_embed = cfg.use_spk_embed
        if not self.use_spk_embed:
            self.speaker_encoder = ECAPA_TDNN(C=1024)
            self.speaker_encoder.train()
            self.load_speaker_encoder(cfg.speaker_encoder_path)
            for param in self.speaker_encoder.parameters():
                param.requires_grad = False
        self.rs_dropout = nn.Dropout(p=cfg.dropout)

        self.label_rate = task_cfg.label_rate
        if task_cfg.label_rate != 25:
            assert (
                task_cfg.speech_encoder_type == "ecapa"
                or task_cfg.speech_encoder_type == "cam++"
                or task_cfg.speech_encoder_type == "ecapa_wespeaker"
                or task_cfg.speech_encoder_type == "resnet34_wespeaker"
            ), "Only support ecapa and cam++ for label rate not 25"

        # Speech Encoder
        self.speech_encoder_type = task_cfg.speech_encoder_type
        sample_times = 16000 / task_cfg.sample_rate
        #self.torchfbank = torch.nn.Sequential(
        #        PreEmphasis(),
        #        torchaudio.transforms.MelSpectrogram(
        #            sample_rate=task_cfg.sample_rate,
        #            n_fft=512,
        #            win_length=400,
        #            hop_length=160,
        #            f_min=20,
        #            f_max=7600,
        #            window_fn=torch.hamming_window,
        #            n_mels=80,
        #        ),
        #)
        if self.speech_encoder_type == "wavlm":
            checkpoint = torch.load(cfg.speech_encoder_path, map_location="cuda")
            wavlm_cfg = WavLMConfig(checkpoint["cfg"])
            wavlm_cfg.encoder_layers = 6
            self.speech_encoder = WavLM(wavlm_cfg)
            self.speech_encoder.train()
            self.speech_encoder.load_state_dict(checkpoint["model"], strict=False)
            self.speech_down = nn.Sequential(
                nn.Conv1d(
                    768,
                    cfg.speaker_embed_dim,
                    5,
                    stride=int(2 // sample_times),
                    padding=2,
                ),
                BatchNorm1D(num_features=cfg.speaker_embed_dim),
                nn.ReLU(),
            )
        elif self.speech_encoder_type == "ecapa":
            self.speech_encoder = ECAPA_TDNN(
                C=1024, dropout=cfg.dropout, speech_encoder=True
            )
            self.speech_encoder.train()
            if cfg.speech_encoder_path is not None:
                self.load_speaker_encoder(
                    cfg.speech_encoder_path, module_name="speech_encoder"
                )
            else:
                logger.warn("Not load speech encoder!!")

            if not task_cfg.embed_input:
                stride = int(4 // sample_times) if task_cfg.label_rate == 25 else 1
                self.speech_down = nn.Sequential(
                    nn.Conv1d(1536, cfg.speaker_embed_dim, 5, stride=stride, padding=2),
                    BatchNorm1D(num_features=cfg.speaker_embed_dim),
                    nn.ReLU(),
                )
        elif self.speech_encoder_type == "ecapa_wespeaker":
            #ECAPA_TDNN_GLOB_c1024(feat_dim=80,embed_dim=192,pooling_func='ASTP',speech_encoder=True).train()

            self.speech_encoder = ECAPA_TDNN_GLOB_c1024(feat_dim=80,embed_dim=192,pooling_func='ASTP',speech_encoder=True)
            self.speech_encoder.train()
            if cfg.speech_encoder_path is not None:
                self.load_speaker_encoder(
            #        #cfg.speech_encoder_path, module_name="speech_encoder"
                    cfg.speech_encoder_path, module_name="speech_encoder"
                )
            else:
                logger.warn("Not load speech encoder!!")

            if not task_cfg.embed_input:
                stride = int(4 // sample_times) if task_cfg.label_rate == 25 else 1
                ## the input shape of self.speech_down except is (B,F,T)
                self.speech_down = nn.Sequential(
                    # here 1536 means it is feature dimension  before pool layer of ecapa_wespeaker model dimension
                    nn.Conv1d(1536, cfg.speaker_embed_dim, 5, stride=stride, padding=2),
                    BatchNorm1D(num_features=cfg.speaker_embed_dim),
                    nn.ReLU(),
                )


        elif self.speech_encoder_type == "fbank":
            self.torchfbank = torch.nn.Sequential(
                PreEmphasis(),
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=task_cfg.sample_rate,
                    n_fft=512,
                    win_length=400,
                    hop_length=160,
                    f_min=20,
                    f_max=7600,
                    window_fn=torch.hamming_window,
                    n_mels=80,
                ),
            )
            self.speech_up = nn.Sequential(
                nn.Conv1d(
                    80,
                    cfg.speaker_embed_dim,
                    5,
                    stride=int(4 // sample_times),
                    padding=2,
                ),
                BatchNorm1D(num_features=cfg.speaker_embed_dim),
                nn.ReLU(),
            )
        elif self.speech_encoder_type == "cam++": # its input is fbank , it is extract from data/ts_vad_dataset.py
            self.speech_encoder = CAMPPlus(feat_dim=80,embedding_size=192)# embedding_size is from pretrain model embedding_size
            self.speech_encoder.train()
            self.load_speaker_encoder(
                cfg.speech_encoder_path, module_name="speech_encoder"
            )
            # input of cam++ model is fbank, means that 1s has 100 frames
            # we set target label rate is 25, means that 1s has 25 frames 
            # cam++ model downsample scale is 2, so frame rate is cam++ model output is 50, so I should set stride equal to 2.
            stride = int(2 // sample_times) if task_cfg.label_rate == 25 else 1
            ## the input shape of self.speech_down except is (B,F,T) 
            self.speech_down = nn.Sequential(
                # here 512 means it is feature dimension  before pool layer of cam++(it is also from wespeaker ) model.
                nn.Conv1d(512, cfg.speaker_embed_dim, 5, stride=stride, padding=2),
                BatchNorm1D(num_features=cfg.speaker_embed_dim),
                nn.ReLU(),
            )

        #resnet_wespeaker
        elif self.speech_encoder_type == "resnet34_wespeaker":
            self.speech_encoder = ResNet34(feat_dim=80, embed_dim=256, pooling_func="TSTP", two_emb_layer=False,speech_encoder=True)
            self.speech_encoder.train()
            self.load_speaker_encoder(
                cfg.speech_encoder_path, module_name="speech_encoder"
            )
            # input of cam++ model is fbank, means that 1s has 100 frames
            # we set target label rate is 25, means that 1s has 25 frames
            # resnet34_wespeaker model downsample scale is 8, so frame rate is cam++ model output is 12.5, so I should set stride equal to 2.
            #stride = int(2 // sample_times) if task_cfg.label_rate == 25 else 1
            upsample = 2 if task_cfg.label_rate == 25 else 1
            ## the input shape of self.speech_up except is (B,T,F)
            #self.speech_up = SpeechFeatUpsample(speaker_embed_dim=cfg.speaker_embed_dim, upsample=upsample)
            self.speech_up = SpeechFeatUpsample2(speaker_embed_dim=cfg.speaker_embed_dim, upsample=upsample)

        # Projection
        if cfg.speaker_embed_dim * 2 != cfg.transformer_embed_dim:
            self.proj_layer = nn.Linear(
                cfg.speaker_embed_dim * 2, cfg.transformer_embed_dim
            )
        else:
            self.proj_layer = None

        # TS-VAD Backend
        if cfg.use_lstm_single:
            self.single_backend = nn.LSTM(
                input_size=cfg.transformer_embed_dim,
                hidden_size=cfg.transformer_embed_dim // 2,
                num_layers=cfg.num_transformer_layer,
                dropout=cfg.dropout,
                bidirectional=True,
            )
        else:
            self.single_backend = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=cfg.transformer_embed_dim,
                    dim_feedforward=cfg.transformer_ffn_embed_dim,
                    nhead=cfg.num_attention_head,
                    dropout=cfg.dropout,
                ),
                num_layers=cfg.num_transformer_layer,
            )

        self.pos_encoder = PositionalEncoding(
            cfg.transformer_embed_dim,
            dropout=cfg.dropout,
            max_len=(task_cfg.rs_len * self.label_rate),
        )
        self.use_jsd_block = cfg.use_jsd_block
        if self.use_jsd_block:
            self.multi_backend = JointSpeakerDet(cfg)

            # final projection
            self.fc = nn.Linear(cfg.transformer_embed_dim, task_cfg.max_num_speaker)
        else:
            self.backend_down = nn.Sequential(
                nn.Conv1d(
                    cfg.transformer_embed_dim * task_cfg.max_num_speaker,
                    cfg.transformer_embed_dim,
                    5,
                    stride=1,
                    padding=2,
                ),
                BatchNorm1D(num_features=cfg.transformer_embed_dim),
                nn.ReLU(),
            )
            if cfg.use_lstm_multi:
                self.multi_backend = nn.LSTM(
                    input_size=cfg.transformer_embed_dim,
                    hidden_size=cfg.transformer_embed_dim // 2,
                    num_layers=cfg.num_transformer_layer,
                    dropout=cfg.dropout,
                    bidirectional=True,
                )
            else:
                self.multi_backend = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=cfg.transformer_embed_dim,
                        dim_feedforward=cfg.transformer_ffn_embed_dim,
                        nhead=cfg.num_attention_head,
                        dropout=cfg.dropout,
                    ),
                    num_layers=cfg.num_transformer_layer,
                )
            # final projection
            self.add_ind_proj = cfg.add_ind_proj
            if cfg.add_ind_proj:
                self.pre_fc = nn.ModuleList(
                    [
                        nn.Linear(cfg.transformer_embed_dim, 192)
                        for _ in range(task_cfg.max_num_speaker)
                    ]
                )
                self.fc = nn.Linear(192, 1)
            else:
                self.fc = nn.Linear(cfg.transformer_embed_dim, task_cfg.max_num_speaker)

        self.loss = nn.BCEWithLogitsLoss()
        self.m = nn.Sigmoid()

        if task_cfg.support_mc:
            self.co_attn = CoAttention_Simple(
                out_channels=192, embed_dim=256, num_heads=256
            )

        # others
        self.label_rate = task_cfg.label_rate
        self.freeze_speech_encoder_updates = cfg.freeze_speech_encoder_updates
        self.inference = task_cfg.inference
        self.max_num_speaker = task_cfg.max_num_speaker
        self.embed_input = task_cfg.embed_input
        self.scale_factor = 0.04 / task_cfg.embed_shift
        self.use_lstm_single = cfg.use_lstm_single
        self.use_lstm_multi = cfg.use_lstm_multi
        self.num_updates = 0

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""

        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    # B: batchsize, T: number of frames (1 frame = 0.04s)
    # Obtain the reference speech represnetation(it should be mix speech representation)
    def rs_forward(self, x, max_len, fix_encoder=False):  # B, 25 * T
        B = x.size(0)
        T = x.size(1)
        if self.speech_encoder_type == "wavlm":
            with torch.no_grad() if fix_encoder else contextlib.ExitStack():
                x = self.speech_encoder.extract_features(x)[0]
            x = x.view(B, -1, 768)  # B, 50 * T, 768
            x = x.transpose(1, 2)
            x = self.speech_down(x)
        elif self.speech_encoder_type == "ecapa":
            with torch.no_grad() if fix_encoder else contextlib.ExitStack():
                if self.embed_input:
                    x = self.speech_encoder(x.view(B * T, -1)).view(B, T, -1)
                    x = x.transpose(1, 2)
                else:
                    x = self.speech_encoder(x, get_time_out=True)
            if not self.embed_input:
                x = self.speech_down(x)
        elif self.speech_encoder_type == "ecapa_wespeaker":
            with torch.no_grad()if fix_encoder else contextlib.ExitStack():
                x = self.speech_encoder(x, get_time_out=True)
            x = self.speech_down(x)


        elif self.speech_encoder_type == "fbank":
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    x = self.torchfbank(x) + 1e-6
                    x = x.log()
                    x = x - torch.mean(x, dim=-1, keepdim=True)
            x = self.speech_up(x)
        elif self.speech_encoder_type == "cam++": # its input is fbank, it is extract from data/ts_vad_dataset.py
            with torch.no_grad() if fix_encoder else contextlib.ExitStack():
                x = self.speech_encoder(x, get_time_out=True)
            x = self.speech_down(x)

        elif self.speech_encoder_type == "resnet34_wespeaker": # its input is fbank,it is extract from data/ts_vad_dataset.py
            with torch.no_grad() if fix_encoder else contextlib.ExitStack():
                x = self.speech_encoder(x, get_time_out=True)
            x = self.speech_up(x)

        assert (
            x.size(-1) - max_len <= 2 and x.size(-1) - max_len >= -1
        ), f"label and input diff: {x.size(-1) - max_len}"
        if x.size(-1) - max_len == -1:
            x = nn.functional.pad(x, (0, 1))
        x = x[:, :, :max_len]# (B,D,T)
        x = x.transpose(1, 2) #(B,T,D)

        return x

    # Obtain the target speaker represnetation(utterance level speaker embedding)
    def ts_forward(self, x):  # B, 4, 80, T * 100
        if self.use_spk_embed:
            return self.rs_dropout(x)
        B, _, D, T = x.shape
        x = x.view(B * self.max_num_speaker, D, T)
        x = self.speaker_encoder.forward(x)
        x = x.view(B, self.max_num_speaker, -1)  # B, 4, 192
        return x

    # Obtain the target speaker represnetation for joint model
    def ts_joint_forward(self, x):  # B, 4, 80, T * 100
        if self.use_spk_embed:
            return self.rs_dropout(x)
        x = self.speaker_encoder.forward(x)
        return x

    # Combine for ts-vad results
    def cat_single_forward(self, rs_embeds, ts_embeds):
        # Extend ts_embeds for time alignemnt
        ts_embeds = ts_embeds.unsqueeze(2)  # B, 4, 1, 256
        ts_embeds = ts_embeds.repeat(1, 1, rs_embeds.shape[1], 1)  # B, 4, T, 256  ## repeat T to cat mix frame-level information
        B, _, T, _ = ts_embeds.shape

        # Transformer for single speaker
        cat_embeds = []
        for i in range(self.max_num_speaker):
            ts_embed = ts_embeds[:, i, :, :]  # B, T, 256
            cat_embed = torch.cat(
                (ts_embed, rs_embeds), 2
            )  # B, T, 256 + B, T, 256 -> B, T, 512
            if self.proj_layer is not None:
                cat_embed = self.proj_layer(cat_embed)
            cat_embed = cat_embed.transpose(0, 1)  # T, B, 512
            cat_embed = self.pos_encoder(cat_embed)
            if self.use_lstm_single:
                cat_embed, _ = self.single_backend(cat_embed)  # T, B, 512
            else:
                cat_embed = self.single_backend(cat_embed)  # T, B, 512
            cat_embed = cat_embed.transpose(0, 1)  # B, T, 512
            cat_embeds.append(cat_embed)
        cat_embeds = torch.stack(cat_embeds)  # 4, B, T, 384
        cat_embeds = torch.permute(cat_embeds, (1, 0, 3, 2))  # B, 4, 384, T
        # Combine the outputs
        cat_embeds = cat_embeds.reshape(B, -1, T)  # B, 4 * 384, T
        return cat_embeds

    def cat_multi_forward(self, cat_embeds):
        B, _, T = cat_embeds.size()
        # Downsampling
        if self.use_jsd_block:
            cat_embeds = cat_embeds.reshape(
                B, self.max_num_speaker, T, -1
            )  # B, 4 * 512, T
            cat_embeds = self.multi_backend(cat_embeds)  # B, S, T, D
            cat_embeds = cat_embeds.reshape((B, T, -1))
        else:
            cat_embeds = self.backend_down(cat_embeds)  # B, 384, T
            # Transformer for multiple speakers
            cat_embeds = self.pos_encoder(torch.permute(cat_embeds, (2, 0, 1)))
            if self.use_lstm_multi:
                cat_embeds, _ = self.multi_backend(cat_embeds)  # T, B, 384
            else:
                cat_embeds = self.multi_backend(cat_embeds)  # T, B, 384

            cat_embeds = cat_embeds.transpose(0, 1)

        return cat_embeds

    def calculate_loss(self, outs, labels, labels_len):
        total_loss = 0

        for i in range(labels_len.size(0)):
            total_loss += self.loss(
                outs[i, :, : labels_len[i]], labels[i, :, : labels_len[i]]
            )

        outs_prob = self.m(outs)
        outs_prob = outs_prob.data.cpu().numpy()

        return total_loss / labels_len.size(0), outs_prob

    def forward(
        self,
        ref_speech: torch.Tensor,
        target_speech: torch.Tensor,
        labels: torch.Tensor,
        labels_len: torch.Tensor = None,
        file_path=None,
        speaker_ids=None,
        start=None,
        extract_features=False,
        fix_encoder=True,
    ):
        # if self.embed_input:
        #     labels = nn.functional.interpolate(labels, scale_factor=self.scale_factor)
        #     labels_len = (labels_len * self.scale_factor).type(torch.LongTensor)
        #     start = [int(i * self.scale_factor) for i in start]
        # import pdb; pdb.set_trace()
        # rs_embeds_list = []
        # for i in range(ref_speech.size(1)):
        # rs_embeds_list.append(self.rs_forward(ref_speech[:, i], labels.size(-1), fix_encoder=(self.num_updates < self.freeze_speech_encoder_updates and fix_encoder)))
        # import pdb; pdb.set_trace()
        # rs_embeds = torch.stack(rs_embeds_list, dim=1)
        rs_embeds = self.rs_forward(
            ref_speech,
            labels.size(-1),
            fix_encoder=(
                self.num_updates < self.freeze_speech_encoder_updates and fix_encoder
            ),
        )
        # rs_embeds = self.co_attn(rs_embeds) # B, C, T, D --> B, T, D

        ts_embeds = self.ts_forward(target_speech)
        cat_embeds = self.cat_single_forward(rs_embeds, ts_embeds)
        outs_pre = self.cat_multi_forward(cat_embeds)

        if self.add_ind_proj:
            outs_pre = torch.stack(
                [self.pre_fc[i](outs_pre) for i in range(self.max_num_speaker)], dim=1
            )
            outs = (
                self.fc(outs_pre.view(-1, labels.size(-1), 192))
                .view(ref_speech.size(0), self.max_num_speaker, labels.size(-1))
                .transpose(1, 2)
            )
        else:
            outs = self.fc(outs_pre)  # B T 3

        outs = outs.transpose(1, 2)  # B 3 T

        loss, outs_prob = self.calculate_loss(outs, labels, labels_len)
        result = {"losses": {"diar": loss}}

        mi, fa, cf, acc, der = self.calc_diarization_result(
            outs_prob.transpose((0, 2, 1)), labels.transpose(1, 2), labels_len
        )

        result["DER"] = der
        result["ACC"] = acc
        result["MI"] = mi
        result["FA"] = fa
        result["CF"] = cf

        if extract_features:
            # if self.add_ind_proj:
            return outs_pre.transpose(1, 2), result, outs_prob
            # else:
            #     return outs, mi, fa, cf, acc, der

        if self.inference:
            res_dict = defaultdict(lambda: defaultdict(list))
            B, _, _ = outs.shape
            for b in range(B):
                for t in range(labels_len[b]):
                    n = max(speaker_ids[b])
                    for i in range(n):
                        id = speaker_ids[b][i]
                        name = file_path[b]
                        out = outs_prob[b, i, t]
                        t0 = start[b]
                        res_dict[str(name) + "-" + str(id)][t0 + t].append(out)

            return result, res_dict

        return result

    @classmethod
    def build_model(cls, cfg: TSVADConfig, task: TSVADTaskConfig):
        """Build a new model instance."""

        model = TSVADModel(cfg, task.cfg)
        return model

    def load_speaker_encoder(self, model_path, module_name="speaker_encoder"):
        loadedState = torch.load(model_path, map_location="cuda")
        selfState = self.state_dict()
        for name, param in loadedState.items():
            origname = name

            if (
                module_name == "speaker_encoder"
                and hasattr(self.speaker_encoder, "bn1")
                and isinstance(self.speaker_encoder.bn1, BatchNorm1D)
                and ".".join(name.split(".")[:-1]) + ".running_mean" in loadedState
            ):
                name = ".".join(name.split(".")[:-1]) + ".bn." + name.split(".")[-1]

            if (
                module_name == "speech_encoder"
                and hasattr(self.speech_encoder, "bn1")
                and isinstance(self.speech_encoder.bn1, BatchNorm1D)
                and ".".join(name.split(".")[:-1]) + ".running_mean" in loadedState
            ):
                name = ".".join(name.split(".")[:-1]) + ".bn." + name.split(".")[-1]

            if name.startswith("speaker_encoder"):
                name = name.replace("speaker_encoder", module_name)
            else:
                name = f"{module_name}." + name

            if name not in selfState:
                logger.warn("%s is not in the model." % origname)
                continue
            if selfState[name].size() != loadedState[origname].size():
                sys.stderr.write(
                    "Wrong parameter length: %s, model: %s, loaded: %s"
                    % (origname, selfState[name].size(), loadedState[origname].size())
                )
                continue
            selfState[name].copy_(param)
#    def load_speaker_encoder(self, model_path, module_name="speaker_encoder"):
#        loadedState = torch.load(model_path, map_location="cuda")
#        selfState = self.state_dict()
#        for name, param in loadedState.items():
#            origname = name
#            name = f"{module_name}." + name
#            if name not in selfState:
#                logger.warn("%s is not in the model." % origname)
#                continue
#            selfState[name].copy_(param)


    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    @staticmethod
    def calc_diarization_error(pred, label, length):
        # Note (jiatong): Credit to https://github.com/hitachi-speech/EEND

        (batch_size, max_len, num_output) = label.size()
        # mask the padding part
        mask = np.zeros((batch_size, max_len, num_output))
        for i in range(batch_size):
            mask[i, : length[i], :] = 1

        # pred and label have the shape (batch_size, max_len, num_output)
        label_np = label.data.cpu().numpy().astype(int)
        # pred_np = (pred.data.cpu().numpy() > 0).astype(int)
        pred_np = (pred > 0.5).astype(int)
        label_np = label_np * mask
        pred_np = pred_np * mask
        length = length.data.cpu().numpy()

        # compute speech activity detection error
        n_ref = np.sum(label_np, axis=2)
        n_sys = np.sum(pred_np, axis=2)
        speech_scored = float(np.sum(n_ref > 0))
        speech_miss = float(np.sum(np.logical_and(n_ref > 0, n_sys == 0)))
        speech_falarm = float(np.sum(np.logical_and(n_ref == 0, n_sys > 0)))

        # compute speaker diarization error
        speaker_scored = float(np.sum(n_ref))
        speaker_miss = float(np.sum(np.maximum(n_ref - n_sys, 0)))
        speaker_falarm = float(np.sum(np.maximum(n_sys - n_ref, 0)))
        n_map = np.sum(np.logical_and(label_np == 1, pred_np == 1), axis=2)
        speaker_error = float(np.sum(np.minimum(n_ref, n_sys) - n_map))
        correct = float(1.0 * np.sum((label_np == pred_np) * mask) / num_output)
        num_frames = np.sum(length)

        return (
            correct,
            num_frames,
            speech_scored,
            speech_miss,
            speech_falarm,
            speaker_scored,
            speaker_miss,
            speaker_falarm,
            speaker_error,
        )

    @staticmethod
    def calc_diarization_result(outs_prob, labels, labels_len):
        # DER
        (
            correct,
            num_frames,
            speech_scored,
            speech_miss,
            speech_falarm,
            speaker_scored,
            speaker_miss,
            speaker_falarm,
            speaker_error,
        ) = TSVADModel.calc_diarization_error(outs_prob, labels, labels_len)

        if speech_scored == 0 or speaker_scored == 0:
            logger.warn("All labels are zero")
            return 0, 0, 0, 0, 0

        _, _, mi, fa, cf, acc, der = (
            speech_miss / speech_scored,
            speech_falarm / speech_scored,
            speaker_miss / speaker_scored,
            speaker_falarm / speaker_scored,
            speaker_error / speaker_scored,
            correct / num_frames,
            (speaker_miss + speaker_falarm + speaker_error) / speaker_scored,
        )

        return mi, fa, cf, acc, der

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
