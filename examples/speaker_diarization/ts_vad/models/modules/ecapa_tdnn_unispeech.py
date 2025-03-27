#!/usr/bin/env python3
# Author: Duo MA
# Email: maduo@cuhk.edu.cn
# modified from https://github.com/microsoft/UniSpeech/blob/main/downstreams/speaker_verification/models/ecapa_tdnn.py
# I add specify pretrain model case i.e.:wavlm, then remove s3prl related code.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as trans
from examples.speaker_diarization.ts_vad.models.modules.util_s3prl import UpstreamExpert
from examples.speaker_diarization.ts_vad.models.modules.WavLM import WavLM, WavLMConfig

''' Res2Conv1d + BatchNorm1d + ReLU
'''


class Res2Conv1dReluBn(nn.Module):
    '''
    in_channels == out_channels == channels
    '''

    def __init__(self, channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, scale=4):
        super().__init__()
        assert channels % scale == 0, "{} % {} != 0".format(channels, scale)
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1

        self.convs = []
        self.bns = []
        for i in range(self.nums):
            self.convs.append(nn.Conv1d(self.width, self.width, kernel_size, stride, padding, dilation, bias=bias))
            self.bns.append(nn.BatchNorm1d(self.width))
        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            # Order: conv -> relu -> bn
            sp = self.convs[i](sp)
            sp = self.bns[i](F.relu(sp))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        out = torch.cat(out, dim=1)

        return out


''' Conv1d + BatchNorm1d + ReLU
'''


class Conv1dReluBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


''' The SE connection of 1D case.
'''


class SE_Connect(nn.Module):
    def __init__(self, channels, se_bottleneck_dim=128):
        super().__init__()
        self.linear1 = nn.Linear(channels, se_bottleneck_dim)
        self.linear2 = nn.Linear(se_bottleneck_dim, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        out = x * out.unsqueeze(2)

        return out


''' SE-Res2Block of the ECAPA-TDNN architecture.
'''


# def SE_Res2Block(channels, kernel_size, stride, padding, dilation, scale):
#     return nn.Sequential(
#         Conv1dReluBn(channels, 512, kernel_size=1, stride=1, padding=0),
#         Res2Conv1dReluBn(512, kernel_size, stride, padding, dilation, scale=scale),
#         Conv1dReluBn(512, channels, kernel_size=1, stride=1, padding=0),
#         SE_Connect(channels)
#     )


class SE_Res2Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, scale, se_bottleneck_dim):
        super().__init__()
        self.Conv1dReluBn1 = Conv1dReluBn(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.Res2Conv1dReluBn = Res2Conv1dReluBn(out_channels, kernel_size, stride, padding, dilation, scale=scale)
        self.Conv1dReluBn2 = Conv1dReluBn(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.SE_Connect = SE_Connect(out_channels, se_bottleneck_dim)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )

    def forward(self, x):
        residual = x
        if self.shortcut:
            residual = self.shortcut(x)

        x = self.Conv1dReluBn1(x)
        x = self.Res2Conv1dReluBn(x)
        x = self.Conv1dReluBn2(x)
        x = self.SE_Connect(x)

        return x + residual


''' Attentive weighted mean and standard deviation pooling.
'''


class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim, attention_channels=128, global_context_att=False):
        super().__init__()
        self.global_context_att = global_context_att

        # Use Conv1d with stride == 1 rather than Linear, then we don't need to transpose inputs.
        if global_context_att:
            self.linear1 = nn.Conv1d(in_dim * 3, attention_channels, kernel_size=1)  # equals W and b in the paper
        else:
            self.linear1 = nn.Conv1d(in_dim, attention_channels, kernel_size=1)  # equals W and b in the paper
        self.linear2 = nn.Conv1d(attention_channels, in_dim, kernel_size=1)  # equals V and k in the paper

    def forward(self, x):

        if self.global_context_att:
            context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
            context_std = torch.sqrt(torch.var(x, dim=-1, keepdim=True) + 1e-10).expand_as(x)
            x_in = torch.cat((x, context_mean, context_std), dim=1)
        else:
            x_in = x

        # DON'T use ReLU here! In experiments, I find ReLU hard to converge.
        alpha = torch.tanh(self.linear1(x_in))
        # alpha = F.relu(self.linear1(x_in))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * (x ** 2), dim=2) - mean ** 2
        std = torch.sqrt(residuals.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)


class ECAPA_TDNN(nn.Module):
    def __init__(self, feat_dim=80, channels=512, emb_dim=192, global_context_att=False,
                 feat_type='fbank', sr=16000, update_extract=False, pretrain_ckpt=None):
        super().__init__()

        self.feat_type = feat_type
        self.update_extract = update_extract
        self.sr = sr

        if feat_type == "fbank" or feat_type == "mfcc":
            self.update_extract = False

        win_len = int(sr * 0.025)
        hop_len = int(sr * 0.01)

        if feat_type == 'fbank':
            self.feature_extract = trans.MelSpectrogram(sample_rate=sr, n_fft=512, win_length=win_len,
                                                        hop_length=hop_len, f_min=0.0, f_max=sr // 2,
                                                        pad=0, n_mels=feat_dim)
        elif feat_type == 'mfcc':
            melkwargs = {
                'n_fft': 512,
                'win_length': win_len,
                'hop_length': hop_len,
                'f_min': 0.0,
                'f_max': sr // 2,
                'pad': 0
            }
            self.feature_extract = trans.MFCC(sample_rate=sr, n_mfcc=feat_dim, log_mels=False,
                                              melkwargs=melkwargs)
        else:
            #if config_path is None:
            #    self.feature_extract = torch.hub.load('s3prl/s3prl', feat_type)
            #else:
            #    self.feature_extract = UpstreamExpert(config_path)
            self.feature_extract = UpstreamExpert(pretrain_ckpt)
            if len(self.feature_extract.model.encoder.layers) == 24 and hasattr(self.feature_extract.model.encoder.layers[23].self_attn, "fp32_attention"):
                self.feature_extract.model.encoder.layers[23].self_attn.fp32_attention = False
            if len(self.feature_extract.model.encoder.layers) == 24 and hasattr(self.feature_extract.model.encoder.layers[11].self_attn, "fp32_attention"):
                self.feature_extract.model.encoder.layers[11].self_attn.fp32_attention = False
            """
            if feat_type=="wavlm_base_plus":
                checkpoint = torch.load(pretrain_ckpt, map_location=torch.device("cpu"))
                cfg = WavLMConfig(checkpoint['cfg'])
                self.feature_extract = WavLM(cfg) # instance WavLM model
                self.wavlm_encoder_layers = checkpoint['cfg']["encoder_layers"]
                print(f"self.feature_extract: {self.feature_extract}")
            elif feat_type=="wavlm_large":
                checkpoint = torch.load(pretrain_ckpt, map_location=torch.device("cpu"))
                cfg = WavLMConfig(checkpoint['cfg'])
                self.feature_extract = WavLM(cfg) # instance WavLM model
                self.wavlm_encoder_layers = checkpoint['cfg']["encoder_layers"]
                print(f"self.feature_extract: {self.feature_extract}")
            if len(self.feature_extract.encoder.layers) == 24 and hasattr(self.feature_extract.encoder.layers[23].self_attn, "fp32_attention"):
                self.feature_extract.encoder.layers[23].self_attn.fp32_attention = False
            if len(self.feature_extract.encoder.layers) == 24 and hasattr(self.feature_extract.encoder.layers[11].self_attn, "fp32_attention"):
                self.feature_extract.encoder.layers[11].self_attn.fp32_attention = False
            """
            self.feat_num = self.get_feat_num()
            self.feature_weight = nn.Parameter(torch.zeros(self.feat_num))

        if feat_type != 'fbank' and feat_type != 'mfcc':
            freeze_list = ['final_proj', 'label_embs_concat', 'mask_emb', 'project_q', 'quantizer']
            for name, param in self.feature_extract.named_parameters():
                for freeze_val in freeze_list:
                    if freeze_val in name:
                        param.requires_grad = False
                        break

        if not self.update_extract:
            for param in self.feature_extract.parameters():
                param.requires_grad = False

        self.instance_norm = nn.InstanceNorm1d(feat_dim)
        # self.channels = [channels] * 4 + [channels * 3]
        self.channels = [channels] * 4 + [1536]

        self.layer1 = Conv1dReluBn(feat_dim, self.channels[0], kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(self.channels[0], self.channels[1], kernel_size=3, stride=1, padding=2, dilation=2, scale=8, se_bottleneck_dim=128)
        self.layer3 = SE_Res2Block(self.channels[1], self.channels[2], kernel_size=3, stride=1, padding=3, dilation=3, scale=8, se_bottleneck_dim=128)
        self.layer4 = SE_Res2Block(self.channels[2], self.channels[3], kernel_size=3, stride=1, padding=4, dilation=4, scale=8, se_bottleneck_dim=128)

        # self.conv = nn.Conv1d(self.channels[-1], self.channels[-1], kernel_size=1)
        cat_channels = channels * 3
        self.conv = nn.Conv1d(cat_channels, self.channels[-1], kernel_size=1)
        self.pooling = AttentiveStatsPool(self.channels[-1], attention_channels=128, global_context_att=global_context_att)
        self.bn = nn.BatchNorm1d(self.channels[-1] * 2)
        self.linear = nn.Linear(self.channels[-1] * 2, emb_dim)


    def get_feat_num(self):
        self.feature_extract.eval()
        wav = [torch.randn(self.sr).to(next(self.feature_extract.parameters()).device)]
        #wav = torch.randn(1,self.sr).to(next(self.feature_extract.parameters()).device)
        with torch.no_grad():
            features = self.feature_extract(wav) # forward pass of  model
            #print(f"features: {features}")
        select_feature = features["default"][1] # res["layer_results"], if model=wavlm_large, it will have 24+1 layer feature
        #print(f"select_feature len: {len(select_feature)}, select_feature[0][0] shape: {select_feature[0][0].shape}") #len:12+1, (T,B,F)
        if isinstance(select_feature, (list, tuple)):
            return len(select_feature)
        else:
            return 1

    def get_feat(self, x):
        # reference from https://github.com/microsoft/unilm/tree/master/wavlm
        if self.update_extract:
            features = self.feature_extract([sample for sample in x])
            layer_results = features["default"][1] # x is list,it is also res["layer_results"]
            x = [x.transpose(0, 1) for x, _ in layer_results] # #[(B,T,F),(B,T,F)....]
        else:
            with torch.no_grad():
                features = self.feature_extract([sample for sample in x])
                layer_results = features["default"][1]# x is list,it is also res["layer_results"],
                x = [x.transpose(0, 1) for x, _ in layer_results] # #[(B,T,F),(B,T,F)....]
        if self.feat_type != "fbank" and self.feat_type != "mfcc":
            if isinstance(x, (list, tuple)):
                x = torch.stack(x, dim=0)#(num_layers,B,T,F)
            norm_weights = F.softmax(self.feature_weight, dim=-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            x = (norm_weights * x).sum(dim=0)
            x = torch.transpose(x, 1, 2) + 1e-6 # (B,F,T)

        x = self.instance_norm(x) # (B,F,T)
        return x

    def forward(self, x,get_time_out=False):
        x = self.get_feat(x)
        #print(f"wavlm output shape: {x.shape}") #(B,F,T) i.e.(2,768,99)
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        out = torch.cat([out2, out3, out4], dim=1)
        out = F.relu(self.conv(out))
        if get_time_out: ## as for speech encoder of ts_vad, get frame-level information
            return out # (B,F,T)
        out = self.bn(self.pooling(out))
        out = self.linear(out)

        return out #(B,F)


def ECAPA_TDNN_SMALL(feat_dim, emb_dim=256, feat_type='fbank', sr=16000, update_extract=False, pretrain_ckpt=None):
    return ECAPA_TDNN(feat_dim=feat_dim, channels=512, emb_dim=emb_dim,
                      feat_type=feat_type, sr=sr,  update_extract=update_extract, pretrain_ckpt=pretrain_ckpt)

if __name__ == '__main__':
    x = torch.zeros(2, 32000)
    model = ECAPA_TDNN_SMALL(feat_dim=768, emb_dim=256, feat_type='wavlm_base_plus',
                              update_extract=False,pretrain_ckpt="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Base+.pt")
    # 1024 is from encoder_dim of wavlm_large pretrain model
    #model = ECAPA_TDNN_SMALL(feat_dim=1024, emb_dim=256, feat_type='wavlm_large',
    #                          update_extract=False,pretrain_ckpt="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Large.pt")
    out = model(x,get_time_out=True)
    print(str(model))
    print(out.shape)#(B,1536,99)
