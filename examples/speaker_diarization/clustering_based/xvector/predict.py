#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Authors: Lukas Burget, Federico Landini, Jan Profant
# @Emails: burget@fit.vutbr.cz, landini@fit.vutbr.cz, jan.profant@phonexia.com

import argparse
import logging
import os
import time

import kaldi_io
import numpy as np
import onnxruntime
import soundfile as sf
import torch.backends

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

#import features
#from diarizer.models.resnet import *

torch.backends.cudnn.enabled = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()
        if self.name:
            logger.info(f"Start: {self.name}: ")

    def __exit__(self, type, value, traceback):
        if self.name:
            logger.info(
                f"End:   {self.name}: Elapsed: {time.time() - self.tstart} seconds"
            )
        else:
            logger.info(f"End:   {self.name}: ")


def initialize_gpus(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


def load_utt(ark, utt, position):
    with open(ark, "rb") as f:
        f.seek(position - len(utt) - 1)
        ark_key = kaldi_io.read_key(f)
        assert ark_key == utt, f"Keys does not match: `{ark_key}` and `{utt}`."
        mat = kaldi_io.read_mat(f)
        return mat


def write_txt_vectors(path, data_dict):
    """Write vectors file in text format.

    Args:
        path (str): path to txt file
        data_dict: (Dict[np.array]): name to array mapping
    """
    with open(path, "w") as f:
        for name in sorted(data_dict):
            f.write(
                f'{name}  [ {" ".join(str(x) for x in data_dict[name])} ]{os.linesep}'
            )


def get_embedding(fea, model, label_name=None, input_name=None, backend="pytorch"):
    if backend == "pytorch":
        data = torch.from_numpy(fea).to(device)
        data = data[None, :, :]
        data = torch.transpose(data, 1, 2)
        spk_embeds = model(data)
        return spk_embeds.data.cpu().numpy()[0]
    elif backend == "onnx":
        return model.run(
            [label_name],
            {input_name: fea.astype(np.float32).transpose()[np.newaxis, :, :]},
        )[0].squeeze()

def framing(a, window, shift=1):
    shape = ((a.shape[0] - window) // shift + 1, window) + a.shape[1:]
    strides = (a.strides[0]*shift, a.strides[0]) + a.strides[1:]
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


# Mel and inverse Mel scale warping functions
def mel_inv(x):
    return (np.exp(x/1127.) - 1.) * 700.


def mel(x):
    return 1127. * np.log(1. + x/700.)


def preemphasis(x, coef=0.97):
    return x - np.c_[x[..., :1], x[..., :-1]] * coef


def mel_fbank_mx(winlen_nfft, fs, NUMCHANS=20, LOFREQ=0.0, HIFREQ=None, warp_fn=mel, inv_warp_fn=mel_inv, htk_bug=True):
    """Returns mel filterbank as an array (NFFT/2+1 x NUMCHANS)
    winlen_nfft - Typically the window length as used in mfcc_htk() call. It is
                  used to determine number of samples for FFT computation (NFFT).
                  If positive, the value (window lenght) is rounded up to the
                  next higher power of two to obtain HTK-compatible NFFT.
                  If negative, NFFT is set to -winlen_nfft. In such case, the
                  parameter nfft in mfcc_htk() call should be set likewise.
    fs          - sampling frequency (Hz, i.e. 1e7/SOURCERATE)
    NUMCHANS    - number of filter bank bands
    LOFREQ      - frequency (Hz) where the first filter starts
    HIFREQ      - frequency (Hz) where the last filter ends (default fs/2)
    warp_fn     - function for frequency warping and its inverse
    inv_warp_fn - inverse function to warp_fn
    """
    HIFREQ = 0.5 * fs if not HIFREQ else HIFREQ
    nfft = 2**int(np.ceil(np.log2(winlen_nfft))) if winlen_nfft > 0 else -int(winlen_nfft)

    fbin_mel = warp_fn(np.arange(nfft / 2 + 1, dtype=float) * fs / nfft)
    cbin_mel = np.linspace(warp_fn(LOFREQ), warp_fn(HIFREQ), NUMCHANS + 2)
    cind = np.floor(inv_warp_fn(cbin_mel) / fs * nfft).astype(int) + 1
    mfb = np.zeros((len(fbin_mel), NUMCHANS))
    for i in range(NUMCHANS):
        mfb[cind[i]:cind[i+1], i] = (cbin_mel[i] - fbin_mel[cind[i]:cind[i+1]]) / (cbin_mel[i] - cbin_mel[i+1])
        mfb[cind[i+1]:cind[i+2], i] = (cbin_mel[i+2] - fbin_mel[cind[i+1]:cind[i+2]]) / \
                                      (cbin_mel[i+2] - cbin_mel[i+1])
    if LOFREQ > 0.0 and float(LOFREQ) / fs * nfft + 0.5 > cind[0] and htk_bug:
        mfb[cind[0], :] = 0.0  # Just to be HTK compatible
    return mfb


def fbank_htk(x, window, noverlap, fbank_mx, nfft=None, _E=None,
              USEPOWER=False, RAWENERGY=True, PREEMCOEF=0.97, ZMEANSOURCE=False,
              ENORMALISE=True, ESCALE=0.1, SILFLOOR=50.0, USEHAMMING=True):
    """Mel log Mel-filter bank channel outputs
    Returns NUMCHANS-by-M matrix of log Mel-filter bank outputs extracted from
    signal x, where M is the number of extracted frames, which can be computed
    as floor((length(x)-noverlap)/(window-noverlap)). Remaining parameters
    have the following meaning:
    x         - input signal
    window    - frame window length (in samples, i.e. WINDOWSIZE/SOURCERATE)
                or vector of window weights override default windowing function
                (see option USEHAMMING)
    noverlap  - overlapping between frames (in samples, i.e window-TARGETRATE/SOURCERATE)
    fbank_mx  - array with (Mel) filter bank (as returned by function mel_fbank_mx()).
                Note that this must be compatible with the parameter 'nfft'.
    nfft      - number of samples for FFT computation. By default, it is set in the
                HTK-compatible way to the window length rounded up to the next higher
                power of two.
    _E        - include energy as the "first" or the "last" coefficient of each
                feature vector. The possible values are: "first", "last", None.

    Remaining options have exactly the same meaning as in HTK.

    See also:
      mel_fbank_mx:
          to obtain the matrix for the parameter fbank_mx
      add_deriv:
          for adding delta, double delta, ... coefficients
      add_dither:
          for adding dithering in HTK-like fashion
    """
    if type(USEPOWER) == bool:
        USEPOWER += 1
    if np.isscalar(window):
        window = np.hamming(window) if USEHAMMING else np.ones(window)
    if nfft is None:
        nfft = 2**int(np.ceil(np.log2(window.size)))
    x = framing(x.astype("float"), window.size, window.size-noverlap).copy()
    if ZMEANSOURCE:
        x -= x.mean(axis=1)[:, np.newaxis]
    if _E is not None and RAWENERGY:
        energy = np.log((x**2).sum(axis=1))
    if PREEMCOEF is not None:
        x = preemphasis(x, PREEMCOEF)
    x *= window
    if _E is not None and not RAWENERGY:
        energy = np.log((x**2).sum(axis=1))
    x = np.fft.rfft(x, nfft)
    x = x.real**2 + x.imag**2
    if USEPOWER != 2:
        x **= 0.5 * USEPOWER
    x = np.log(np.maximum(1.0, np.dot(x, fbank_mx)))
    if _E is not None and ENORMALISE:
        energy = (energy - energy.max()) * ESCALE + 1.0
        min_val = -np.log(10**(SILFLOOR/10.)) * ESCALE + 1.0
        energy[energy < min_val] = min_val

    return np.hstack(([energy[:, np.newaxis]] if _E == "first" else []) + [x] +
                     ([energy[:, np.newaxis]] if (_E in ["last", True]) else []))


def povey_window(winlen):
    return np.power(0.5 - 0.5*np.cos(np.linspace(0, 2*np.pi, winlen)), 0.85)


def add_dither(x, level=8):
    return x + level * (np.random.rand(*x.shape)*2 - 1)


def cmvn_floating_kaldi(x, LC, RC, norm_vars=True):
    """Mean and variance normalization over a floating window.
    x is the feature matrix (nframes x dim)
    LC, RC are the number of frames to the left and right defining the floating
    window around the current frame. This function uses Kaldi-like treatment of
    the initial and final frames: Floating windows stay of the same size and
    for the initial and final frames are not centered around the current frame
    but shifted to fit in at the beginning or the end of the feature segment.
    Global normalization is used if nframes is less than LC+RC+1.
    """
    N, dim = x.shape
    win_len = min(len(x), LC+RC+1)
    win_start = np.maximum(np.minimum(np.arange(-LC, N-LC), N-win_len), 0)
    f = np.r_[np.zeros((1, dim)), np.cumsum(x, 0)]
    x = x - (f[win_start+win_len] - f[win_start]) / win_len
    if norm_vars:
        f = np.r_[np.zeros((1, dim)), np.cumsum(x**2, 0)]
        x /= np.sqrt((f[win_start+win_len] - f[win_start]) / win_len)
    return x


'''ResNet in PyTorch.


Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, reduction=16):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        #self.se = SELayer(planes, reduction)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        #out = self.se(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, reduction=16):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        #self.se = SELayer(planes * 4, reduction)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        #out = self.se(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, m_channels=32, feat_dim=40, embed_dim=128, squeeze_excitation=False):
        super(ResNet, self).__init__()
        self.in_planes = m_channels
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.squeeze_excitation = squeeze_excitation
        if block is BasicBlock:
            self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(m_channels)
            self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, m_channels*2, num_blocks[1], stride=2)
            current_freq_dim = int((feat_dim - 1) / 2) + 1
            self.layer3 = self._make_layer(block, m_channels*4, num_blocks[2], stride=2)
            current_freq_dim = int((current_freq_dim - 1) / 2) + 1
            self.layer4 = self._make_layer(block, m_channels*8, num_blocks[3], stride=2)
            current_freq_dim = int((current_freq_dim - 1) / 2) + 1
            self.embedding = nn.Linear(m_channels * 8 * 2 * current_freq_dim, embed_dim)
        elif block is Bottleneck:
            self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(m_channels)
            self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, m_channels*2, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, m_channels*4, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, m_channels*8, num_blocks[3], stride=2)
            self.embedding = nn.Linear(int(feat_dim/8) * m_channels * 16 * block.expansion, embed_dim)
        else:
            raise ValueError(f'Unexpected class {type(block)}.')
           

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze_(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        pooling_mean = torch.mean(out, dim=-1)
        meansq = torch.mean(out * out, dim=-1)
        pooling_std = torch.sqrt(meansq - pooling_mean ** 2 + 1e-10)
        out = torch.cat((torch.flatten(pooling_mean, start_dim=1),
                         torch.flatten(pooling_std, start_dim=1)), 1)

        embedding = self.embedding(out)
        return embedding 


def ResNet101(feat_dim, embed_dim, squeeze_excitation=False):
    return ResNet(Bottleneck, [3, 4, 23, 3], feat_dim=feat_dim, embed_dim=embed_dim, squeeze_excitation=squeeze_excitation)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpus",
        type=bool,
        default=False,
        help="use gpus (passed to CUDA_VISIBLE_DEVICES)",
    )
    parser.add_argument(
        "--model", required=False, type=str, default=None, help="name of the model"
    )
    parser.add_argument(
        "--weights",
        required=True,
        type=str,
        default=None,
        help="path to pretrained model weights",
    )
    parser.add_argument(
        "--model-file",
        required=False,
        type=str,
        default=None,
        help="path to model file",
    )
    parser.add_argument(
        "--ndim",
        required=False,
        type=int,
        default=64,
        help="dimensionality of features",
    )
    parser.add_argument(
        "--embed-dim",
        required=False,
        type=int,
        default=256,
        help="dimensionality of the emb",
    )
    parser.add_argument(
        "--seg-len", required=False, type=int, default=144, help="segment length"
    )
    parser.add_argument(
        "--seg-jump", required=False, type=int, default=24, help="segment jump"
    )
    parser.add_argument(
        "--in-file-list", required=True, type=str, help="input list of files"
    )
    parser.add_argument(
        "--in-lab-dir", required=True, type=str, help="input directory with VAD labels"
    )
    parser.add_argument(
        "--in-wav-dir", required=True, type=str, help="input directory with wavs"
    )
    parser.add_argument(
        "--out-ark-fn", required=True, type=str, help="output embedding file"
    )
    parser.add_argument(
        "--out-seg-fn", required=True, type=str, help="output segments file"
    )
    parser.add_argument(
        "--backend",
        required=False,
        default="pytorch",
        choices=["pytorch", "onnx"],
        help="backend that is used for x-vector extraction",
    )

    args = parser.parse_args()

    seg_len = args.seg_len
    seg_jump = args.seg_jump

    device = ""
    if args.gpus == True:
        logger.info(f"Using GPU for x-vector extraction")

        # gpu configuration
        # initialize_gpus(args) # already done in queue-freegpu.pl
        device = torch.device(device="cuda")
        _ = torch.ones(1).to(device)  # GPU reserve variable
    else:
        device = torch.device(device="cpu")

    model, label_name, input_name = "", None, None

    if args.backend == "pytorch":
        if args.model_file is not None:
            model = torch.load(args.model_file)
            model = model.to(device)
        elif args.model is not None and args.weights is not None:
            model = eval(args.model)(feat_dim=args.ndim, embed_dim=args.embed_dim)
            model = model.to(device)
            checkpoint = torch.load(args.weights, map_location=device)
            model.load_state_dict(checkpoint["state_dict"], strict=False)
            model.eval()
    elif args.backend == "onnx":
        model = onnxruntime.InferenceSession(args.weights)
        input_name = model.get_inputs()[0].name
        label_name = model.get_outputs()[0].name

    else:
        raise ValueError(
            "Wrong combination of --model/--weights/--model_file "
            "parameters provided (or not provided at all)"
        )

    file_names = np.atleast_1d(np.loadtxt(args.in_file_list, dtype=object))

    with torch.no_grad():
        with open(args.out_seg_fn, "w") as seg_file:
            with open(args.out_ark_fn, "wb") as ark_file:
                for fn in file_names:
                    with Timer(f"Processing file {fn}"):
                        signal, samplerate = sf.read(
                            f"{os.path.join(args.in_wav_dir, fn)}.wav"
                        )
                        labs = np.atleast_2d(
                            (
                                np.loadtxt(
                                    f"{os.path.join(args.in_lab_dir, fn)}.lab",
                                    usecols=(0, 1),
                                )
                                * samplerate
                            ).astype(int)
                        )
                        if samplerate == 8000:
                            noverlap = 120
                            winlen = 200
                            window = povey_window(winlen)
                            fbank_mx = mel_fbank_mx(
                                winlen,
                                samplerate,
                                NUMCHANS=64,
                                LOFREQ=20.0,
                                HIFREQ=3700,
                                htk_bug=False,
                            )
                        elif samplerate == 16000:
                            noverlap = 240
                            winlen = 400
                            window = povey_window(winlen)
                            fbank_mx = mel_fbank_mx(
                                winlen,
                                samplerate,
                                NUMCHANS=64,
                                LOFREQ=20.0,
                                HIFREQ=7600,
                                htk_bug=False,
                            )
                        else:
                            raise ValueError(
                                f"Only 8kHz and 16kHz are supported. Got {samplerate} instead."
                            )

                        LC = 150
                        RC = 149

                        np.random.seed(3)  # for reproducibility
                        signal = add_dither((signal * 2 ** 15).astype(int))

                        for segnum in range(len(labs)):
                            seg = signal[labs[segnum, 0] : labs[segnum, 1]]
                            if (
                                seg.shape[0] > 0.01 * samplerate
                            ):  # process segment only if longer than 0.01s
                                # Mirror noverlap//2 initial and final samples
                                seg = np.r_[
                                    seg[noverlap // 2 - 1 :: -1],
                                    seg,
                                    seg[-1 : -winlen // 2 - 1 : -1],
                                ]
                                fea = fbank_htk(
                                    seg,
                                    window,
                                    noverlap,
                                    fbank_mx,
                                    USEPOWER=True,
                                    ZMEANSOURCE=True,
                                )
                                fea = cmvn_floating_kaldi(
                                    fea, LC, RC, norm_vars=False
                                ).astype(np.float32)

                                slen = len(fea)
                                start = -seg_jump

                                for start in range(0, slen - seg_len, seg_jump):
                                    data = fea[start : start + seg_len]
                                    xvector = get_embedding(
                                        data,
                                        model,
                                        label_name=label_name,
                                        input_name=input_name,
                                        backend=args.backend,
                                    )

                                    key = f"{fn}_{segnum:04}-{start:08}-{(start + seg_len):08}"
                                    if np.isnan(xvector).any():
                                        logger.warning(
                                            f"NaN found, not processing: {key}{os.linesep}"
                                        )
                                    else:
                                        seg_start = round(
                                            labs[segnum, 0] / float(samplerate)
                                            + start / 100.0,
                                            3,
                                        )
                                        seg_end = round(
                                            labs[segnum, 0] / float(samplerate)
                                            + start / 100.0
                                            + seg_len / 100.0,
                                            3,
                                        )
                                        seg_file.write(
                                            f"{key} {fn} {seg_start} {seg_end}{os.linesep}"
                                        )
                                        kaldi_io.write_vec_flt(
                                            ark_file, xvector, key=key
                                        )

                                if slen - start - seg_jump >= 10:
                                    data = fea[start + seg_jump : slen]
                                    xvector = get_embedding(
                                        data,
                                        model,
                                        label_name=label_name,
                                        input_name=input_name,
                                        backend=args.backend,
                                    )

                                    key = f"{fn}_{segnum:04}-{(start + seg_jump):08}-{slen:08}"

                                    if np.isnan(xvector).any():
                                        logger.warning(
                                            f"NaN found, not processing: {key}{os.linesep}"
                                        )
                                    else:
                                        seg_start = round(
                                            labs[segnum, 0] / float(samplerate)
                                            + (start + seg_jump) / 100.0,
                                            3,
                                        )
                                        seg_end = round(
                                            labs[segnum, 1] / float(samplerate), 3
                                        )
                                        seg_file.write(
                                            f"{key} {fn} {seg_start} {seg_end}{os.linesep}"
                                        )
                                        kaldi_io.write_vec_flt(
                                            ark_file, xvector, key=key
                                        )
