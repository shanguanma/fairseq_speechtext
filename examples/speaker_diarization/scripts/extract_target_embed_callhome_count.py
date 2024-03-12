import glob, tqdm, os, soundfile, copy, argparse
from collections import defaultdict
from scipy import signal

import torch
import numpy

import math, torchaudio
import torch.nn as nn
import torch.nn.functional as F


class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            # nn.BatchNorm1d(bottleneck), # I remove this layer
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.se(input)
        return input * x


class Bottle2neck(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8):
        super(Bottle2neck, self).__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        convs = []
        bns = []
        num_pad = math.floor(kernel_size / 2) * dilation
        for i in range(self.nums):
            convs.append(
                nn.Conv1d(
                    width,
                    width,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=num_pad,
                )
            )
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)

        out = self.se(out)
        out += residual
        return out


class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            "flipped_filter",
            torch.FloatTensor([-self.coef, 1.0]).unsqueeze(0).unsqueeze(0),
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), "reflect")
        return F.conv1d(input, self.flipped_filter).squeeze(1)


class FbankAug(nn.Module):

    def __init__(self, freq_mask_width=(0, 8), time_mask_width=(0, 10)):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        super().__init__()

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(
            width_range[0], width_range[1], (batch, 1), device=x.device
        ).unsqueeze(2)
        mask_pos = torch.randint(
            0, max(1, D - mask_len.max()), (batch, 1), device=x.device
        ).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)

        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x


class ECAPA_TDNN(nn.Module):

    def __init__(self, C):

        super(ECAPA_TDNN, self).__init__()

        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_fft=512,
                win_length=400,
                hop_length=160,
                f_min=20,
                f_max=7600,
                window_fn=torch.hamming_window,
                n_mels=80,
            ),
        )

        self.specaug = FbankAug()  # Spec augmentation

        self.conv1 = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3 * C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),  # I add this layer
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)

    def forward(self, x, aug=False):
        with torch.no_grad():
            x = self.torchfbank(x) + 1e-6
            x = x.log()
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if aug == True:
                x = self.specaug(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)

        x = self.layer4(torch.cat((x1, x2, x3), dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat(
            (
                x,
                torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
                torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(
                    1, 1, t
                ),
            ),
            dim=1,
        )

        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-4))

        x = torch.cat((mu, sg), 1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        return x


def init_speaker_encoder(source):
    speaker_encoder = ECAPA_TDNN(C=1024).cuda()
    speaker_encoder.eval()
    loadedState = torch.load(source, map_location="cuda")
    selfState = speaker_encoder.state_dict()
    for name, param in loadedState.items():
        name = name.replace("speaker_encoder.", "")
        if name in selfState:
            selfState[name].copy_(param)
        else:
            print("Not exist ", name)
    for param in speaker_encoder.parameters():
        param.requires_grad = False
    return speaker_encoder


def extract_embeddings(batch, model):
    batch = torch.stack(batch)
    with torch.no_grad():
        embeddings = model.forward(batch.cuda())
    return embeddings


def remove_overlap(aa, bb):
    # Sort the intervals in both lists based on their start time
    a = aa.copy()
    b = bb.copy()
    a.sort()
    b.sort()

    # Initialize the new list of intervals
    result = []

    # Initialize variables to keep track of the current interval in list a and the remaining intervals in list b
    i = 0
    j = 0

    # Iterate through the intervals in list a
    while i < len(a):
        # If there are no more intervals in list b or the current interval in list a does not overlap with the current interval in list b, add it to the result and move on to the next interval in list a
        if j == len(b) or a[i][1] <= b[j][0]:
            result.append(a[i])
            i += 1
        # If the current interval in list a completely overlaps with the current interval in list b, skip it and move on to the next interval in list a
        elif a[i][0] >= b[j][0] and a[i][1] <= b[j][1]:
            i += 1
        # If the current interval in list a partially overlaps with the current interval in list b, add the non-overlapping part to the result and move on to the next interval in list a
        elif a[i][0] < b[j][1] and a[i][1] > b[j][0]:
            if a[i][0] < b[j][0]:
                result.append([a[i][0], b[j][0]])
            a[i][0] = b[j][1]
        # If the current interval in list a starts after the current interval in list b, move on to the next interval in list b
        elif a[i][0] >= b[j][1]:
            j += 1

    # Return the new list of intervals
    return result


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--rttm_path", required=True, help="the path for the rttm_files"
    )
    parser.add_argument(
        "--orig_audio_path", required=True, help="the path for the orig audio"
    )
    parser.add_argument(
        "--target_audio_path", required=True, help="the part for the output audio"
    )
    parser.add_argument("--sample_rate", default=8000, type=int, help="sample rate")
    parser.add_argument("--source", help="the part for the speaker encoder")
    parser.add_argument(
        "--length_embedding",
        type=float,
        default=6,
        help="length of embeddings, seconds",
    )
    parser.add_argument(
        "--step_embedding", type=float, default=1, help="step of embeddings, seconds"
    )
    parser.add_argument(
        "--batch_size", type=int, default=96, help="step of embeddings, seconds"
    )
    args = parser.parse_args()

    return args


def main():
    args = get_args()
    lines = open(args.rttm_path).read().splitlines()
    room_set = set()
    for line in lines:
        data = line.split()
        room_set.add(data[1])

    # model = init_speaker_encoder(args.source)
    count = 0
    for room_id in tqdm.tqdm(room_set):
        print(os.path.join(args.target_audio_path, room_id))
        if os.path.isdir(os.path.join(args.target_audio_path, room_id)):
            print("exist")
            break
        print(count)
        count += 1
        if count == 2865:
            break
        # intervals = defaultdict(list)
        # new_intervals = defaultdict(list)
        # for line in (lines):
        #     data = line.split()
        #     if data[1] == room_id:
        #         stime = float(data[3])
        #         etime = float(data[3]) + float(data[4])
        #         spkr = data[-3]
        #         intervals[spkr].append([stime, etime])

        # # Remove the overlapped speeech
        # for key in intervals:
        #     new_interval = intervals[key]
        #     for o_key in intervals:
        #         if o_key != key:
        #             new_interval = remove_overlap(copy.deepcopy(new_interval), copy.deepcopy(intervals[o_key]))
        #     new_intervals[key] = new_interval

        # wav_file = glob.glob(os.path.join(args.orig_audio_path, room_id) + '.wav')[0]
        # orig_audio, fs = soundfile.read(wav_file)

        # if len(orig_audio.shape) > 1:
        #     orig_audio = orig_audio[:,0]

        # # # Cut and save the clean speech part
        # id_full = wav_file.split('/')[-1][:-4]
        # for key in new_intervals:
        #     output_dir = os.path.join(args.target_audio_path, id_full)
        #     os.makedirs(output_dir, exist_ok = True)
        #     output_wav = os.path.join(output_dir, str(key) + '.pt')
        #     new_audio = []
        #     for interval in new_intervals[key]:
        #         s, e = interval
        #         s *= args.sample_rate
        #         e *= args.sample_rate
        #         new_audio.extend(orig_audio[int(s):int(e)])

        #     if args.sample_rate != 16000:
        #         new_audio_re = signal.resample(new_audio, int((16000 * len(new_audio)) / args.sample_rate)) # upsample to 16000
        #     else:
        #         new_audio_re = new_audio

        #     batch = []
        #     embeddings = []
        #     wav_length = new_audio_re.shape[0]
        #     if wav_length > int(args.length_embedding * 16000):
        #         for start in range(0, wav_length - int(args.length_embedding * 16000), int(args.step_embedding * 16000)):
        #             stop = start + int(args.length_embedding * 16000)
        #             target_speech = torch.FloatTensor(numpy.array(new_audio_re[start:stop]))
        #             batch.append(target_speech)
        #             if len(batch) == args.batch_size:
        #                 embeddings.extend(extract_embeddings(batch, model))
        #                 batch = []
        #     else:
        #         target_speech = torch.FloatTensor(numpy.array(new_audio_re))
        #         embeddings.extend(extract_embeddings([target_speech], model))

        #     if len(batch) != 0:
        #         embeddings.extend(extract_embeddings(batch, model))
        #     embeddings = torch.stack(embeddings)
        #     # os.makedirs(os.path.dirname(output_wav), exist_ok = True)
        #     torch.save(embeddings, output_wav)


if __name__ == "__main__":
    main()
