from collections import defaultdict
import logging
import os
from scipy import signal
import glob, json, random, wave
import re
from tqdm import tqdm

import numpy as np
import soundfile as sf

import torch
import librosa
import torchaudio.compliance.kaldi as kaldi
from fairseq.data.fairseq_dataset import FairseqDataset

logger = logging.getLogger(__name__)


class FBank(object):
    def __init__(
        self,
        n_mels,
        sample_rate,
        mean_nor: bool = False,
    ):
        self.n_mels = n_mels
        self.mean_nor = mean_nor

    def __call__(self, wav, dither=1.0):
        sr = 16000
        if len(wav.shape) == 1:
            wav = wav.unsqueeze(0)
        assert len(wav.shape) == 2 and wav.shape[0] == 1
        wav = wav * (1 << 15)
        feat = kaldi.fbank(
            wav,
            num_mel_bins=self.n_mels,
            sample_frequency=sr,
            dither=dither,
            window_type="hamming",
            use_energy=False,
        )
        # feat: [T, N]
        if self.mean_nor:
            feat = feat - feat.mean(0, keepdim=True)
        return feat


def _collate_data(
    frames,
    is_audio_input: bool = False,
    is_label_input: bool = False,
    is_embed_input: bool = False,
) -> torch.Tensor:
    if is_audio_input:
        max_len = max(frame.size(0) for frame in frames)
        out = frames[0].new_zeros((len(frames), max_len))
        for i, v in enumerate(frames):
            out[i, : v.size(0)] = v

    if is_label_input:
        max_len = max(frame.size(1) for frame in frames)
        out = frames[0].new_zeros((len(frames), frames[0].size(0), max_len))
        for i, v in enumerate(frames):
            out[i, :, : v.size(1)] = v

    if is_embed_input:
        if len(frames[0].size()) == 2:
            max_len = max(frame.size(0) for frame in frames)
            max_len2 = max(frame.size(1) for frame in frames)
            out = frames[0].new_zeros((len(frames), max_len, max_len2))
            for i, v in enumerate(frames):
                out[i, : v.size(0), : v.size(1)] = v
        elif len(frames[0].size()) == 3:
            max_len = max(frame.size(1) for frame in frames)
            out = frames[0].new_zeros(
                (len(frames), frames[0].size(0), max_len, frames[0].size(2))
            )
            for i, v in enumerate(frames):
                out[i, :, : v.size(1), :] = v

    return out


class TSVADDataset(FairseqDataset):
    def __init__(
        self,
        json_path: str,
        audio_path: str,
        ts_len: int,
        rs_len: int,
        is_train: bool,
        spk_path: str = None,
        aux_path: str = None,
        segment_shift: int = 6,
        musan_path: str = None,
        rir_path: str = None,
        noise_ratio: float = 0.5,
        zero_ratio: float = 0.5,
        max_num_speaker: int = 4,
        dataset_name: str = "alimeeting",
        sample_rate: int = 16000,
        embed_len: float = 1,
        embed_shift: float = 0.4,
        embed_input: bool = False,
        fbank_input: bool = False,
        fbank_input_aux: bool = False,
        inference: bool = False,
        label_rate: int = 25,
        random_channel: bool = False,
        support_mc: bool = False,
        random_mask_speaker_prob: float = 0.0,
        random_mask_speaker_step: int = 0,
    ):
        self.audio_path = audio_path
        self.spk_path = spk_path
        self.aux_path = aux_path
        self.ts_len = ts_len  # Number of second for target speech
        self.rs_len = rs_len  # Number of second for reference speech

        self.data_list = []
        self.label_dic = defaultdict(list)
        self.dataset_name = dataset_name
        self.sample_rate = sample_rate
        self.inference = inference
        self.random_mask_speaker_prob = random_mask_speaker_prob
        self.random_mask_speaker_step = random_mask_speaker_step

        lines = open(json_path).read().splitlines()
        filename_set = set()
        self.sizes = []
        self.spk2data = {}
        self.data2spk = {}
        self.id2fullid = {}
        if re.match("^SparseLibri(2|3|23)Mix$|^Libri(2|3|23)Mix$", self.dataset_name):
            self.filename2aux = {}
        # Load the data and labels
        for line in tqdm(lines):
            dict = json.loads(line)
            length = len(dict["labels"])  # Number of frames (1s = 25 frames)
            filename = dict["filename"]
            labels = dict["labels"]
            speaker_id = str(dict["speaker_key"])
            speaker_id_full = str(dict["speaker_id"])
            if "real_data" in dict and dict["real_data"] == "true":
                real_data = True
            else:
                real_data = False
            if filename not in self.id2fullid:
                self.id2fullid[filename] = {}
            self.id2fullid[filename][speaker_id] = speaker_id_full

            if speaker_id_full not in self.spk2data:
                self.spk2data[speaker_id_full] = []
            if self.dataset_name == "alimeeting" or self.dataset_name == "icmc":
                self.spk2data[speaker_id_full].append(filename + "/" + speaker_id)
                self.data2spk[filename + "/" + speaker_id] = speaker_id_full
            elif (
                self.dataset_name == "ami"
                or self.dataset_name == "callhome_sim"
                or self.dataset_name == "libri_css_sim"
            ):
                self.spk2data[speaker_id_full].append(filename + "/" + speaker_id_full)
                self.data2spk[filename + "/" + speaker_id_full] = speaker_id_full
            elif re.match(
                "^SparseLibri(2|3|23)Mix$|^Libri(2|3|23)Mix$", self.dataset_name
            ):
                self.spk2data[speaker_id_full].append(dict["aux"])
                self.data2spk[dict["aux"]] = speaker_id_full

                if filename not in self.filename2aux:
                    self.filename2aux[filename] = {}
                self.filename2aux[filename][int(speaker_id)] = dict["aux"]
            else:
                raise Exception(
                    f"The given dataset {self.dataset_name} is not supported."
                )
            full_id = filename + "_" + speaker_id
            self.label_dic[full_id] = labels
            if filename in filename_set:
                pass
            else:
                filename_set.add(filename)
                dis = label_rate * segment_shift
                chunk_size = label_rate * self.rs_len
                folder = self.audio_path + "/" + filename + "/*.wav"
                if real_data:
                    # logger.warn("")
                    # only for callhome
                    num_speaker = dict["num_spk"]
                elif re.match(
                    "^SparseLibri(2|3|23)Mix$|^Libri(2|3|23)Mix$", self.dataset_name
                ):
                    num_speaker = len(filename.split("_"))
                elif (
                    self.dataset_name == "callhome_sim"
                    or self.dataset_name == "icmc"
                    or self.dataset_name == "libri_css_sim"
                ):
                    num_speaker = dict["spk_num"]
                else:
                    audios = glob.glob(folder)
                    num_speaker = (
                        len(audios) - 1
                    )  # The total number of speakers, 2 or 3 or 4
                if inference:  # only for joint model
                    self.data_list.append(
                        [filename, num_speaker, 0, length - 1, real_data]
                    )
                    self.sizes.append(length)
                else:
                    for start in range(0, length, dis):
                        end = (
                            (start + chunk_size)
                            if start + chunk_size < length
                            else length
                        )
                        if is_train:
                            short_ratio = 3
                        else:
                            short_ratio = 0
                        if end - start > label_rate * short_ratio:
                            data_intro = [filename, num_speaker, start, end, real_data]
                            self.data_list.append(data_intro)
                            self.sizes.append(
                                (end - start) * self.sample_rate / label_rate
                            )

        self.musan_path = musan_path
        if musan_path is not None:
            self.noiselist = {}
            self.noisetypes = ["noise", "speech", "music"]
            self.noisesnr = {"noise": [0, 15], "speech": [13, 20], "music": [5, 15]}
            self.numnoise = {"noise": [1, 1], "speech": [3, 8], "music": [1, 1]}
            # augment_files   = glob.glob(os.path.join(musan_path,'*/*/*/*.wav'))
            augment_files = glob.glob(
                os.path.join(musan_path, "*/*/*.wav")
            )  # musan/*/*/*.wav
            for file in augment_files:
                if file.split("/")[-3] not in self.noiselist:
                    self.noiselist[file.split("/")[-3]] = []
                self.noiselist[file.split("/")[-3]].append(file)
        self.rir_path = rir_path
        if rir_path is not None:
            # self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'))
            self.rir_files = glob.glob(
                os.path.join(rir_path, "*/*.wav")
            )  # RIRS_NOISES/*/*.wav
        self.noise_ratio = noise_ratio
        self.zero_ratio = zero_ratio

        self.is_train = is_train
        self.max_num_speaker = max_num_speaker

        self.embed_len = int(self.sample_rate * embed_len)
        self.embed_shift = int(self.sample_rate * embed_shift)
        self.embed_input = embed_input
        self.label_rate = label_rate
        self.fbank_input = fbank_input
        self.fbank_input_aux = fbank_input_aux
        self.random_channel = random_channel
        self.support_mc = support_mc
        self.update_num = 0
        if fbank_input or fbank_input_aux:
            logger.info(f"model expect fbank as input , fbank_input should be {fbank_input} !!!")
            self.feature_extractor = FBank(
                80, sample_rate=self.sample_rate, mean_nor=True
            )
        else:
            self.feature_extractor = None

        logger.info(
            f"loaded sentence={len(self.sizes)}, "
            f"shortest sent={min(self.sizes)}, longest sent={max(self.sizes)}, "
            f"rs_len={rs_len}, segment_shift={segment_shift}, rir={rir_path is not None}, "
            f"musan={musan_path is not None}, noise_ratio={noise_ratio}"
        )

    def __getitem__(self, index):
        # T: number of frames (1 frame = 0.04s)
        # ref_speech : 16000 * (T / 25)
        # labels : 4, T
        # target_speech: 4, 16000 * (T / 25)
        file, num_speaker, start, stop, _ = self.data_list[index]
        speaker_ids = self.get_ids(num_speaker) # num_speaker means that it contains number of speaker in current mixture utterance.
                                               
        ref_speech, labels, new_speaker_ids, _ = self.load_rs(
            file, speaker_ids, start, stop
        )
        if self.embed_input:
            ref_speech = self.segment_rs(ref_speech)

        if self.fbank_input:
            ref_speech = [self.feature_extractor(rs) for rs in ref_speech]
            ref_speech = torch.stack(ref_speech)

        if self.spk_path is None:
            target_speech, _, _, _ = self.load_ts(file, new_speaker_ids)
        else:
            target_speech = self.load_ts_embed(file, new_speaker_ids)

        sample = {
            "id": index,
            "ref_speech": ref_speech,
            "target_speech": target_speech,
            "labels": labels,
            "file_path": file,
            "speaker_ids": np.array(speaker_ids),
            "start": np.array(start),
        }

        return sample

    def repeat_to_fill(self, x, window_fs):
        length = x.size(0)
        num = (window_fs + length - 1) // length

        return x.repeat(num)[:window_fs]

    def segment_rs(self, ref_speech):
        subsegs = []
        for begin in range(0, ref_speech.size(0), self.embed_shift):
            end = min(begin + self.embed_len, ref_speech.size(0))
            subsegs.append(self.repeat_to_fill(ref_speech[begin:end], self.embed_len))

        return torch.stack(subsegs, dim=0)

    def get_ids(self, num_speaker, add_ext=False):
        speaker_ids = []
        for i in range(self.max_num_speaker - (1 if add_ext else 0)):
            if i < num_speaker:
                speaker_ids.append(i + 1)
            else:
                speaker_ids.append(-1)

        if self.is_train:
            random.shuffle(speaker_ids)

        if add_ext:
            speaker_ids.append(-2)
        return speaker_ids

#    def load_rs(self, file, speaker_ids, start, stop):
#        audio_start = self.sample_rate // self.label_rate * start
#        audio_stop = self.sample_rate // self.label_rate * stop
#        if self.dataset_name == "alimeeting":
#            audio_path = os.path.join(self.audio_path, file + "/all.wav")  ## This audio_path is single channel mixer audio,
#                                                                           ## now it is used in alimeeting dataset,and is stored at target_audio directory.
#            ref_speech, rc = self.read_audio_with_resample(
#                audio_path,
#                start=audio_start,
#                length=(audio_stop - audio_start),
#                support_mc=self.support_mc,
#            )
#            if len(ref_speech.shape) == 1:
#                ref_speech = np.expand_dims(np.array(ref_speech), axis=0)       
#
    def load_rs(self, file, speaker_ids, start, stop):
        audio_start = self.sample_rate // self.label_rate * start
        audio_stop = self.sample_rate // self.label_rate * stop
        if (
            re.match("^SparseLibri(2|3|23)Mix$|^Libri(2|3|23)Mix$", self.dataset_name)
            or self.dataset_name == "callhome_sim"
            or self.dataset_name == "libri_css_sim"
        ):
            audio_path = os.path.join(self.audio_path, file + ".wav")
        elif self.dataset_name == "icmc":
            audio_path = os.path.join(self.audio_path, file + "/DX02C01.wav")
        else:
            audio_path = os.path.join(
                self.audio_path, file + "/all.wav"
            )  ## This audio_path is single channel mixer audio,
            ## now it is used in alimeeting dataset,and is stored at target_audio directory.

        if self.dataset_name == "icmc" and self.support_mc:
            ref_speech_list = []
            for i in range(4):
                ref_speech_list.append(
                    self.read_audio_with_resample(
                        os.path.join(self.audio_path, file + f"/DX0{i + 1}C01.wav"),
                        start=audio_start,
                        length=(audio_stop - audio_start),
                    )[0]
                )

            ref_speech = np.stack(ref_speech_list)
            rc = -1
        else:
            ref_speech, rc = self.read_audio_with_resample(
                audio_path,
                start=audio_start,
                length=(audio_stop - audio_start),
                support_mc=self.support_mc,
            )
            if len(ref_speech.shape) == 1:
                ref_speech = np.expand_dims(np.array(ref_speech), axis=0)

        frame_len = audio_stop - audio_start

        assert (
            frame_len - ref_speech.shape[1] <= 100
        ), f"frame_len {frame_len} ref_speech.shape[1] {ref_speech.shape[1]}"
        if frame_len - ref_speech.shape[1] > 0:
            new_ref_speech = np.zeros((ref_speech.shape[0], frame_len))
            new_ref_speech[:, : ref_speech.shape[1]] = ref_speech
            ref_speech = new_ref_speech

        if self.rir_path is not None or self.musan_path is not None:
            add_noise = np.random.choice(2, p=[1 - self.noise_ratio, self.noise_ratio])
            if add_noise == 1:
                if self.rir_path is not None and self.musan_path is not None:
                    noise_type = random.randint(0, 1)
                    if noise_type == 0:
                        ref_speech = self.add_rev(ref_speech, length=frame_len)
                    elif noise_type == 1:
                        ref_speech = self.choose_and_add_noise(
                            random.randint(0, 2), ref_speech, frame_len
                        )
                elif self.rir_path is not None:
                    ref_speech = self.add_rev(ref_speech, length=frame_len)
                elif self.musan_path is not None:
                    ref_speech = self.choose_and_add_noise(
                        random.randint(0, 2), ref_speech, frame_len
                    )

        ref_speech = torch.FloatTensor(np.array(ref_speech))

        labels = []
        new_speaker_ids = []
        residual_label = np.zeros(stop - start)
        for speaker_id in speaker_ids:
            if speaker_id == -1:
                labels.append(np.zeros(stop - start))  # Obatin the labels for silence
            elif speaker_id == -2:
                residual_label[residual_label > 1] = 1
                labels.append(residual_label)
                new_speaker_ids.append(-2)
                continue
            else:
                full_label_id = file + "_" + str(speaker_id)
                label = self.label_dic[full_label_id]
                labels.append(
                    label[start:stop]
                )  # Obatin the labels for the reference speech

            mask_prob = 0
            if self.random_mask_speaker_prob != 0:
                mask_prob = self.random_mask_speaker_prob * min(
                    self.update_num / self.random_mask_speaker_step, 1.0
                )
            if sum(labels[-1]) == 0 and self.is_train:
                new_speaker_ids.append(-1)
            elif (
                sum(new_speaker_ids) != -1 * len(new_speaker_ids)
                and np.random.choice(2, p=[1 - mask_prob, mask_prob])
                and self.is_train
            ):
                new_speaker_ids.append(-1)
                residual_label = residual_label + labels[-1]
                labels[-1] = np.zeros(stop - start)
            else:
                new_speaker_ids.append(speaker_id)

        labels = torch.from_numpy(np.array(labels)).float()  # 4, T
        return ref_speech, labels, new_speaker_ids, rc

    def load_ts_embed(self, file, speaker_ids):
        target_speeches = []
        exist_spk = []
        #print(f"file:{file}, speaker_ids: {speaker_ids}")
        for speaker_id in speaker_ids:
            if speaker_id != -1 and speaker_id != -2:
                audio_filename = speaker_id
                exist_spk.append(self.data2spk[f"{file}/{audio_filename}"])

        for speaker_id in speaker_ids:
            if speaker_id == -1: # Obatin the labels for silence
                if np.random.choice(2, p=[1 - self.zero_ratio, self.zero_ratio]) == 1 or not self.is_train:

                    # (TODO) maduo add speaker embedding dimension parameter to replace hard code.
                    feature = torch.zeros(256) # speaker embedding dimension of speaker model
                else:
                    random_spk = random.choice(list(self.spk2data))
                    while random_spk in exist_spk:
                        random_spk = random.choice(list(self.spk2data))
                    exist_spk.append(random_spk)
                    
                    path = os.path.join(self.spk_path,
                            f"{random.choice(self.spk2data[random_spk])}.pt",
                        )
                    feature = torch.load(path, map_location="cpu")
            elif speaker_id == -2: # # Obatin the labels for extral
                feature = torch.zeros(256) # speaker embedding dimension of speaker model
            else: # # Obatin the labels for speaker
                path = os.path.join(self.spk_path, file, str(audio_filename) + ".pt")
                feature = torch.load(path, map_location="cpu")
                #logger.info(f"path: {path}!")
                #logger.info(f"feature size: {feature.size()}!")

            if len(feature.size()) == 2:# (T,D)
                if self.is_train:
                    feature = feature[random.randint(0, feature.shape[0] - 1), :]
                else:
                    # feature = torch.mean(feature, dim = 0)
                    feature = torch.mean(feature, dim=0)
            target_speeches.append(feature)
        target_speeches = torch.stack(target_speeches)
        return target_speeches

    def load_alimeeting_ts_embed(self, file, speaker_ids):
        target_speeches = []
        exist_spk = []
        #print(f"file:{file}, speaker_ids: {speaker_ids}")
        for speaker_id in speaker_ids:
            if speaker_id != -1 and speaker_id != -2:
                audio_filename = speaker_id
                exist_spk.append(self.data2spk[f"{file}/{audio_filename}"])

        for speaker_id in speaker_ids:
            if speaker_id == -1: # Obatin the labels for silence
                if np.random.choice(2, p=[1 - self.zero_ratio, self.zero_ratio]) == 1 or not self.is_train:

                    # (TODO) maduo add speaker embedding dimension parameter to replace hard code.
                    feature = torch.zeros(256) # speaker embedding dimension of speaker model
                else:
                    random_spk = random.choice(list(self.spk2data))
                    while random_spk in exist_spk:
                        random_spk = random.choice(list(self.spk2data))
                    exist_spk.append(random_spk)

                    path = os.path.join(self.spk_path,
                            f"{random.choice(self.spk2data[random_spk])}.pt",
                        )
                    feature = torch.load(path, map_location="cpu")
            elif speaker_id == -2: # # Obatin the labels for extral
                feature = torch.zeros(256) # speaker embedding dimension of speaker model
            else: # # Obatin the labels for speaker
                path = os.path.join(self.spk_path, file, str(audio_filename) + ".pt")
                feature = torch.load(path, map_location="cpu")
                #logger.info(f"path: {path}!")
                #logger.info(f"feature size: {feature.size()}!")

            if len(feature.size()) == 2:
                if self.is_train:
                    feature = feature[random.randint(0, feature.shape[0] - 1), :]
                else:
                    # feature = torch.mean(feature, dim = 0)
                    feature = torch.mean(feature, dim = 0)

            target_speeches.append(feature)
        target_speeches = torch.stack(target_speeches)
        return target_speeches

        

    #    def load_ts_embed(self, file, speaker_ids):
    #        target_speeches = []
    #        exist_spk = []
    #        for speaker_id in speaker_ids:
    #            if speaker_id != -1 and speaker_id != -2:
    #                if re.match(
    #                    "^SparseLibri(2|3|23)Mix$|^Libri(2|3|23)Mix$", self.dataset_name
    #                ):
    #                    exist_spk.append(self.data2spk[self.filename2aux[file][speaker_id]])
    #                else:
    #                    if self.dataset_name == "alimeeting" or self.dataset_name == "icmc":
    #                        audio_filename = speaker_id
    #                    elif (
    #                        self.dataset_name == "ami"
    #                        or self.dataset_name == "callhome_sim"
    #                        or self.dataset_name == "libri_css_sim"
    #                    ):
    #                        audio_filename = self.id2fullid[file][str(speaker_id)]
    #                    exist_spk.append(self.data2spk[f"{file}/{audio_filename}"])
    #
    #        for speaker_id in speaker_ids:
    #            if speaker_id == -1: # Obatin the labels for silence
    #                if (
    #                    np.random.choice(2, p=[1 - self.zero_ratio, self.zero_ratio]) == 1
    #                    or not self.is_train
    #                ):
                          
    #                    feature = torch.zeros(192)
    #                else:
    #                    random_spk = random.choice(list(self.spk2data))
    #                    while random_spk in exist_spk:
    #                        random_spk = random.choice(list(self.spk2data))
    #                    exist_spk.append(random_spk)
    #                    if self.dataset_name == "icmc":
    #                        path = os.path.join(self.spk_path, f"{random_spk}.pt")
    #                    else:
    #                        path = os.path.join(
    #                            self.spk_path,
    #                            f"{random.choice(self.spk2data[random_spk])}.pt",
    #                        )
    #                    feature = torch.load(path, map_location="cpu")
    #            elif speaker_id == -2: # # Obatin the labels for extral
    #                feature = torch.zeros(192)
    #            else:
    #                if re.match(
    #                    "^SparseLibri(2|3|23)Mix$|^Libri(2|3|23)Mix$", self.dataset_name
    #                ):
    #                    path = os.path.join(
    #                        self.spk_path, self.filename2aux[file][speaker_id] + ".pt"
    #                    )
    #                elif self.dataset_name == "icmc":
    #                    if self.is_train:
    #                        path = os.path.join(
    #                            self.spk_path, self.data2spk[f"{file}/{speaker_id}"] + ".pt"
    #                        )
    #                    else:
    #                        path = os.path.join(
    #                            self.spk_path,
    #                            file,
    #                            self.data2spk[f"{file}/{speaker_id}"] + ".pt",
    #                        )
    #                else:
    #                    if self.dataset_name == "alimeeting":
    #                        audio_filename = speaker_id
    #                    elif (
    #                        self.dataset_name == "ami"
    #                        or self.dataset_name == "callhome_sim"
    #                        or self.dataset_name == "libri_css_sim"
    #                    ):
    #                        audio_filename = self.id2fullid[file][str(speaker_id)]
    #                    path = os.path.join(
    #                        self.spk_path, file, str(audio_filename) + ".pt"
    #                    )
    #                feature = torch.load(path, map_location="cpu")
    #                logger.info(f"path: {path}!")
    #            logger.info(f"feature size: {feature.size()}!")
    #            if len(feature.size()) == 2:
    #                if self.is_train:
    #                    feature = feature[random.randint(0, feature.shape[0] - 1), :]
    #                else:
    #                    # feature = torch.mean(feature, dim = 0)
    #                    feature = torch.mean(feature, dim=0)
    #            target_speeches.append(feature)
    #            logger.info(f"target_speeches len: {len(target_speeches)}") #{target_speeches[0].shape}, {target_speeches[1].shape}, {target_speeches[2].shape}")
    #        target_speeches = torch.stack(target_speeches)
    #        return target_speeches

    def load_ts(self, file, speaker_ids, rc=0):
        # assert self.dataset_name.startswith('Libri') or self.dataset_name.startswith('SparseLibri'), "only support librimix dataset"
        target_speeches = []
        exist_spk = []
        speaker_id_full_list = []
        for speaker_id in speaker_ids:
            if speaker_id != -1 and speaker_id != -2:
                if (
                    self.dataset_name.startswith("Libri")
                    and self.dataset_name.endswith("Mix")
                    or self.dataset_name.startswith("SparseLibri")
                ):
                    exist_spk.append(self.data2spk[self.filename2aux[file][speaker_id]])
                elif self.dataset_name == "libri_css_sim":
                    audio_filename = self.id2fullid[file][str(speaker_id)]
                    exist_spk.append(self.data2spk[f"{file}/{audio_filename}"])

        ts_mask = []
        for speaker_id in speaker_ids:
            if speaker_id == -1:
                if (
                    np.random.choice(2, p=[1 - self.zero_ratio, self.zero_ratio]) == 0
                    and self.is_train
                ):
                    random_spk = random.choice(list(self.spk2data))
                    while random_spk in exist_spk:
                        random_spk = random.choice(list(self.spk2data))
                    exist_spk.append(random_spk)
                    random_speech = random.choice(self.spk2data[random_spk])
                    if (
                        self.dataset_name.startswith("Libri")
                        and self.dataset_name.endswith("Mix")
                        or self.dataset_name.startswith("SparseLibri")
                    ):
                        if len(random_speech.split("-")) > 1:
                            spk, chapter, _ = random_speech.split("-")
                            path = os.path.join(
                                self.aux_path, f"{spk}/{chapter}/{random_speech}.flac"
                            )
                        else:
                            spk = random_speech
                            path = os.path.join(self.aux_path, f"{random_speech}.wav")
                    elif self.dataset_name == "libri_css_sim":
                        spk = random_spk
                        path = os.path.join(self.aux_path, f"{random_speech}.wav")
                else:
                    spk = "-1"
                    path = None
            elif speaker_id == -2:
                spk = "-2"
                path = None
            else:
                if (
                    self.dataset_name.startswith("Libri")
                    and self.dataset_name.endswith("Mix")
                    or self.dataset_name.startswith("SparseLibri")
                ):
                    if len(self.filename2aux[file][speaker_id].split("-")) > 1:
                        spk, chapter, _ = self.filename2aux[file][speaker_id].split("-")
                        path = os.path.join(
                            self.aux_path,
                            f"{spk}/{chapter}/{self.filename2aux[file][speaker_id]}.flac",
                        )
                    else:
                        spk = self.filename2aux[file][speaker_id]
                        path = os.path.join(
                            self.aux_path, f"{self.filename2aux[file][speaker_id]}.wav"
                        )
                elif self.dataset_name == "libri_css_sim":
                    spk = self.id2fullid[file][str(speaker_id)]
                    path = os.path.join(
                        self.aux_path,
                        file,
                        f"{self.id2fullid[file][str(speaker_id)]}.wav",
                    )

            speaker_id_full_list.append(spk)
            if path is not None and librosa.get_duration(path=path) > 0.01:
                aux_len = librosa.get_duration(path=path)
                if aux_len <= self.ts_len:
                    target_speech, _ = sf.read(path)
                else:
                    if self.fbank_input_aux:
                        sr_cur = 16000
                    else:
                        sr_cur = self.sample_rate
                    if self.is_train:
                        start_frame = np.int64(
                            random.random() * (aux_len - self.ts_len) * sr_cur
                        )
                    else:
                        start_frame = 0
                    if self.inference:
                        target_speech, _ = self.read_audio_with_resample(
                            path, sr_cur=sr_cur, rc=rc
                        )
                    else:
                        target_speech, _ = self.read_audio_with_resample(
                            path,
                            start=start_frame,
                            length=int(self.ts_len * sr_cur),
                            sr_cur=sr_cur,
                            rc=rc,
                        )

                target_speech = torch.FloatTensor(np.array(target_speech))
                ts_mask.append(1)
            else:
                target_speech = torch.zeros(192)  # fake one
                ts_mask.append(0)

            if self.fbank_input_aux:
                if target_speech.size(0) != 192:
                    target_speeches.append(self.feature_extractor(target_speech))
                else:
                    target_speeches.append(torch.zeros(10, 80))
            else:
                target_speeches.append(target_speech)

        target_len = torch.tensor(
            [ts.size(0) for ts in target_speeches], dtype=torch.long
        )
        ts_mask = torch.tensor(ts_mask, dtype=torch.long)
        if self.fbank_input_aux:
            target_speeches = _collate_data(target_speeches, is_embed_input=True)
        else:
            target_speeches = _collate_data(target_speeches, is_audio_input=True)

        return target_speeches, ts_mask, target_len, speaker_id_full_list

    def __len__(self):
        return len(self.sizes)

    def size(self, index):
        return self.sizes[index]

    def num_tokens(self, index):
        return self.size(index)

    def collater(self, samples):
        if len([s["labels"] for s in samples]) == 0:
            return {}

        ref_speech_len = [s["ref_speech"].size(1) for s in samples]
        if sum(ref_speech_len) == len(ref_speech_len) * self.rs_len * self.sample_rate:
            labels = torch.stack([s["labels"] for s in samples], dim=0)
            ref_speech = torch.stack([s["ref_speech"] for s in samples], dim=0)
        else:
            labels = _collate_data([s["labels"] for s in samples], is_label_input=True)
            # if self.embed_input or self.fbank_input:
            #     ref_speech = _collate_data([s["ref_speech"] for s in samples], is_embed_input=True)
            # else:
            ref_speech = _collate_data(
                [s["ref_speech"] for s in samples], is_embed_input=True
            )

        target_speech = torch.stack([s["target_speech"] for s in samples], dim=0)
        labels_len = torch.tensor(
            [s["labels"].size(1) for s in samples], dtype=torch.long
        )

        if not self.support_mc:
            assert ref_speech.size(1) == 1
            ref_speech = ref_speech[:, 0, :]
        net_input = {
            "ref_speech": ref_speech,
            "target_speech": target_speech,
            "labels": labels,
            "labels_len": labels_len,
            "file_path": [s["file_path"] for s in samples],
            "speaker_ids": [s["speaker_ids"] for s in samples],
            "start": [s["start"] for s in samples],
        }

        batch = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": net_input,
        }

        return batch

    def ordered_indices(self):
        order = [np.random.permutation(len(self))]
        order.append(self.sizes)
        return np.lexsort(order)[::-1]

    def add_rev(self, audio, length):
        rir_file = random.choice(self.rir_files)
        rir, _ = self.read_audio_with_resample(rir_file)
        rir = np.expand_dims(rir.astype(float), 0)
        rir = rir / np.sqrt(np.sum(rir**2))
        return signal.convolve(audio, rir, mode="full")[:, :length]

    def add_noise(self, audio, noisecat, length):
        clean_db = 10 * np.log10(max(1e-4, np.mean(audio**2)))
        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(
            self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1])
        )
        noises = []
        for noise in noiselist:
            noiselength = wave.open(noise, "rb").getnframes()
            noise_sample_rate = wave.open(noise, "rb").getframerate()
            if noise_sample_rate != self.sample_rate:
                noiselength = int(noiselength * self.sample_rate / noise_sample_rate)
            if noiselength <= length:
                noiseaudio, _ = self.read_audio_with_resample(noise)
                noiseaudio = np.pad(noiseaudio, (0, length - noiselength), "wrap")
            else:
                start_frame = np.int64(random.random() * (noiselength - length))
                noiseaudio, _ = self.read_audio_with_resample(
                    noise, start=start_frame, length=length
                )
            noiseaudio = np.stack([noiseaudio], axis=0)
            noise_db = 10 * np.log10(max(1e-4, np.mean(noiseaudio**2)))
            noisesnr = random.uniform(
                self.noisesnr[noisecat][0], self.noisesnr[noisecat][1]
            )
            noises.append(
                np.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio
            )
        noise = np.sum(np.concatenate(noises, axis=0), axis=0, keepdims=True)
        if noise.shape[1] < length:
            assert length - noise.shape[1] < 10
            audio[:, : noise.shape[1]] = noise + audio[:, : noise.shape[1]]
            return audio
        else:
            return noise[:, :length] + audio

    def choose_and_add_noise(self, noise_type, ref_speech, frame_len):
        assert self.musan_path is not None
        if noise_type == 0:
            return self.add_noise(ref_speech, "speech", length=frame_len)
        elif noise_type == 1:
            return self.add_noise(ref_speech, "music", length=frame_len)
        elif noise_type == 2:
            return self.add_noise(ref_speech, "noise", length=frame_len)

    def read_audio_with_resample(
        self, audio_path, start=None, length=None, sr_cur=None, support_mc=False, rc=-1
    ):
        if sr_cur is None:
            sr_cur = self.sample_rate
        audio_sr = librosa.get_samplerate(audio_path)

        if audio_sr != self.sample_rate:
            try:
                if start is not None:
                    audio, _ = librosa.load(
                        audio_path,
                        offset=start / sr_cur,
                        duration=length / sr_cur,
                        sr=sr_cur,
                        mono=False,
                    )
                else:
                    audio, _ = librosa.load(audio_path, sr=sr_cur, mono=False)
            except Exception as e:
                logger.info(e)
                audio, _ = librosa.load(audio_path, sr=sr_cur, mono=False)
                audio = audio[start : start + length]
            if len(audio.shape) > 1 and not support_mc:
                if self.random_channel and self.is_train:
                    if rc == -1:
                        rc = np.random.randint(0, audio.shape[1])
                    audio = audio[rc, :]
                else:
                    # use reference channel
                    audio = audio[0, :]
        else:
            try:
                if start is not None:
                    audio, _ = sf.read(
                        audio_path, start=start, stop=start + length, dtype="float32"
                    )
                else:
                    audio, _ = sf.read(audio_path, dtype="float32")
            except Exception as e:
                logger.info(e)
                audio, _ = sf.read(audio_path, dtype="float32")
                audio = audio[start : start + length]
            if len(audio.shape) > 1 and not support_mc:
                if self.random_channel and self.is_train:
                    if rc == -1:
                        rc = np.random.randint(0, audio.shape[1])
                    audio = audio[:, rc]
                else:
                    # use reference channel
                    audio = audio[:, 0]
            else:
                audio = np.transpose(audio)

        return audio, rc
