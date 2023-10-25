# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from itertools import chain
import logging
import os
import sys
import random
import secrets

from typing import Any, List, Optional, Union

import numpy as np

import torch
from torch import Tensor
import torch.nn.functional as F
from fairseq.data import data_utils
from fairseq.data.fairseq_dataset import FairseqDataset
from fairseq.data.audio.audio_utils import (
    parse_path,
    read_from_stored_zip,
)
import io
from skimage.util.shape import view_as_windows  # for cut big list into small list


logger = logging.getLogger(__name__)


def load_audio(manifest_path, max_keep, min_keep):
    n_long, n_short = 0, 0
    names, inds, sizes = [], [], []
    with open(manifest_path) as f:
        root = f.readline().strip()
        for ind, line in enumerate(f):
            items = line.strip().split("\t")
            assert len(items) == 2, line
            sz = int(items[1])
            if min_keep is not None and sz < min_keep:
                n_short += 1
            elif max_keep is not None and sz > max_keep:
                n_long += 1
            else:
                names.append(items[0])
                inds.append(
                    ind
                )  ## inds is very import , it is used to index audio label.
                sizes.append(sz)
    tot = ind + 1
    logger.info(
        (
            f"max_keep={max_keep}, min_keep={min_keep}, "
            f"loaded {len(names)}, skipped {n_short} short and {n_long} long, "
            f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
        )
    )
    return root, names, inds, tot, sizes


def repeat_audio(
    audio_names: List[str],
    audio_inds: List[int],
    audio_sizes: List[int],
    tot: int,
    text_ratio: int,
) -> Tuple[List[str], List[int], List[int], int]:
    audio_namess = []
    audio_indss = []
    audio_sizess = []
    tots = []
    for i in range(text_ratio):
        audio_namess.append(audio_names)
        audio_indss.append(audio_inds)
        audio_sizess.append(audio_sizes)
        tots.append(tot)
    audio_namess = list(
        chain.from_iterable(audio_namess)
    )  ## 2-dim list -> flatten 1-dim list
    audio_indss = list(
        chain.from_iterable(audio_indss)
    )  ## 2-dim list -> flatten 1-dim list
    audio_sizess = list(
        chain.from_iterable(audio_sizess)
    )  ## 2-dim list -> flatten 1-dim list
    tots = sum(tots)  # int
    return audio_namess, audio_indss, audio_sizess, tots  ##
    # return audio_namess


def repeat_label(labels: List[Tensor], text_ratio: int) -> List[Tensor]:
    labelss = []
    for i in range(text_ratio):
        labelss.append(labels)
    labelss = list(chain.from_iterable(labelss))  ## 2-dim list -> flatten 1-dim list
    return labelss


def load_text(manifest_text_path, max_keep, min_keep):
    logger.info(f"manifest_text_path: {manifest_text_path}")
    text_contents = []
    text_uttids = []
    sizes = []
    n_long = 0
    n_short = 0
    with open(manifest_text_path, "r") as f:
        for i, line in enumerate(f):
            items = line.strip()
            sz = len(items.split())
            if min_keep is not None and sz < min_keep:
                n_short += 1
            elif max_keep is not None and sz > max_keep:
                n_long += 1
            else:
                text_uttids.append(i)
                text_contents.append(items)
                sizes.append(sz)
    logger.info(
        f"max_keep={max_keep}, min_keep={min_keep},  "
        f"loaded {len(text_uttids)} texts, skipped {n_short} short and {n_long} long "
        f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
    )
    return text_uttids, text_contents


def load_label(label_path, inds, tot):
    with open(label_path) as f:
        labels = [line.rstrip() for line in f]
        assert (
            len(labels) == tot
        ), f"number of labels does not match ({len(labels)} != {tot})"
        labels = [labels[i] for i in inds]
    return labels


def load_label_offset(label_path, inds, tot):
    """
    >>> label_path="tests/dev-clean_10.speechphncode"
    >>> inds = [0,2,4,7,8,9] ## audio utts index list
    >>> with open(label_path) as f:
    ...     code_lengths = [len(line.encode("utf-8")) for line in f]
    ...     print(code_lengths)
    ...     offsets = list(itertools.accumulate([0] + code_lengths))
    ...     print(offsets)
    ...     offsets = [(offsets[i], offsets[i + 1]) for i in inds]
    ...     print(offsets)
    ...
    [786, 1364, 780, 961, 884, 1220, 433, 913, 348, 1307]
    [0, 786, 2150, 2930, 3891, 4775, 5995, 6428, 7341, 7689, 8996]
    [(0, 786), (2150, 2930), (3891, 4775), (6428, 7341), (7341, 7689), (7689, 8996)]
    """
    logger.info(f"label_path:{label_path}")
    with open(label_path) as f:
        code_lengths = [len(line.encode("utf-8")) for line in f]
        assert (
            len(code_lengths) == tot
        ), f"number of labels does not match ({len(code_lengths)} != {tot})"
        offsets = list(itertools.accumulate([0] + code_lengths))
        offsets = [(offsets[i], offsets[i + 1]) for i in inds]
    return offsets


def get_pre_labels(label_path, inds, tot, sizes) -> List[str]:
    """get labels befor label_processing"""
    label_offsets_list = load_label_offset(label_path, inds, tot)
    indexs = np.arange(len(sizes))
    labels = []

    with open(label_path) as f:
        for ind in indexs:
            offset_s, offset_e = label_offsets_list[ind]
            f.seek(offset_s)
            label = f.read(offset_e - offset_s)  ## str
            # logger.info(f"type(label): {type(label)}, label: {label}")
            # assert label is str, f"label: {label}"
            labels.append(label)  # List[str]
    return labels


def prepare_multi_modal_text_utt(label: Tensor, text_utt: str) -> str:
    """prepare one multi modal text utterance base on one audio label"""

    ### ## strs -> tensor
    label = label.strip().split()
    label_tensor = torch.IntTensor(len(label))  ## random elements tensor
    for i, element in enumerate(label):
        label_tensor[i] = int(element)
    label = label_tensor

    label_unique, count = torch.unique_consecutive(label, return_counts=True)
    label2counts = dict()

    for ele, c in zip(label_unique.tolist(), count.tolist()):
        ele = str(ele)  ## int to str
        c = str(c)
        if ele not in label2counts.keys():
            label2counts[ele] = [c]
        else:
            label2counts[ele] += [c]  ### list splicing

    unqiue_labels = len(label2counts)
    k = unqiue_labels // 2
    labels_keys_list = random.choices(list(label2counts), k=k)
    new_l = []
    for s in text_utt.split():
        if s in labels_keys_list:
            frames_count_list = label2counts[s]
            n = secrets.choice(
                frames_count_list
            )  ## Choose a random item from the list securely
            new_l.extend([s] * int(n))
        else:
            new_l.extend([s])
    new_utt = " ".join(new_l)  ## str, it is multi modal text seq utterance
    return new_utt


## step: load big text -> random cut into small text, on small text, we construt multi modal text utterance
def get_small_list_from_big_list(
    text_contents: List[str], audio_names: List[str]
) -> List[str]:
    assert len(text_contents) >= len(
        audio_names
    ), f"len(text_contents): {len(text_contents)}, len(audio_names): {len(audio_names)}"
    logger.info(f"at len(text_contents) > len(audio_names) !!!")
    steps = len(audio_names)
    # sublist = [text_contents[i:i+steps] for i in range(0,len(text_uttids),steps)] #
    windows_shape = (steps,)
    sublists_contents = view_as_windows(
        np.array(text_contents), windows_shape, step=steps
    )  # 2-dim list, the time it consumes is almost a constant.
    # it is very important for cut big list into small list.
    # length of last elements  may be  less than `steps`
    logger.info(
            f"model actual using text utterance nums: {steps}, it accounts for a {len(sublists_contents)} of the total text ! "
    )
    idx = np.random.choice(
        np.arange(len(sublists_contents))
    )  ## np.arange(nums), is nums is very big, so np.arange(nums) will consum big time.
    logger.info(f"after cut text, choice index: {idx}!!! ")
    text_contents = sublists_contents[
        idx
    ]  # List[str] its length is normal number, not very big number.

    return text_contents


def load_post_text(
    text_contents: List[str], labels: List[str]
) -> Tuple[List[int], List[str]]:
    # labels = get_pre_labels(label_path, inds, tot, sizes)
    new_utts = []
    uttids = []
    list_id = np.arange(len(text_contents))
    for i, label in enumerate(labels):
        ## random select one utterance text
        idx = np.random.choice(list_id)
        utt = text_contents[idx]
        new_utt = prepare_multi_modal_text_utt(label, utt)
        new_utts.append(new_utt)
        uttids.append(i)

    logger.info(f"model input data utt nums: {len(uttids)}!")
    return uttids, new_utts


def post_final_audio_text(
    label_path,
    manifest_path,
    max_keep_sample_size,
    min_keep_sample_size,
    manifest_text_path,
    max_keep_phone_size,
    min_keep_phone_size,
    text_ratio,
):
    ## step1: load text
    ## step0: load audio
    audio_root, audio_names, inds, tot, sizes = load_audio(
        manifest_path, max_keep_sample_size, min_keep_sample_size
    )
    text_uttids, text_contents = load_text(
        manifest_text_path, max_keep_phone_size, min_keep_phone_size
    )

    if len(text_contents) > len(audio_names) and text_ratio > 1:
        ## repeat audio
        audio_namess, audio_indss, audio_sizess, tots = repeat_audio(
            audio_names, audio_inds, audio_sizes, tot, text_ratio
        )
        ## repeat label
        labels = get_pre_labels(label_path[0], inds, tot, sizes)
        labels = repeat_label(labels, text_ratio)
        ## prepare text
        text_contents = get_small_list_from_big_list(text_contents, audio_namess)
        text_uttids, text_contents = load_post_text(text_contents, labels)
    # elif len(text_contents) > len(self.audio_names) and text_ratio==1:
    elif len(text_contents) > len(audio_names) and text_ratio == 1:
        audio_namess = audio_names
        audio_indss = inds
        audio_sizess = sizes
        tots = tot
        text_contents = get_small_list_from_big_list(text_contents, audio_namess)
        labels = get_pre_labels(label_path[0], inds, tot, sizes)
        text_uttids, text_contents = load_post_text(text_contents, labels)

    else:
        audio_namess = audio_names
        audio_indss = inds
        audio_sizess = sizes
        tots = tot
        labels = get_pre_labels(label_path[0], inds, tot, sizes)
        text_uttids, text_contents = load_post_text(text_contents, labels)

    return (
        audio_root,
        audio_namess,
        audio_indss,
        audio_sizess,
        tots,
        labels,
        text_contents,
    )


def verify_label_lengths(
    audio_sizes,
    audio_rate,
    labels,
    label_rate,
    inds,
    tot,
    tol=0.1,  # tolerance in seconds
):
    if label_rate < 0:
        logger.info(f"{label_path} is sequence label. skipped")
        return

    lengths = [len(line.rstrip().split()) for line in labels]
    assert len(lengths) == tot
    num_invalid = 0
    for i, ind in enumerate(inds):
        dur_from_audio = audio_sizes[i] / audio_rate
        dur_from_label = lengths[i] / label_rate
        if abs(dur_from_audio - dur_from_label) > tol:
            logger.warning(
                (
                    f"audio and label duration differ too much "
                    f"(|{dur_from_audio} - {dur_from_label}| > {tol}) "
                    f"in line {ind+1} of {label_path}. Check if `label_rate` "
                    f"is correctly set (currently {label_rate}). "
                    f"num. of samples = {audio_sizes[i]}; "
                    f"label length = {lengths[i]}"
                )
            )
            num_invalid += 1
    if num_invalid > 0:
        logger.warning(
            f"total {num_invalid} (audio, label) pairs with mismatched lengths"
        )


## compared sthubert_dataset2.py , its difference is as follows:
## 1. We construct multimodal utterance in raw text phone code based on audio pesudo label
## 2. different from av-hubert, our fusion style is either residual cross attention or add.

## Voicelm2
## The core of it is to determine the number of unpaired audio label sentences based on the number of unpaired text  sentences.
## This allows training with several times more text than audio.
# This version of the dataset is as follows:
# 1. In data preparation, there is no requirement that the number of unpaired text and unpaired audio sentences be equal, or not equal.
# 2. At model input, the number of unpaired text sentences is equal to the number of unpaired audio sentences.
#    The number of sentences must be equal to the number of unpaired audio sentences.

## detail steps:
## step1: load_audio(), filter longest and shortest audio, get audio input of model
## step2: load_text(), filter longest and shortest text, output text_contents and text_uttids
## step3: load_label_offset(), filter unuse label based on audio utt index, get label offsets
## step4: get_pre_labels(), filter unuse label based on audio utt index, get labels list ,evey element is tensor type utt label.
## step5: get_small_list_from_big_list(), cut text_contents list into small list(equal to audio utterances)
## step6: load_post_text(), We construct multimodal utterance in raw text phone code based on audio pesudo label.
## step7: post_final_audio_text(), get more audio, andio label, multi-modal text sequences than raw audio utterances or equal to raw audio utterances.

class Voicelm2DatasetBigtext(FairseqDataset):
    def __init__(
        self,
        manifest_path: str,
        manifest_text_path: str,
        sample_rate: float,
        label_paths: List[str],
        label_rates: Union[List[float], float],  # -1 for sequence labels
        pad_list: List[str],
        eos_list: List[str],
        label_processors: Optional[List[Any]] = None,
        text_processors: Optional[List[Any]] = None,
        max_keep_sample_size: Optional[int] = None,
        min_keep_sample_size: Optional[int] = None,
        max_keep_phone_size: Optional[int] = None,
        min_keep_phone_size: Optional[int] = None,
        max_sample_size: Optional[int] = None,
        text_seq: bool = True,  ## if it is true,it will keep all text utterance. here text_seq is unpaired text.it is used as text modal
        shuffle: bool = True,
        pad_audio: bool = False,
        normalize: bool = False,
        #store_labels: bool = False,
        random_crop: bool = False,
        single_target: bool = False,
        is_s2s: bool = False,  ## it is used to determine ctc finetune or cross entropy loss finetune.
        ## if it is true, target text is origanized for cross entropy loss fintune.
        ## otherwise, target text is origanized for ctc loss fintune.
        text_drop: bool = False,  # if it is true, speech and paired text are used to finetune model, unpair text is missing.
        # if it is false, speech and paired code label and unpair text code are used to pretrain model.
        text_ratio: int = 1,  # more than 1, it means repeat audio utterances to  get the number of text utterances.
        pair_data: bool = False,  # if false, it means speech and text is unpaired , otherwise it is paired
    ):
        if not self.text_drop:
            (
                self.audio_root,
                self.audio_names,
                self.labels,
                self.text_contents,
                self.audio_indss,
                self.audio_sizess,
                self.tots,
            ) = post_final_audio_text(
                label_path,
                manifest_path,
                max_keep_sample_size,
                min_keep_sample_size,
                manifest_text_path,
                max_keep_phone_size,
                min_keep_phone_size,
                text_ratio,
            )
        else:
            self.text_contents=None

        self.manifest_text_path = manifest_text_path
        #self.text_uttids = text_uttids
        self.text_contents = text_contents
        self.sample_rate = sample_rate
        self.shuffle = shuffle
        self.random_crop = random_crop
        self.text_seq = text_seq
        self.num_labels = len(label_paths)
        self.pad_list = pad_list
        self.eos_list = eos_list
        self.text_processors = text_processors
        self.label_processors = label_processors
        self.single_target = single_target
        self.is_s2s = is_s2s
        self.text_drop = text_drop
        self.pair_data = pair_data

        self.label_rates = (
            [label_rates for _ in range(len(label_paths))]
            if isinstance(label_rates, float)
            else label_rates
        )
        assert label_processors is None or len(label_processors) == self.num_labels
        for label_path, label_rate in zip(label_paths, self.label_rates):
            verify_label_lengths(
                self.sizess, sample_rate, self.labels, label_rate, self.indss, self.tots
            )

        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.pad_audio = pad_audio
        self.normalize = normalize
        logger.info(
            f"pad_audio={pad_audio}, random_crop={random_crop}, "
            f"normalize={normalize}, max_sample_size={self.max_sample_size},"
            f"seqs2seq data={self.is_s2s}."
        )

    def get_audio(self, index):
        import soundfile as sf

        wav_path = os.path.join(self.audio_root, self.audio_names[index])
        _path, slice_ptr = parse_path(wav_path)
        if len(slice_ptr) == 0:
            wav, cur_sample_rate = sf.read(_path)
        else:
            assert _path.endswith(".zip")
            data = read_from_stored_zip(_path, slice_ptr[0], slice_ptr[1])
            f = io.BytesIO(data)
            wav, cur_sample_rate = sf.read(f)
        wav = torch.from_numpy(wav).float()
        wav = self.postprocess(wav, cur_sample_rate)
        return wav

    def get_text(self, index):
        utt = self.text_contents[index]
        ## encode every  text utterances into tensor
        if self.text_processors is not None:
            utt = self.text_processors[0](utt)
        return utt

    def get_label(self, index, label_idx):
        label = self.labels[index]
        if self.label_processors is not None:
            label = self.label_processors[label_idx](label)
        return label

    def get_labels(self, index):
        return [self.get_label(index, i) for i in range(self.num_labels)]

    def __getitem__(self, index):
        wav = self.get_audio(index)
        if self.text_contents is not None:
            text = self.get_text(index)
            # logger.info(f"__getitem__:text: {text}, index: {index}")
            labels = self.get_labels(index)
            return {"id": index, "source": wav, "text": text, "label_list": labels}
        else:
            labels = self.get_labels(index)
            return {"id": index, "source": wav, "label_list": labels}

    def __len__(self):
        return len(self.sizess)

    def crop_to_max_size(self, wav, target_size):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav, 0

        start, end = 0, target_size
        if self.random_crop:
            start = np.random.randint(0, diff + 1)
            end = size - diff + start
        return wav[start:end], start

    def collater(self, samples):
        # target = max(sizes) -> random_crop not used
        # target = max_sample_size -> random_crop used for long
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        audios = [s["source"] for s in samples]
        audio_sizes = [len(s) for s in audios]
        if self.pad_audio:
            audio_size = min(max(audio_sizes), self.max_sample_size)
        else:
            audio_size = min(min(audio_sizes), self.max_sample_size)
        collated_audios, padding_mask, audio_starts = self.collater_audio(
            audios, audio_size
        )
        # texts = [[s["text"] for s in samples]]
        # logger.info(f"in collater, texts lengths : {len(texts)}, texts : {texts}") # texts lengths=1,
        if (
            not self.text_drop and self.manifest_text_path is not None
        ):  ## pretrain mode or finetune mode with unpaired text code
            # logger.info(f"self.text_drop: {self.text_drop}, manifest_text_path: {self.manifest_text_path}")
            texts = [[s["text"] for s in samples]]
            collated_texts, text_lengths_list, text_ntokens_list = self.collater_text(
                texts, audio_size, audio_starts
            )
        else:  ## only for normal finetune (i.e.: only using speech and paired text)
            collated_texts, text_lengths_list, text_ntokens_list = (
                [None],
                [None],
                [None],
            )

        targets_by_label = [
            [s["label_list"][i] for s in samples] for i in range(self.num_labels)
        ]
        targets_list, label_lengths_list, label_ntokens_list = self.collater_label(
            targets_by_label, audio_size, audio_starts
        )
        source = {"audio": collated_audios, "text": collated_texts[0]}
        net_input = {"source": source, "padding_mask": padding_mask}

        batch = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": net_input,
        }
        ## single_target default is false for pretrain
        ## single_target default is true for finetune
        if self.single_target:
            batch["target_lengths"] = label_lengths_list[0]
            batch["ntokens"] = label_ntokens_list[0]
            if self.is_s2s:
                batch["target"], net_input["prev_output_tokens"] = (
                    targets_list[0][0],
                    targets_list[0][1],
                )
            else:
                batch["target"] = targets_list[0]
            batch["text_lengths_list"] = text_lengths_list[0]
            batch["text_ntokens_list"] = text_ntokens_list[0]
        else:
            batch["target_lengths_list"] = label_lengths_list
            batch["ntokens_list"] = label_ntokens_list
            batch["target_list"] = targets_list
            batch["text_lengths_list"] = text_lengths_list
            batch["text_ntokens_list"] = text_ntokens_list
        return batch

    def collater_audio(self, audios, audio_size):
        collated_audios = audios[0].new_zeros(len(audios), audio_size)
        padding_mask = (
            torch.BoolTensor(collated_audios.shape).fill_(False)
            # if self.pad_audio else None
        )
        audio_starts = [0 for _ in audios]
        for i, audio in enumerate(audios):
            diff = len(audio) - audio_size
            if diff == 0:
                collated_audios[i] = audio
            elif diff < 0:
                assert self.pad_audio
                collated_audios[i] = torch.cat([audio, audio.new_full((-diff,), 0.0)])
                padding_mask[i, diff:] = True
            else:
                collated_audios[i], audio_starts[i] = self.crop_to_max_size(
                    audio, audio_size
                )
        return collated_audios, padding_mask, audio_starts

    def collater_frm_label(self, targets, audio_size, audio_starts, label_rate, pad):
        assert label_rate > 0
        s2f = label_rate / self.sample_rate
        frm_starts = [int(round(s * s2f)) for s in audio_starts]
        frm_size = int(round(audio_size * s2f))
        if not self.pad_audio:
            rem_size = [len(t) - s for t, s in zip(targets, frm_starts)]
            frm_size = min(frm_size, *rem_size)
        targets = [t[s : s + frm_size] for t, s in zip(targets, frm_starts)]
        logger.debug(f"audio_starts={audio_starts}")
        logger.debug(f"frame_starts={frm_starts}")
        logger.debug(f"frame_size={frm_size}")

        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        targets = data_utils.collate_tokens(targets, pad_idx=pad, left_pad=False)
        return targets, lengths, ntokens

    def collater_seq_label(self, targets, pad):
        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        targets = data_utils.collate_tokens(targets, pad_idx=pad, left_pad=False)
        return targets, lengths, ntokens

    ## it is from https://github.com/facebookresearch/av_hubert/blob/main/avhubert/hubert_dataset.py#L488
    def collater_seq_label_s2s(self, targets, pad):
        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        pad, eos = (
            self.label_processors[0].dictionary.pad(),
            self.label_processors[0].dictionary.eos(),
        )
        targets_ = data_utils.collate_tokens(
            targets, pad_idx=pad, eos_idx=eos, left_pad=False
        )
        prev_output_tokens = data_utils.collate_tokens(
            targets,
            pad_idx=pad,
            eos_idx=eos,
            left_pad=False,
            move_eos_to_beginning=True,
        )
        return (targets_, prev_output_tokens), lengths, ntokens

    def collater_frm_text(self, texts, audio_size, audio_starts, label_rate, pad):
        ## this function is not used.
        assert label_rate > 0
        s2f = label_rate / self.sample_rate
        frm_starts = [int(round(s * s2f)) for s in audio_starts]
        frm_size = int(round(audio_size * s2f))
        ### (TODO) text must be equal to speech lenght?
        # if not self.pad_audio:
        #    rem_size = [len(t) - s for t, s in zip(texts, frm_starts)]
        #    frm_size = min(frm_size, *rem_size)
        texts = [t[s : s + frm_size] for t, s in zip(texts, frm_starts)]
        logger.info(f"in collater_frm_text audio_starts={audio_starts}")
        logger.info(f"in collater_frm_text frame_starts={frm_starts}")
        logger.info(f"in collater_frm_text frame_size={frm_size}")

        lengths = torch.LongTensor([len(t) for t in texts])
        # logger.info(f"in collater_frm_text , texts: {texts}")
        ntokens = lengths.sum().item()
        texts = data_utils.collate_tokens(texts, pad_idx=pad, left_pad=False)
        return texts, lengths, ntokens

    def collater_seq_text(self, texts, pad):
        # logger.info(f"in collater_seq_text, texts lengths : {len(texts)}, texts: {texts}")
        lengths = torch.LongTensor([len(t) for t in texts])
        ntokens = lengths.sum().item()
        texts = data_utils.collate_tokens(texts, pad_idx=pad, left_pad=False)
        # logger.info(f"after collect_token func : texts : {texts}")
        return texts, lengths, ntokens

    def collater_text(self, texts_by_text, audio_size, audio_starts):
        texts_list, lengths_list, ntokens_list = [], [], []
        itr = zip(texts_by_text, self.label_rates, self.pad_list)
        for texts, label_rate, pad in itr:
            if self.text_seq:
                texts, lengths, ntokens = self.collater_seq_text(texts, pad)
            else:
                texts, lengths, ntokens = self.collater_frm_text(
                    texts, audio_size, audio_starts, label_rate, pad
                )
            texts_list.append(texts)
            lengths_list.append(lengths)
            ntokens_list.append(ntokens)
        return texts_list, lengths_list, ntokens_list

    def collater_label(self, targets_by_label, audio_size, audio_starts):
        targets_list, lengths_list, ntokens_list = [], [], []
        itr = zip(targets_by_label, self.label_rates, self.pad_list)
        for targets, label_rate, pad in itr:
            if label_rate == -1:
                if self.is_s2s:
                    targets, lengths, ntokens = self.collater_seq_label_s2s(
                        targets, pad
                    )
                else:
                    targets, lengths, ntokens = self.collater_seq_label(targets, pad)
            else:
                targets, lengths, ntokens = self.collater_frm_label(
                    targets, audio_size, audio_starts, label_rate, pad
                )
            targets_list.append(targets)
            lengths_list.append(lengths)
            ntokens_list.append(ntokens)
        return targets_list, lengths_list, ntokens_list

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        if self.pad_audio:
            return self.sizess[index]
        return min(self.sizess[index], self.max_sample_size)

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.sizess)
        return np.lexsort(order)[::-1]

    def postprocess(self, wav, cur_sample_rate):
        if wav.dim() == 2:
            wav = wav.mean(-1)
        assert wav.dim() == 1, wav.dim()

        if cur_sample_rate != self.sample_rate:
            raise Exception(f"sr {cur_sample_rate} != {self.sample_rate}")

        if self.normalize:
            with torch.no_grad():
                wav = F.layer_norm(wav, wav.shape)
        return wav
