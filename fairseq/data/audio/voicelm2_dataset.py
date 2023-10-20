# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import os
import sys
import random
import secrets

from typing import Any, List, Optional, Union

import numpy as np

import torch
import torch.nn.functional as F
from fairseq.data import data_utils
from fairseq.data.fairseq_dataset import FairseqDataset
from fairseq.data.audio.audio_utils import (
    parse_path,
    read_from_stored_zip,
)
import io
from skimage.util.shape import view_as_windows # for cut big list into small list

# import torch.nn as nn
# from fairseq import search, utils
# from fairseq.models import FairseqIncrementalDecoder
# from torch import Tensor
# from fairseq.ngram_repeat_block import NGramRepeatBlock


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
                inds.append(ind)
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
    logger.info(f"label_path:{label_path}")
    with open(label_path) as f:
        code_lengths = [len(line.encode("utf-8")) for line in f]
        assert (
            len(code_lengths) == tot
        ), f"number of labels does not match ({len(code_lengths)} != {tot})"
        offsets = list(itertools.accumulate([0] + code_lengths))
        offsets = [(offsets[i], offsets[i + 1]) for i in inds]
    return offsets


def verify_label_lengths(
    audio_sizes,
    audio_rate,
    label_path,
    label_rate,
    inds,
    tot,
    tol=0.1,  # tolerance in seconds
):
    if label_rate < 0:
        logger.info(f"{label_path} is sequence label. skipped")
        return

    with open(label_path) as f:
        lengths = [len(line.rstrip().split()) for line in f]
        assert len(lengths) == tot
        lengths = [lengths[i] for i in inds]
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
## 1. in the get_text(),construct multimodal utterance in raw text phone code.
## 2. different from av-hubert, our fusion style is either residual cross attention or add.


### The version of this dataset requires number of text utternce same as  number of speech utterance.
### however,it is random text utterances of in per batch.
### and add text ntokens into batch, this parameter is only used by computing ctc loss for text part
class Voicelm2Dataset(FairseqDataset):
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
        store_labels: bool = True,
        random_crop: bool = False,
        single_target: bool = False,
        is_s2s: bool = False,  ## it is used to determine ctc finetune or cross entropy loss finetune.
        ## if it is true, target text is origanized for cross entropy loss fintune.
        ## otherwise, target text is origanized for ctc loss fintune.
        text_drop: bool = False,  # if it is true, speech and paired text are used to finetune model, unpair text is missing.
        # if it is false, speech and paired code label and unpair text code are used to pretrain model.

        pair_data: bool = False, # if false, it means speech and text is unpaired , otherwise it is paired
    ):
        self.audio_root, self.audio_names, inds, tot, self.sizes = load_audio(
            manifest_path, max_keep_sample_size, min_keep_sample_size
        )
        
        if not text_drop and manifest_text_path is not None:
            text_uttids, text_contents = load_text(
                manifest_text_path, max_keep_phone_size, min_keep_phone_size
            )
        else:
            text_uttids=None
            text_contents=None
        self.manifest_text_path = manifest_text_path
        self.text_uttids = text_uttids
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
        self.store_labels = store_labels
        if store_labels:
            self.label_list = [load_label(p, inds, tot) for p in label_paths]
        else:
            self.label_paths = label_paths
            #logger.info(f"label_paths: {label_paths} in __init__")
            self.label_offsets_list = [
                load_label_offset(p, inds, tot) for p in label_paths
            ]
        assert label_processors is None or len(label_processors) == self.num_labels
        for label_path, label_rate in zip(label_paths, self.label_rates):
            verify_label_lengths(
                self.sizes, sample_rate, label_path, label_rate, inds, tot
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

    def get_random_text_utt(self,index):
        utt=""
        if self.text_uttids is not None and not self.pair_data:
            if len(self.text_uttids) >= len(self.audio_names):
                logger.info(f"at len(self.text_uttids) > len(self.audio_names) !!!")
                steps=len(self.audio_names)
                #sublist = [self.text_contents[i:i+steps] for i in range(0,len(self.text_uttids),steps)] # 
                windows_shape=(steps,)
                sublists = view_as_windows(np.array(self.text_contents),windows_shape,step=steps) # 2-dim list, the time it consumes is almost a constant. 
                                                                                                  # it is very important for cut big list into small list.
                idx = np.random.choice(np.arange(len(sublists)))  ## np.arange(nums), is nums is very big, so np.arange(nums) will consum big time.
                utt = sublists[idx][index]
            elif len(self.text_uttids) < len(self.audio_names):
                logger.info(f"at len(self.text_uttids) < len(self.audio_names) !!!")
                list_id = np.arange(len(self.text_uttids))
                idx = np.random.choice(list_id)
                while idx == index and len(list_id) > 1: ## it requires that the audio and text have an equal number of sentences/
                    idx = np.random.choice(list_id)
                utt = self.text_contents[idx]
        elif self.text_uttids is not None and not self.pair_data:
             utt = self.text_contents[index]  ## str
        return utt


    def get_text(self, index):
        #print(f"in the get_text func: index: {index}")
        #utt = self.text_contents[index]  ## str
        utt = self.get_random_text_utt(index) ## str

        label = self.get_label(index, 0)  ## label is a utt speech label, it is a tensor
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
        for s in utt.split():
            if s in labels_keys_list:
                frames_count_list = label2counts[s]
                n = secrets.choice(
                    frames_count_list
                )  ## Choose a random item from the list securely
                new_l.extend([s] * int(n))
            else:
                new_l.extend([s])
        new_utt = " ".join(new_l)  ## str, it is multi modal text seq utterance

        ## encode every  text utterances into tensor
        if self.text_processors is not None:
            utt = self.text_processors[0](new_utt)
        return utt

#    def get_text(self, index):
#        #print(f"in the get_text func: index: {index}")
#        ## random select text utterance, and it doesn't require the audio equivalent of the labe;
#        utt=""
#        if self.text_uttids is not None and not self.pair_data:
#           #list_id = np.arange(len(self.text_uttids))
#           #idx = np.random.choice(list_id)
#           #utt = self.text_contents[idx]  ## str
#           utt = np.random.choice(self.text_contents)
#           logger.info(f"in the get_text(): utt: {utt}, index: {index} ")
#        elif self.text_uttids is not None and self.pair_data:
#            utt = self.text_contents[index]  ## str     
#   
#        label = self.get_label(index, 0)  ## label is a utt speech label, it is a tensor
#        label_unique, count = torch.unique_consecutive(label, return_counts=True)
#        label2counts = dict()
#
#        for ele, c in zip(label_unique.tolist(), count.tolist()):
#            ele = str(ele)  ## int to str
#            c = str(c)
#            if ele not in label2counts.keys():
#                label2counts[ele] = [c]
#            else:
#                label2counts[ele] += [c]  ### list splicing
#
#        unqiue_labels = len(label2counts)
#        k = unqiue_labels // 2
#        labels_keys_list = random.choices(list(label2counts), k=k)
#        new_l = []
#        for s in utt.split():
#            if s in labels_keys_list:
#                frames_count_list = label2counts[s]
#                n = secrets.choice(
#                    frames_count_list
#                )  ## Choose a random item from the list securely
#                new_l.extend([s] * int(n))
#            else:
#                new_l.extend([s])
#        new_utt = " ".join(new_l)  ## str, it is multi modal text seq utterance
#：
#        ## encode every  text utterances into tensor
#        if self.text_processors is not None:
#            utt = self.text_processors[0](new_utt)
#        return utt

    def get_label(self, index, label_idx):
        if self.store_labels:
            label = self.label_list[label_idx][index]
        else:
            with open(self.label_paths[label_idx]) as f:
                offset_s, offset_e = self.label_offsets_list[label_idx][index]
                f.seek(offset_s)
                label = f.read(offset_e - offset_s)

        if self.label_processors is not None:
            label = self.label_processors[label_idx](label)
        return label

    def get_labels(self, index):
        return [self.get_label(index, i) for i in range(self.num_labels)]

         
#    def __getitem__(self, index):
#        wav = self.get_audio(index)
#        if self.text_uttids is not None and not self.pair_data:
#            # choose text
#            list_id = np.arange(len(self.text_uttids))
#            idx = np.random.choice(list_id)
#            while idx == index and len(list_id) > 1: ## it requires that the audio and text have an equal number of sentences/
#                idx = np.random.choice(list_id) 
#            text = self.get_text(idx)
#            labels = self.get_labels(index)
#            return {"id": index, "source": wav, "text": text, "label_list": labels}
#        elif self.text_uttids is not None and self.pair_data:
#            text = self.get_text(index)
#            labels = self.get_labels(index)
#            return {"id": index, "source": wav, "text": text, "label_list": labels}
#        else:
#            labels = self.get_labels(index)
#            return {"id": index, "source": wav, "label_list": labels}
#

    def __getitem__(self, index):
        wav = self.get_audio(index)
        if self.text_uttids is not None:
            text = self.get_text(index)
            #logger.info(f"__getitem__:text: {text}, index: {index}")
            labels = self.get_labels(index)
            return {"id": index, "source": wav, "text": text, "label_list": labels}
        else:
            labels = self.get_labels(index)
            return {"id": index, "source": wav, "label_list": labels}
    def __len__(self):
        return len(self.sizes)

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
        #texts = [[s["text"] for s in samples]]
        # logger.info(f"in collater, texts lengths : {len(texts)}, texts : {texts}") # texts lengths=1,
        if not self.text_drop and self.manifest_text_path is not None:  ## pretrain mode or finetune mode with unpaired text code
            #logger.info(f"self.text_drop: {self.text_drop}, manifest_text_path: {self.manifest_text_path}")
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
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.sizes)
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


