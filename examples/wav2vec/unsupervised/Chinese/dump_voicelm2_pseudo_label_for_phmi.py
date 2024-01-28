import itertools
from itertools import chain
import logging
import os
import sys
import random
import secrets
import argparse
from typing import Any, List, Optional, Union, Tuple

import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from skimage.util.shape import view_as_windows  # for cut big list into small list
from fairseq.data import Dictionary
import fairseq
#from npy_append_array import NpyAppendArray
from common import ApplyKmeans
from tqdm import tqdm

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
    logging.info(
        (
            f"load audio data from: {manifest_path}, "
            f"max_keep={max_keep}, min_keep={min_keep}, "
            f"tot: {tot}, loaded {len(names)}, skipped {n_short} short and {n_long} long, "
            f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)} "
            f"audio_contents len: {len(names)}, "
            # f"in load_audio() inds: {inds}"
            # f"actual use {len(names)} tot: {tot} "
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


def repeat_label(labels: List[str], text_ratio: int) -> List[str]:
    labelss = []
    for i in range(text_ratio):
        labelss.append(labels)
    labelss = list(chain.from_iterable(labelss))  ## 2-dim list -> flatten 1-dim list
    return labelss


def load_text(manifest_text_path, max_keep, min_keep):
    # logging.info(f"manifest_text_path: {manifest_text_path}")
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
    logging.info(
        f"load text data from: {manifest_text_path} "
        f"max_keep={max_keep}, min_keep={min_keep},  "
        f"loaded {len(text_uttids)} texts, skipped {n_short} short and {n_long} long "
        f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
        f"actual use {len(text_contents)}"
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
    logging.info(f"load audio label data from :{label_path}")
    with open(label_path) as f:
        code_lengths = [len(line.encode("utf-8")) for line in f]
        assert (
            len(code_lengths) == tot
        ), f"number of labels does not match ({len(code_lengths)} != {tot})"
        offsets = list(itertools.accumulate([0] + code_lengths))
        # logging.info(f"load_label_offset(): inds len: {len(inds)},tot: {tot}, offsets len: {len(offsets)}")
        # logging.info(f"inds len: {len(inds)}, offsets len: {len(offsets)}")
        # logging.info(f"inds: {inds}")
        # logging.info(f"offsets: {offsets}")
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
            # logging.info(f"type(label): {type(label)}, label: {label}")
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
    logging.info(f"at len(text_contents) > len(audio_names) !!!")
    steps = len(audio_names)
    # sublist = [text_contents[i:i+steps] for i in range(0,len(text_uttids),steps)] #
    windows_shape = (steps,)
    sublists_contents = view_as_windows(
        np.array(text_contents), windows_shape, step=steps
    )  # 2-dim list, the time it consumes is almost a constant.
    # it is very important for cut big list into small list.
    # length of last elements  may be  less than `steps`
    logging.info(
            f"model actual using text utterance nums: {steps}, it accounts for a {len(sublists_contents)} of the total text ! "
    )
    #idx = np.random.choice(
    #    np.arange(len(sublists_contents))
    #)  ## np.arange(nums), is nums is very big, so np.arange(nums) will consum big time.
    
    ## idx is from first iter voicelm2, so that the second iter data is complete same as the first iter data.
    idx = 6 
    logging.info(f"after cut text, choice index: {idx}!!! ")
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

    logging.info(f"model input data utt nums: {len(uttids)}!")
    return uttids, new_utts


def post_final_audio_text(
    label_paths,
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
    audio_root, audio_names, audio_inds, tot, audio_sizes = load_audio(
        manifest_path, max_keep_sample_size, min_keep_sample_size
    )
    text_uttids, text_contents = load_text(
        manifest_text_path, max_keep_phone_size, min_keep_phone_size
    )

    if len(text_contents) > len(audio_names) and text_ratio > 1:
        logging.info(f"len(text_contents) > len(audio_names) and text_ratio > 1!!! ")
        ## repeat audio
        audio_namess, audio_indss, audio_sizess, tots = repeat_audio(
            audio_names, audio_inds, audio_sizes, tot, text_ratio
        )
        ## repeat label
        labels = get_pre_labels(label_paths[0], audio_inds, tot, audio_sizes)
        labels = repeat_label(labels, text_ratio)
        # logging.info(f"labels part: {labels[:3]}")
        ## prepare text
        text_contents = get_small_list_from_big_list(text_contents, audio_namess)
        text_uttids, text_contents = load_post_text(text_contents, labels)
    elif len(text_contents) > len(audio_names) and text_ratio == 1:
        logging.info(f"len(text_contents) > len(audio_names) and text_ratio == 1!!!!")
        audio_namess = audio_names
        audio_indss = audio_inds
        audio_sizess = audio_sizes
        tots = tot
        text_contents = get_small_list_from_big_list(text_contents, audio_namess)
        labels = get_pre_labels(label_paths[0], audio_inds, tot, audio_sizes)
        # logging.info(f"labels part: {labels[:3]}")
        text_uttids, text_contents = load_post_text(text_contents, labels)
    else:
        logging.info(f"len(text_contents) < len(audio_names)!!!")
        audio_namess = audio_names
        audio_indss = audio_inds
        audio_sizess = audio_sizes
        tots = tot
        labels = get_pre_labels(label_paths[0], audio_inds, tot, audio_sizes)
        # logging.info(f"labels part: {labels[:3]}")
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


class TextEncoder(object):
    def __init__(self, dictionary: Dictionary) -> None:
        self.dictionary = dictionary

    def __call__(self, label: str) -> List[str]:
        return self.dictionary.encode_line(
            label,
            append_eos=False,
            add_if_not_exist=False,
        )


class Voicelm2FeatureReader(object):
    def __init__(
        self,
        sample_rate,
        ckpt_path,
        layer,
        label_paths,
        manifest_path,
        manifest_text_path,
        text_procs,
        #max_keep_sample_size=250000,
        #min_keep_sample_size=32000,
        #max_keep_phone_size=100,
        #min_keep_phone_size=50,
        #text_ratio=1,
    ):
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [ckpt_path]
        )
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model[0].eval().to(self.device)
        self.layer = layer
        self.task = task
        self.sample_rate = sample_rate
        self.text_procs = text_procs
        logging.info(f"Task config: \n {self.task.cfg}")
        self.task.cfg.max_sample_size
        max_sample_size=None
        min_sample_size=None
        max_phone_size=None
        min_phone_size=None
        (
            self.audio_root,
            self.audio_namess,
            self.audio_indss,
            self.audio_sizess,
            self.tots,
            self.labels,
            self.text_contents,
        ) = post_final_audio_text(
            label_paths,
            manifest_path,
            max_sample_size,
            min_sample_size,
            manifest_text_path,
            max_phone_size,
            min_phone_size,
            self.task.cfg.text_ratio,
        )

    def read_audio(self, index):
        wav_path = os.path.join(self.audio_root, self.audio_namess[index])
        wav, sr = sf.read(wav_path)
        assert sr == self.sample_rate, sr
        if wav.ndim == 2:
            wav = wav.mean(-1)
        wav = torch.from_numpy(wav).float()
        #wav = torch.from_numpy(wav).to(torch.float16)
        return wav

    def get_text(self, index):
        utt = self.text_contents[index]
        ## encode every  text utterances into tensor
        if self.text_procs is not None:
            utt = self.text_procs[0](utt)
        return utt

    def sizes(self):
        return len(self.audio_sizess)
    
    def get_feats(self, index):
        x = self.read_audio(index)
        text = self.get_text(index)
        with torch.no_grad() and torch.cuda.amp.autocast():
            x = x.to(self.device)
            x = x.view(1, -1)
            text = text.to(self.device)
            text = text.view(1, -1)
            source = {"audio": x, "text": text}
            feat, _ = self.model.extract_features(
                source=source, padding_mask=None, mask=False, output_layer=self.layer
            )  # (B,T,C)->(1,T,C)
            feat = feat.detach()
            #assert feat.requires_grad == False, f"feat.requires_grad = {feat.requires_grad}"
            feat = feat.squeeze(0).cpu()  # (T,C)
        
        return feat


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--text_dict",
        type=str,
        default="dataset/format/librispeech/40M_librispeech_lm_monophncode_librispeech_frame_monophncode_using_wav2vec-u2_model/dict.textphncode.txt",
        help="dict of text code sequence ",
    )
    parser.add_argument(
        "--label_paths",
        type=str,
        default="dataset/format/librispeech/40M_librispeech_lm_monophncode_librispeech_frame_monophncode_using_wav2vec-u2_model/train-960.speechphncode",
        help="audio pseudo label file ",
    )
    parser.add_argument(
        "--audio_tsv",
        type=str,
        default="dataset/format/librispeech/40M_librispeech_lm_monophncode_librispeech_frame_monophncode_using_wav2vec-u2_model/train-960.tsv",
        help="audio wavforms file ",
    )

    parser.add_argument(
        "--text_path",
        type=str,
        default="dataset/format/librispeech/40M_librispeech_lm_monophncode_librispeech_frame_monophncode_using_wav2vec-u2_model/train-960.textphncode",
        help="text code sequence file ",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=7,
        help="which layer is output representation in voicelm2 model",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="voicelm2 expect audio sample rate",
    )
    parser.add_argument(
        "--label_path",
        type=str,
        default="dataset/format/librispeech/feat_dir/voicelm2_feat_of_0.1_960h_librispeech/voicelm_7layer_label_dir/dev-other.km",
        help="specify store feature directory ",
    )
    parser.add_argument(
        "--km_model_path",
        type=str,
        default="dataset/format/librispeech/feat_dir/voicelm2_feat_of_0.1_960h_librispeech/train-960_10_percent_voicelm2_7layer_km_100_clusters.mdl",
        help="specify store feature directory ",
    )
    
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="exp/pretrain/pretrain_on_base_voicelm2_4gpu_8update_960h_400k_update_flash_attention_lr4e_4_40M_unpaired_text_1to40/checkpoint_best.pt",
        help="voicelm2 model path ",
    )
    return parser

if __name__ == "__main__":
    parser =  get_parser()
    args = parser.parse_args()
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(
        format=formatter,
        level=logging.INFO,
    )


    dicts = [Dictionary.load(f"{args.text_dict}")]
    dicts_text = [dicts[-1]]
    text_procs = [TextEncoder(dict) for dict in dicts_text]


    label_paths = [args.label_paths]
    manifest_path = args.audio_tsv
    manifest_text_path = args.text_path

    reader = Voicelm2FeatureReader(
        args.sample_rate,
        args.ckpt_path,
        args.layer,
        label_paths,
        manifest_path,
        manifest_text_path,
        text_procs,
    )
    ## dump pseudo label 
    #from tqdm import tqdm
    logging.info(f"args.km_model_path: {args.km_model_path}")
    apply_kmeans = ApplyKmeans(args.km_model_path)
    logging.info(f"kmeans model: {apply_kmeans}")
    utts = reader.sizes() ## int
    with open(args.label_path,'w')as f:
      
        for i in tqdm(range(utts),desc="pesudo label"):
            feat = reader.get_feats(i)
            p_lab = apply_kmeans(feat).tolist()
            f.write(" ".join(map(str,p_lab)) + "\n")
        
    logging.info(f"Dump pseudo label successfully, it is at {args.label_path}") 
