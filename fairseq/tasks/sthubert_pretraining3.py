# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from dataclasses import dataclass, field
from fairseq.data import Dictionary, StHubertDataset3, HubertDataset
from fairseq.dataclass.configs import FairseqDataclass
from fairseq.tasks import register_task
from fairseq.tasks.fairseq_task import FairseqTask
from omegaconf import MISSING

logger = logging.getLogger(__name__)


class LabelEncoder(object):
    def __init__(self, dictionary: Dictionary) -> None:
        self.dictionary = dictionary

    def __call__(self, label: str) -> List[str]:
        return self.dictionary.encode_line(
            label,
            append_eos=False,
            add_if_not_exist=False,
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

@dataclass
class StHubertPretrainingConfig3(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    fine_tuning: bool = field(
        default=False, metadata={"help": "set to true if fine-tuning Hubert"}
    )
    labels: List[str] = field(
        default_factory=lambda: ["ltr"],
        metadata={
            "help": (
                "extension of the label files to load, frame-level labels for"
                " pre-training, and sequence-level label for fine-tuning"
            )
        },
    )

    label_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "if set, looks for labels in this directory instead",
        },
    )
    label_rate: float = field(
        default=-1.0,
        metadata={"help": "label frame rate. -1.0 for sequence label"},
    )
    sample_rate: int = field(
        default=16_000,
        metadata={
            "help": "target sample rate. audio files will be up/down "
            "sampled to this rate"
        },
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "if set, normalizes input to have 0 mean and unit variance"},
    )
    enable_padding: bool = field(
        default=False,
        metadata={"help": "pad shorter samples instead of cropping"},
    )
    max_keep_size: Optional[int] = field(
        default=None,
        metadata={"help": "exclude sample longer than this"},
    )
    max_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "max sample size to crop to for batching"},
    )
    min_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "min sample size to crop to for batching"},
    )
    max_phone_size: Optional[int] = field(
        default=None,
        metadata={"help": "max phone sequeuce size to crop to for batching"},
    )
    min_phone_size: Optional[int] = field(
        default=None,
        metadata={"help": "min phone sequeuce size to crop to for batching"},
    )
   
    single_target: Optional[bool] = field(
        default=False,
        metadata={
            "help": "if set, AddTargetDatasets outputs same keys " "as AddTargetDataset"
        },
    )
    random_crop: Optional[bool] = field(
        default=True,
        metadata={"help": "always crop from the beginning if false"},
    )
    pad_audio: Optional[bool] = field(
        default=False,
        metadata={"help": "pad audio to the longest one in the batch if true"},
    )
    ### sthubert text part
    max_phone_size: Optional[int] = field(
        default=None,
        metadata={"help": "max phone sequeuce size to crop to for batching"},
    )
    min_phone_size: Optional[int] = field(
        default=None,
        metadata={"help": "min phone sequeuce size to crop to for batching"},
    )
    text_seq: Optional[bool] = field(
        default=True,
        metadata={"help": "## if it is true,  it will used colletor_seq_text independent audio, otherwise, will colletor_frm_text"},
    )


@register_task("sthubert_pretraining3", dataclass=StHubertPretrainingConfig3)
class StHubertPretrainingTask3(FairseqTask):
    cfg: StHubertPretrainingConfig3

    def __init__(
        self,
        cfg: StHubertPretrainingConfig3,
    ) -> None:
        super().__init__(cfg)

        logger.info(f"current directory is {os.getcwd()}")
        logger.info(f"HubertPretrainingTask Config {cfg}")

        self.cfg = cfg
        self.fine_tuning = cfg.fine_tuning

        if cfg.fine_tuning:
            self.state.add_factory("target_dictionary", self.load_dictionaries)
        else:
            self.state.add_factory("dictionaries", self.load_dictionaries)
            #self.state.add_factory("text_dictionary",self.load_text_dictionaries)

        self.blank_symbol = "<s>"

    @property
    def source_dictionary(self) -> Optional[Dictionary]:
        #return self.state.text_dictionary
         return None
    @property
    def target_dictionary(self) -> Optional[Dictionary]:
        return self.state.target_dictionary

    @property
    def dictionaries(self) -> List[Dictionary]:
        #dict_list=[self.state.dictionaries,self.state.text_dictionary]
        #return dict_list
        return self.state.dictionaries

    @classmethod
    def setup_task(
        cls, cfg: StHubertPretrainingConfig3, **kwargs
    ) -> "StHubertPretrainingTask3":
        return cls(cfg)

    def load_dictionaries(self):
        label_dir = self.cfg.data if self.cfg.label_dir is None else self.cfg.label_dir
        dictionaries = [
            Dictionary.load(f"{label_dir}/dict.{label}.txt")
            for label in self.cfg.labels
        ]
        return dictionaries[0] if self.cfg.fine_tuning else dictionaries
    #def load_text_dictionaries(self):
    #    label_dir = self.cfg.data if self.cfg.label_dir is None else self.cfg.label_dir
    #    dictionaries = [
    #        Dictionary.load(f"{label_dir}/dict.{label}.txt")
    #        for label in self.cfg.labels
    #    ]
    #    return dictionaries[1] 

    def get_label_dir(self) -> str:
        if self.cfg.label_dir is None:
            return self.cfg.data
        return self.cfg.label_dir

    def load_dataset(self, split: str, **kwargs) -> None:
        manifest = f"{self.cfg.data}/{split}.tsv"
        dicts = [self.target_dictionary] if self.cfg.fine_tuning else self.dictionaries
        logger.info(f"dicts: {dicts}") # dicts[0] is kmeans dict dicts[1] is phn code dictionary
        dicts_km = [dicts[0]] # remove text phn dictionary
        dicts_phncode = [dicts[1]]
        #ori_dicts = [dicts[0][0],dicts[1]]
        #dicts=ori_dicts
        #for dict1 in dicts:
            
        #    logger.info(f"dict1: {dict1[0]}")
        #    logger.info(f"dict1.pad(): {dict1[0].pad()}")
        #    logger.info(f"dict1: {dict1[1]}")
        pad_list = [dict.pad() for dict in dicts_km ]
        eos_list = [dict.eos() for dict in dicts_km]
        procs = [LabelEncoder(dict) for dict in dicts_km]
        text_procs = [ TextEncoder(dict) for dict in dicts_phncode]
        paths = [f"{self.get_label_dir()}/{split}.{l}" for l in self.cfg.labels]
        logger.info(f"paths: {paths}")
        # text_paths=[f"{self.get_label_dir()}/{split}.{l}" for l in self.cfg.texts_type]
        # hubert v1: pad_audio=True, random_crop=False;
        if len(paths) !=2: ## fintune case
            self.datasets[split] = HubertDataset(
                manifest,
                sample_rate=self.cfg.sample_rate,
                label_paths=paths,
                label_rates=self.cfg.label_rate,
                pad_list=pad_list,
                eos_list=eos_list,
                label_processors=procs,
                max_keep_sample_size=self.cfg.max_keep_size,
                min_keep_sample_size=self.cfg.min_sample_size,
                max_sample_size=self.cfg.max_sample_size,
                pad_audio=self.cfg.pad_audio,
                normalize=self.cfg.normalize,
                store_labels=False,
                random_crop=self.cfg.random_crop,
                single_target=self.cfg.single_target,
            )
        else:
            self.datasets[split] = StHubertDataset3(
                manifest, 
                manifest_text_path=paths[1],
                sample_rate=self.cfg.sample_rate,
                label_paths=[paths[0]],
                label_rates=self.cfg.label_rate,
                pad_list=pad_list,
                eos_list=eos_list,
                text_seq=self.cfg.text_seq,
                label_processors=procs,
                text_processors=text_procs,
                max_keep_sample_size=self.cfg.max_keep_size,
                min_keep_sample_size=self.cfg.min_sample_size,
                max_sample_size=self.cfg.max_sample_size,
                max_keep_phone_size=self.cfg.max_phone_size,
                min_keep_phone_size=self.cfg.min_phone_size,
                pad_audio=self.cfg.pad_audio,
                normalize=self.cfg.normalize,
                store_labels=False,
                random_crop=self.cfg.random_crop,
                single_target=self.cfg.single_target,
            )

    def max_positions(self) -> Tuple[int, int]:
        return (sys.maxsize, sys.maxsize)

    def filter_indices_by_size(self, indices: np.array, *args, **kwargs) -> np.array:
        return indices
