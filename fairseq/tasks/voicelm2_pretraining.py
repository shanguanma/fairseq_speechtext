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
from fairseq.data import Dictionary, Voicelm2Dataset, HubertDataset, encoders
from fairseq.dataclass.configs import FairseqDataclass
from fairseq.tasks import register_task
from fairseq.tasks.fairseq_task import FairseqTask
from omegaconf import MISSING

logger = logging.getLogger(__name__)


#from fairseq.models.hubert.voicelm2_sequence_generator import SequenceGenerator, SequenceGeneratorWithAlignment
from fairseq import metrics, search
from omegaconf import II
from argparse import Namespace

## it will use letter unit to finetune via ctc loss
## or it will use code unit to pretrain via mlm loss.
class LabelEncoder(object):
    def __init__(self, dictionary: Dictionary) -> None:
        self.dictionary = dictionary

    def __call__(self, label: str) -> List[str]:
        return self.dictionary.encode_line(
            label,
            append_eos=False,
            add_if_not_exist=False,
        )


## it will use bpe unit to finetune via seq2seq cross entropy loss.
class LabelEncoderS2SToken(object):
    def __init__(self, dictionary: Dictionary, bpe_tokenizer) -> None:
        self.bpe_tokenizer = bpe_tokenizer
        self.dictionary = dictionary

    def __call__(self, label: str) -> List[str]:
        label = self.bpe_tokenizer.encode(label.lower())
        return self.dictionary.encode_line(
            label,
            append_eos=True,
            add_if_not_exist=False,
        ).long()

    def decode(self, tok, symbols_ignore=None):
        tok = self.dictionary.string(tok, extra_symbols_to_ignore=symbols_ignore)
        if self.bpe_tokenizer:
            tok = self.bpe_tokenizer.decode(tok)
        return tok


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
class Voicelm2PretrainingConfig(FairseqDataclass):
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
    # max_keep_size: Optional[int] = field(
    #    default=None,
    #    metadata={"help": "exclude sample longer than this"},
    # )
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
    ###  unpaired text part
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
        metadata={
            "help": "## if it is true,  it will used colletor_seq_text independent audio, otherwise, will colletor_frm_text"
        },
    )

    is_s2s: bool = field(
        default=False,
        metadata={"help": "if true, seq2seq fine-tuning only, else ctc finetune only"},
    )
    tokenizer_bpe_name: Optional[str] = field(
        default=None, metadata={"help": "tokenizer model name"}
    )
    tokenizer_bpe_model: Optional[str] = field(
        default=None, metadata={"help": "tokenizer model path"}
    )
    text_drop: bool = field(
        default=False,
        metadata={
            "help": """if it is true, speech and paired text are used to finetune model, unpair text is missing.
                                                    if it is false, speech and paired code label and unpair text code are used to pretrain model."""
        },
    )
    inference_mode: bool = field(default=False, metadata={"help": "it is diffence from finetune mode, because here the finetune model can accept two style label. inference_model=true, it will only accept one style label"})


@register_task("voicelm2_pretraining", dataclass=Voicelm2PretrainingConfig)
class Voicelm2PretrainingTask(FairseqTask):
    cfg: Voicelm2PretrainingConfig

    def __init__(
        self,
        cfg: Voicelm2PretrainingConfig,
    ) -> None:
        super().__init__(cfg)

        logger.info(f"current directory is {os.getcwd()}")
        logger.info(f"Voicelm2PretrainingTask Config {cfg}")

        self.cfg = cfg
        self.fine_tuning = cfg.fine_tuning

        if cfg.fine_tuning:
            self.state.add_factory("target_dictionary", self.load_dictionaries)
            if cfg.is_s2s:
                self.state.add_factory("s2s_tokenizer", self.load_tokenizer)
        else:
            self.state.add_factory("dictionaries", self.load_dictionaries)
            # self.state.add_factory("text_dictionary",self.load_text_dictionaries)

        self.blank_symbol = "<s>"

    @property
    def source_dictionary(self) -> Optional[Dictionary]:
        # return self.state.text_dictionary
        return None

    @property
    def target_dictionary(self) -> Optional[Dictionary]: ## it is used at fairseq_criterion.py 
        return self.state.target_dictionary

    @property
    def dictionaries(self) -> List[Dictionary]:
        # dict_list=[self.state.dictionaries,self.state.text_dictionary]
        # return dict_list
        #return self.state.dictionaries[0] if self.cfg.fine_tuning else self.state.dictionaries 
        return self.state.dictionaries

    def load_dictionaries(self):
        label_dir = self.cfg.data if self.cfg.label_dir is None else self.cfg.label_dir
        dictionaries = [
            Dictionary.load(f"{label_dir}/dict.{label}.txt")
            for label in self.cfg.labels
        ]
        return dictionaries[0] if self.cfg.fine_tuning else dictionaries
        #return dictionaries

    def load_tokenizer(self):
        bpe_args = Namespace(
            **{
                "bpe": self.cfg.tokenizer_bpe_name,
                f"{self.cfg.tokenizer_bpe_name}_model": self.cfg.tokenizer_bpe_model,
            }
        )
        bpe_tokenizer = encoders.build_bpe(bpe_args)
        return bpe_tokenizer

    @property
    def s2s_tokenizer(self):
        return self.state.s2s_tokenizer

    @classmethod
    def setup_task(
        cls, cfg: Voicelm2PretrainingConfig, **kwargs
    ) -> "Voicelm2PretrainingTask":
        return cls(cfg)

    def get_label_dir(self) -> str:
        if self.cfg.label_dir is None:
            return self.cfg.data
        return self.cfg.label_dir

    def load_dataset(self, split: str, **kwargs) -> None:
        """
        manifest = f"{self.cfg.data}/{split}.tsv"
        if self.cfg.fine_tuning:
            dicts = [
                Dictionary.load(f"{self.cfg.label_dir}/dict.{label}.txt")
                for label in self.cfg.labels
            ]
        else:
            dicts=self.dictionaries
        #dicts = [self.target_dictionary] if self.cfg.fine_tuning else self.dictionaries

    
        logger.info(f"dicts: {dicts}")
        if len(dicts)==2: ##  finetune mode
            dicts_speech_label = [dicts[0]]  # remove text phn dictionary
            dicts_text = [dicts[1]]  
        else:
            dicts_speech_label = [dicts[0]]
            dicts_text = dicts_speech_label # unparied text drop
        #dicts_text = dicts_speech_label 
        pad_list = [dict.pad() for dict in dicts_speech_label]
        eos_list = [dict.eos() for dict in dicts_speech_label]
        # procs = [LabelEncoder(dict) for dict in dicts]
        if not self.cfg.is_s2s:
            procs = [LabelEncoder(dict) for dict in dicts_speech_label]
        else:
            logger.info(f"Using tokenizer")
            bpe_tokenizer = self.s2s_tokenizer
            procs = [
                LabelEncoderS2SToken(dict, bpe_tokenizer) for dict in dicts_speech_label
            ]

        text_procs = [TextEncoder(dict) for dict in dicts_text]
        paths = [f"{self.get_label_dir()}/{split}.{l}" for l in self.cfg.labels]
        if len(paths)==2:
            path_text=paths[1]
            path_label=paths[0]
        elif 
        logger.info(f"paths: {paths}")
        """
        ### prepared pretrain mode and finetune mode and inference mode
        ## I hope both pretrain and finetuen mode  can accept two or more two style label, and inference mode cant accept one style label
        manifest = f"{self.cfg.data}/{split}.tsv"
        speech_procs=None
        if not self.cfg.inference_mode: ## finetune mode and pretrain mode
            ## dict
            ## data path
            ## data proccess
            dicts = [
                Dictionary.load(f"{self.cfg.label_dir}/dict.{label}.txt")
                for label in self.cfg.labels
            ]
            dicts_speech_label = [dicts[0]] if len(dicts)==2 else  dicts[:-1]  # remove text phn dictionary
            dicts_text = [dicts[-1]]
            pad_list = [dict.pad() for dict in dicts_speech_label]
            eos_list = [dict.eos() for dict in dicts_speech_label]
            if not self.cfg.is_s2s:
                speech_procs = [LabelEncoder(dict) for dict in dicts_speech_label]
            else:
                logger.info(f"Using tokenizer")
                bpe_tokenizer = self.s2s_tokenizer
                speech_procs = [
                    LabelEncoderS2SToken(dict, bpe_tokenizer) for dict in dicts_speech_label
                ]
            text_procs = [TextEncoder(dict) for dict in dicts_text]
            paths = [f"{self.get_label_dir()}/{split}.{l}" for l in self.cfg.labels]
            path_text=paths[-1]
            path_label=paths[:-1]
        else: ## inference mode
            ## dict
            ## data path
            ## data proccess
            dicts = [
                Dictionary.load(f"{self.cfg.label_dir}/dict.{label}.txt")
                for label in self.cfg.labels
            ]
            dicts_speech_label = [dicts[0]]  # remove text phn dictionary
            dicts_text = None
            pad_list = [dict.pad() for dict in dicts_speech_label]
            eos_list = [dict.eos() for dict in dicts_speech_label]
            if not self.cfg.is_s2s:
                speech_procs = [LabelEncoder(dict) for dict in dicts_speech_label]
            else:
                logger.info(f"Using tokenizer")
                bpe_tokenizer = self.s2s_tokenizer
                speech_procs = [
                    LabelEncoderS2SToken(dict, bpe_tokenizer) for dict in dicts_speech_label
                ]
            text_procs = None
            paths = [f"{self.get_label_dir()}/{split}.{l}" for l in self.cfg.labels]
            path_text=None
            path_label=paths[0]         
  

        # hubert v1: pad_audio=True, random_crop=False;
        if self.cfg.fine_tuning:
            if (
                self.cfg.text_drop
            ):  ## normal fintune case(actual using speech and pair text to finetune, unpair text feature is setting 0 )
                self.datasets[split] = Voicelm2Dataset(
                    manifest,
                    manifest_text_path=path_text,
                    sample_rate=self.cfg.sample_rate,
                    label_paths=path_label,
                    label_rates=self.cfg.label_rate,
                    pad_list=pad_list,
                    eos_list=eos_list,
                    text_seq=self.cfg.text_seq,
                    label_processors=speech_procs,
                    text_processors=text_procs,
                    max_keep_sample_size=self.cfg.max_sample_size,
                    min_keep_sample_size=self.cfg.min_sample_size,
                    max_sample_size=self.cfg.max_sample_size,
                    max_keep_phone_size=self.cfg.max_phone_size,
                    min_keep_phone_size=self.cfg.min_phone_size,
                    pad_audio=self.cfg.pad_audio,
                    normalize=self.cfg.normalize,
                    store_labels=False,
                    random_crop=self.cfg.random_crop,
                    single_target=self.cfg.single_target,
                    is_s2s=self.cfg.is_s2s,  ## choice ctc or seq2seq finetune flag
                    text_drop=self.cfg.text_drop,  ## whether unpaired text is used to finetune
                )
            else:  ## fintune case (in finetune case, i also use unpaired text code.)
                self.datasets[split] = Voicelm2Dataset(
                    manifest,
                    manifest_text_path=path_text,
                    sample_rate=self.cfg.sample_rate,
                    label_paths=path_label,
                    label_rates=self.cfg.label_rate,
                    pad_list=pad_list,
                    eos_list=eos_list,
                    text_seq=self.cfg.text_seq,
                    label_processors=speech_procs,
                    text_processors=text_procs,
                    max_keep_sample_size=self.cfg.max_sample_size,
                    min_keep_sample_size=self.cfg.min_sample_size,
                    max_sample_size=self.cfg.max_sample_size,
                    max_keep_phone_size=self.cfg.max_phone_size,
                    min_keep_phone_size=self.cfg.min_phone_size,
                    pad_audio=self.cfg.pad_audio,
                    normalize=self.cfg.normalize,
                    store_labels=False,
                    random_crop=self.cfg.random_crop,
                    single_target=self.cfg.single_target,
                    is_s2s=self.cfg.is_s2s,  ## choice ctc or seq2seq finetune flag
                    text_drop=self.cfg.text_drop,  ## whether unpaired text is used to finetune
                )
        else:  ## pretrain case
            self.datasets[split] = Voicelm2Dataset(
                manifest,
                manifest_text_path=path_text,
                sample_rate=self.cfg.sample_rate,
                label_paths=path_label,
                label_rates=self.cfg.label_rate,
                pad_list=pad_list,
                eos_list=eos_list,
                text_seq=self.cfg.text_seq,
                label_processors=speech_procs,
                text_processors=text_procs,
                max_keep_sample_size=self.cfg.max_sample_size,
                min_keep_sample_size=self.cfg.min_sample_size,
                max_sample_size=self.cfg.max_sample_size,
                max_keep_phone_size=self.cfg.max_phone_size,
                min_keep_phone_size=self.cfg.min_phone_size,
                pad_audio=self.cfg.pad_audio,
                normalize=self.cfg.normalize,
                store_labels=False,
                random_crop=self.cfg.random_crop,
                single_target=self.cfg.single_target,
                is_s2s=False,
                text_drop=False,
            )

    def max_positions(self) -> Tuple[int, int]:
        return (sys.maxsize, sys.maxsize)

    def filter_indices_by_size(self, indices: np.array, *args, **kwargs) -> np.array:
        return indices

'''
    ## it is only used to seq2seq decoding.
    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
        prefix_allowed_tokens_fn=None,
    ):
        """
        Build a :class:`~fairseq.SequenceGenerator` instance for this
        task.
        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            args (fairseq.dataclass.configs.GenerationConfig):
                configuration object (dataclass) for generation
            extra_gen_cls_kwargs (Dict[str, Any]): extra options to pass
                through to SequenceGenerator
            prefix_allowed_tokens_fn (Callable[[int, torch.Tensor], List[int]]):
                If provided, this function constrains the beam search to
                allowed tokens only at each step. The provided function
                should take 2 arguments: the batch ID (`batch_id: int`)
                and a unidimensional tensor of token ids (`inputs_ids:
                torch.Tensor`). It has to return a `List[int]` with the
                allowed tokens for the next generation step conditioned
                on the previously generated tokens (`inputs_ids`) and
                the batch ID (`batch_id`). This argument is useful for
                constrained generation conditioned on the prefix, as
                described in "Autoregressive Entity Retrieval"
                (https://arxiv.org/abs/2010.00904) and
                https://github.com/facebookresearch/GENRE.
        """
        if getattr(args, "score_reference", False):
            from fairseq.sequence_scorer import SequenceScorer

            return SequenceScorer(
                self.target_dictionary,
                compute_alignment=getattr(args, "print_alignment", False),
            )

        # Choose search strategy. Defaults to Beam Search.
        sampling = getattr(args, "sampling", False)
        sampling_topk = getattr(args, "sampling_topk", -1)
        sampling_topp = getattr(args, "sampling_topp", -1.0)
        diverse_beam_groups = getattr(args, "diverse_beam_groups", -1)
        diverse_beam_strength = getattr(args, "diverse_beam_strength", 0.5)
        match_source_len = getattr(args, "match_source_len", False)
        diversity_rate = getattr(args, "diversity_rate", -1)
        constrained = getattr(args, "constraints", False)
        if prefix_allowed_tokens_fn is None:
            prefix_allowed_tokens_fn = getattr(args, "prefix_allowed_tokens_fn", None)
        if (
            sum(
                int(cond)
                for cond in [
                    sampling,
                    diverse_beam_groups > 0,
                    match_source_len,
                    diversity_rate > 0,
                ]
            )
            > 1
        ):
            raise ValueError("Provided Search parameters are mutually exclusive.")
        assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
        assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

        if sampling:
            search_strategy = search.Sampling(
                self.target_dictionary, sampling_topk, sampling_topp
            )
        elif diverse_beam_groups > 0:
            search_strategy = search.DiverseBeamSearch(
                self.target_dictionary, diverse_beam_groups, diverse_beam_strength
            )
        elif match_source_len:
            # this is useful for tagging applications where the output
            # length should match the input length, so we hardcode the
            # length constraints for simplicity
            search_strategy = search.LengthConstrainedBeamSearch(
                self.target_dictionary,
                min_len_a=1,
                min_len_b=0,
                max_len_a=1,
                max_len_b=0,
            )
        elif diversity_rate > -1:
            search_strategy = search.DiverseSiblingsSearch(
                self.target_dictionary, diversity_rate
            )
        elif constrained:
            search_strategy = search.LexicallyConstrainedBeamSearch(
                self.target_dictionary, args.constraints
            )
        elif prefix_allowed_tokens_fn:
            search_strategy = search.PrefixConstrainedBeamSearch(
                self.target_dictionary, prefix_allowed_tokens_fn
            )
        else:
            search_strategy = search.BeamSearch(self.target_dictionary)

        extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
        if seq_gen_cls is None:
            if getattr(args, "print_alignment", False):
                seq_gen_cls = SequenceGeneratorWithAlignment
                extra_gen_cls_kwargs["print_alignment"] = args.print_alignment
            else:
                seq_gen_cls = SequenceGenerator

        return seq_gen_cls(
            models,
            self.target_dictionary,
            beam_size=getattr(args, "beam", 5),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search_strategy,
            **extra_gen_cls_kwargs,
        )
'''
