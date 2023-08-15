#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Run inference for pre-processed data with a trained model.
"""

import ast
from collections import namedtuple
from dataclasses import dataclass, field
from enum import Enum, auto
import hydra
from hydra.core.config_store import ConfigStore
import logging
import math
import os
from omegaconf import OmegaConf
from typing import Optional
import sys

import editdistance
import torch

from hydra.core.hydra_config import HydraConfig

from fairseq import checkpoint_utils, progress_bar, tasks, utils
from fairseq.data.data_utils import post_process
from fairseq.dataclass.configs import FairseqDataclass, FairseqConfig
from fairseq.logging.meters import StopwatchMeter
from omegaconf import open_dict

from examples.speech_recognition.kaldi.kaldi_decoder import KaldiDecoderConfig

logging.root.setLevel(logging.INFO)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


class DecoderType(Enum):
    VITERBI = auto()
    VITERBI_FRAME = auto()
    KENLM = auto()
    FAIRSEQ = auto()
    KALDI = auto()


@dataclass
class UnsupGenerateConfig(FairseqDataclass):
    fairseq: FairseqConfig = FairseqConfig()
    lm_weight: float = field(
        default=2.0,
        metadata={"help": "language model weight"},
    )
    w2l_decoder: DecoderType = field(
        default=DecoderType.VITERBI,
        metadata={"help": "type of decoder to use"},
    )
    kaldi_decoder_config: Optional[KaldiDecoderConfig] = None
    lexicon: Optional[str] = field(
        default=None,
        metadata={
            "help": "path to lexicon. This is also used to 'phonemize' for unsupvised param tuning"
        },
    )
    lm_model: Optional[str] = field(
        default=None,
        metadata={"help": "path to language model (kenlm or fairseq)"},
    )
    decode_stride: Optional[int] = field(
        default=None,
        metadata={"help": "changing the decoding frequency of the generator"},
    )
    unit_lm: bool = field(
        default=False,
        metadata={"help": "whether to use unit lm"},
    )
    beam_threshold: float = field(
        default=50.0,
        metadata={"help": "beam score threshold"},
    )
    beam_size_token: float = field(
        default=100.0,
        metadata={"help": "max tokens per beam"},
    )
    beam: int = field(
        default=5,
        metadata={"help": "decoder beam size"},
    )
    nbest: int = field(
        default=1,
        metadata={"help": "number of results to return"},
    )
    word_score: float = field(
        default=1.0,
        metadata={"help": "word score to add at end of word"},
    )
    unk_weight: float = field(
        default=-math.inf,
        metadata={"help": "unknown token weight"},
    )
    sil_weight: float = field(
        default=0.0,
        metadata={"help": "silence token weight"},
    )
    targets: Optional[str] = field(
        default=None,
        metadata={"help": "extension of ground truth labels to compute UER"},
    )
    results_path: Optional[str] = field(
        default=None,
        metadata={"help": "where to store results"},
    )
    post_process: Optional[str] = field(
        default=None,
        metadata={"help": "how to post process results"},
    )
    vocab_usage_power: float = field(
        default=2,
        metadata={"help": "for unsupervised param tuning"},
    )

    viterbi_transcript: Optional[str] = field(
        default=None,
        metadata={"help": "for unsupervised param tuning"},
    )
    min_lm_ppl: float = field(
        default=0,
        metadata={"help": "for unsupervised param tuning"},
    )
    min_vt_uer: float = field(
        default=0,
        metadata={"help": "for unsupervised param tuning"},
    )

    blank_weight: float = field(
        default=0,
        metadata={"help": "value to add or set for blank emission"},
    )
    blank_mode: str = field(
        default="set",
        metadata={
            "help": "can be add or set, how to modify blank emission with blank weight"
        },
    )
    sil_is_blank: bool = field(
        default=False,
        metadata={"help": "if true, <SIL> token is same as blank token"},
    )

    unsupervised_tuning: bool = field(
        default=False,
        metadata={
            "help": "if true, returns a score based on unsupervised param selection metric instead of UER"
        },
    )
    is_ax: bool = field(
        default=False,
        metadata={
            "help": "if true, assumes we are using ax for tuning and returns a tuple for ax to consume"
        },
    )


def get_dataset_itr(cfg, task):
    return task.get_batch_iterator(
        dataset=task.dataset(cfg.fairseq.dataset.gen_subset),
        max_tokens=cfg.fairseq.dataset.max_tokens,
        max_sentences=cfg.fairseq.dataset.batch_size,
        max_positions=(sys.maxsize, sys.maxsize),
        ignore_invalid_inputs=cfg.fairseq.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.fairseq.dataset.required_batch_size_multiple,
        num_shards=cfg.fairseq.dataset.num_shards,
        shard_id=cfg.fairseq.dataset.shard_id,
        num_workers=cfg.fairseq.dataset.num_workers,
        data_buffer_size=cfg.fairseq.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)


def process_predictions(
    cfg: UnsupGenerateConfig,
    hypos,
    tgt_dict,
    target_tokens,
    res_files,
):
    retval = []
    word_preds = []
    transcriptions = []
    dec_scores = []

    for i, hypo in enumerate(hypos[: min(len(hypos), cfg.nbest)]):
        if torch.is_tensor(hypo["tokens"]):
            tokens = hypo["tokens"].int().cpu()
            tokens = tokens[tokens >= tgt_dict.nspecial]
            hyp_pieces = tgt_dict.string(tokens)
        else:
            hyp_pieces = " ".join(hypo["tokens"])

        if "words" in hypo and len(hypo["words"]) > 0:
            hyp_words = " ".join(hypo["words"])
        else:
            hyp_words = post_process(hyp_pieces, cfg.post_process)

        to_write = {}
        if res_files is not None:
            to_write[res_files["hypo.units"]] = hyp_pieces
            to_write[res_files["hypo.words"]] = hyp_words

        tgt_words = ""
        if target_tokens is not None:
            if isinstance(target_tokens, str):
                tgt_pieces = tgt_words = target_tokens
            else:
                tgt_pieces = tgt_dict.string(target_tokens)
                tgt_words = post_process(tgt_pieces, cfg.post_process)

            if res_files is not None:
                to_write[res_files["ref.units"]] = tgt_pieces
                to_write[res_files["ref.words"]] = tgt_words

        if not cfg.fairseq.common_eval.quiet:
            logger.info(f"HYPO {i}:" + hyp_words)
            if tgt_words:
                logger.info("TARGET:" + tgt_words)

            if "am_score" in hypo and "lm_score" in hypo:
                logger.info(
                    f"DECODER AM SCORE: {hypo['am_score']}, DECODER LM SCORE: {hypo['lm_score']}, DECODER SCORE: {hypo['score']}"
                )
            elif "score" in hypo:
                logger.info(f"DECODER SCORE: {hypo['score']}")

            logger.info("___________________")

        hyp_words_arr = hyp_words.split()
        tgt_words_arr = tgt_words.split()

        retval.append(
            (
                editdistance.eval(hyp_words_arr, tgt_words_arr),
                len(hyp_words_arr),
                len(tgt_words_arr),
                hyp_pieces,
                hyp_words,
            )
        )
        word_preds.append(hyp_words_arr)
        transcriptions.append(to_write)
        dec_scores.append(-hypo.get("score", 0))  # negate cuz kaldi returns NLL

    if len(retval) > 1:
        best = None
        for r, t in zip(retval, transcriptions):
            if best is None or r[0] < best[0][0]:
                best = r, t
        for dest, tran in best[1].items():
            print(tran, file=dest)
            dest.flush()
        return best[0]

    assert len(transcriptions) == 1
    for dest, tran in transcriptions[0].items():
        print(tran, file=dest)

    return retval[0]


def prepare_result_files(cfg: UnsupGenerateConfig):
    def get_res_file(file_prefix):
        if cfg.fairseq.dataset.num_shards > 1:
            file_prefix = f"{cfg.fairseq.dataset.shard_id}_{file_prefix}"
        path = os.path.join(
            cfg.results_path,
            "{}{}.txt".format(
                cfg.fairseq.dataset.gen_subset,
                file_prefix,
            ),
        )
        return open(path, "w", buffering=1)

    if not cfg.results_path:
        return None

    return {
        "hypo.words": get_res_file(""),
        "hypo.units": get_res_file("_units"),
        "ref.words": get_res_file("_ref"),
        "ref.units": get_res_file("_ref_units"),
        "hypo.nbest.words": get_res_file("_nbest_words"),
    }


def optimize_models(cfg: UnsupGenerateConfig, use_cuda, models):
    """Optimize ensemble for generation"""
    for model in models:
        model.eval()
        if cfg.fairseq.common.fp16:
            model.half()
        if use_cuda:
            model.cuda()


GenResult = namedtuple(
    "GenResult",
    [
        "count",
        "errs_t",
        "gen_timer",
        "lengths_hyp_unit_t",
        "lengths_hyp_t",
        "lengths_t",
        "lm_score_t",
        "num_feats",
        "num_sentences",
        "num_symbols",
        "vt_err_t",
        "vt_length_t",
    ],
)


def generate(cfg: UnsupGenerateConfig, models, saved_cfg, use_cuda):
    task = tasks.setup_task(cfg.fairseq.task)
    saved_cfg.task.labels = cfg.fairseq.task.labels
    task.load_dataset(cfg.fairseq.dataset.gen_subset, task_cfg=saved_cfg.task)
    # Set dictionary
    tgt_dict = task.target_dictionary
    logger.info(
        "| {} {} {} examples".format(
            cfg.fairseq.task.data,
            cfg.fairseq.dataset.gen_subset,
            len(task.dataset(cfg.fairseq.dataset.gen_subset)),
        )
    )
    # Load dataset (possibly sharded)
    itr = get_dataset_itr(cfg, task)
    logger.info(f"itr length : {len(itr)}")
    # Initialize generator
    gen_timer = StopwatchMeter()

    def build_generator(cfg: UnsupGenerateConfig):
        w2l_decoder = cfg.w2l_decoder
        if w2l_decoder == DecoderType.VITERBI:
            from examples.speech_recognition.w2l_decoder import W2lViterbiDecoder

            return W2lViterbiDecoder(cfg, task.target_dictionary)

        elif w2l_decoder == DecoderType.VITERBI_FRAME:
            from examples.speech_recognition.w2l_decoder import W2lViterbiFrameDecoder

            return W2lViterbiFrameDecoder(cfg, task.target_dictionary)
        elif w2l_decoder == DecoderType.KENLM:
            from examples.speech_recognition.w2l_decoder import W2lKenLMDecoder

            return W2lKenLMDecoder(cfg, task.target_dictionary)
        elif w2l_decoder == DecoderType.FAIRSEQ:
            from examples.speech_recognition.w2l_decoder import W2lFairseqLMDecoder

            return W2lFairseqLMDecoder(cfg, task.target_dictionary)
        elif w2l_decoder == DecoderType.KALDI:
            from examples.speech_recognition.kaldi.kaldi_decoder import KaldiDecoder

            assert cfg.kaldi_decoder_config is not None

            return KaldiDecoder(
                cfg.kaldi_decoder_config,
                cfg.beam,
            )
        else:
            raise NotImplementedError(
                "only wav2letter decoders with (viterbi, kenlm, fairseqlm) options are supported at the moment but found "
                + str(w2l_decoder)
            )

    generator = build_generator(cfg)

    kenlm = None
    fairseq_lm = None
    if cfg.lm_model is not None:
        import kenlm

        kenlm = kenlm.Model(cfg.lm_model)

    num_sentences = 0
    if cfg.results_path is not None and not os.path.exists(cfg.results_path):
        os.makedirs(cfg.results_path)

    res_files = prepare_result_files(cfg)
    errs_t = 0
    lengths_hyp_t = 0
    lengths_hyp_unit_t = 0
    lengths_t = 0
    count = 0
    num_feats = 0
    all_hyp_pieces = []
    all_hyp_words = []

    num_symbols = (
        len([s for s in tgt_dict.symbols if not s.startswith("madeup")])
        - tgt_dict.nspecial
    )
    logger.info(f"num_symbols: {num_symbols}")
    logger.info(f"nspecial of dict: {tgt_dict.nspecial}")
    logger.info(f"tgt_dict.symbols: {[s for s in tgt_dict.symbols]}")

    targets = None
    if cfg.targets is not None:
        tgt_path = os.path.join(
            cfg.fairseq.task.data, cfg.fairseq.dataset.gen_subset + "." + cfg.targets
        )
        if os.path.exists(tgt_path):
            with open(tgt_path, "r") as f:
                targets = f.read().splitlines()
    viterbi_transcript = None
    if cfg.viterbi_transcript is not None and len(cfg.viterbi_transcript) > 0:
        logger.info(f"loading viterbi transcript from {cfg.viterbi_transcript}")
        with open(cfg.viterbi_transcript, "r") as vf:
            viterbi_transcript = vf.readlines()
            viterbi_transcript = [v.rstrip().split() for v in viterbi_transcript]

    gen_timer.start()

    start = 0
    end = len(itr)+1
    print(f"end: {end}")
    #logger.info(f"Finish extracting features")
     
    hypo_futures = None
    if cfg.w2l_decoder == DecoderType.KALDI:
        logger.info("Extracting features")
        hypo_futures = []
        samples = []
        with progress_bar.build_progress_bar(cfg.fairseq.common, itr) as t:
            for i, sample in enumerate(t):
                if "net_input" not in sample or i < start or i >= end:
                    continue
                if "padding_mask" not in sample["net_input"]:
                    sample["net_input"]["padding_mask"] = None

                hypos, num_feats = gen_hypos(
                    generator, models, num_feats, sample, task, use_cuda
                )
                hypo_futures.append(hypos)
                samples.append(sample)
        itr = list(zip(hypo_futures, samples))
        start = 0
        end = len(itr)
        logger.info("Finished extracting features")

    logger.info(f"Finish extracting features")

    #phone_id_code_file= cfg.fairseq.task.data + '/' + cfg.fairseq.dataset.gen_subset + ".unsupphncode"    
    phone_id_code_file= cfg.results_path + '/' + cfg.fairseq.dataset.gen_subset + ".unsupphncode"
    with progress_bar.build_progress_bar(cfg.fairseq.common, itr) as t, open(phone_id_code_file,'w')as fw:
        for i, sample in enumerate(t):
            ## import data information of display
            ##logger.info(f"sample: {sample}")
            ##logger.info(f"sample padding mask shape: {sample['net_input']['padding_mask'].shape}")
            logger.info(f"sample: {i+1}, sample feature shape : {sample['net_input']['features'].shape}")
            frame_num=sample['net_input']['features'].size(1)
            logger.info(f"sample: {i+1}, speech frame number: {frame_num}")
            if i < start or i >= end:
                continue

            if hypo_futures is not None:
                hypos, sample = sample
                hypos = [h.result() for h in hypos]
            else:
                if "net_input" not in sample:
                    continue

                hypos, num_feats = gen_hypos(
                    generator, models, num_feats, sample, task, use_cuda
                )
                logger.info(f"sample: {i+1} predict hypos: hypos[0][0]['tokens'] shape: {hypos[0][0]['tokens'].shape}")
                hyp = hypos[0][0]['tokens'].tolist() ## list, 
                                                     ## its per element  is a id,
                                                     ## after it should reduce 4(because it 
                                                     ## adds four specify symbol),
                                                     ## its id  is equal to monophone sets.
                hyp = [int(s)-4 for s in hyp]  ## reduce 4
                hyp_id = ' '.join(map(str,hyp)) ## 
                fw.write(hyp_id+'\n')
              
def gen_hypos(generator, models, num_feats, sample, task, use_cuda):
    sample = utils.move_to_cuda(sample) if use_cuda else sample

    if "features" in sample["net_input"]:
        sample["net_input"]["dense_x_only"] = True
        num_feats += (
            sample["net_input"]["features"].shape[0]
            * sample["net_input"]["features"].shape[1]
        )
    hypos = task.inference_step(generator, models, sample, None)
    return hypos, num_feats


def main(cfg: UnsupGenerateConfig, model=None):
    if (
        cfg.fairseq.dataset.max_tokens is None
        and cfg.fairseq.dataset.batch_size is None
    ):
        cfg.fairseq.dataset.max_tokens = 1024000

    use_cuda = torch.cuda.is_available() and not cfg.fairseq.common.cpu

    task = tasks.setup_task(cfg.fairseq.task)

    overrides = ast.literal_eval(cfg.fairseq.common_eval.model_overrides)

    if cfg.fairseq.task._name == "unpaired_audio_text":
        overrides["model"] = {
            "blank_weight": cfg.blank_weight,
            "blank_mode": cfg.blank_mode,
            "blank_is_sil": cfg.sil_is_blank,
            "no_softmax": True,
            "segmentation": {
                "type": "NONE",
            },
        }
    else:
        overrides["model"] = {
            "blank_weight": cfg.blank_weight,
            "blank_mode": cfg.blank_mode,
        }
    
    if cfg.decode_stride:
        overrides["model"]["generator_stride"] = cfg.decode_stride

    if model is None:
        # Load ensemble
        logger.info("| loading model(s) from {}".format(cfg.fairseq.common_eval.path))
        models, saved_cfg = checkpoint_utils.load_model_ensemble(
            cfg.fairseq.common_eval.path.split("\\"),
            arg_overrides=overrides,
            task=task,
            suffix=cfg.fairseq.checkpoint.checkpoint_suffix,
            strict=(cfg.fairseq.checkpoint.checkpoint_shard_count == 1),
            num_shards=cfg.fairseq.checkpoint.checkpoint_shard_count,
        )
        optimize_models(cfg, use_cuda, models)
    else:
        models = [model]
        saved_cfg = cfg.fairseq

    with open_dict(saved_cfg.task):
        saved_cfg.task.shuffle = False
        saved_cfg.task.sort_by_length = False

    generate(cfg, models, saved_cfg, use_cuda)

@hydra.main(
    config_path=os.path.join("../../..", "fairseq", "config"), config_name="config"
)
def hydra_main(cfg):
    with open_dict(cfg):
        # make hydra logging work with ddp (see # see https://github.com/facebookresearch/hydra/issues/1126)
        cfg.job_logging_cfg = OmegaConf.to_container(
            HydraConfig.get().job_logging, resolve=True
        )

    cfg = OmegaConf.create(
        OmegaConf.to_container(cfg, resolve=False, enum_to_str=False)
    )
    OmegaConf.set_struct(cfg, True)
    logger.info(cfg)

    utils.import_user_module(cfg.fairseq.common)

    main(cfg)    


def cli_main():
    try:
        from hydra._internal.utils import get_args

        cfg_name = get_args().config_name or "config"
    except:
        logger.warning("Failed to get config name from hydra args")
        cfg_name = "config"

    cs = ConfigStore.instance()
    cs.store(name=cfg_name, node=UnsupGenerateConfig)
    hydra_main()


if __name__ == "__main__":
    cli_main()
