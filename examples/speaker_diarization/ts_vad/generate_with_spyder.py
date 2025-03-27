#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""

import ast
import logging
import subprocess
import os
import sys
from argparse import Namespace
from collections import defaultdict
from scipy import signal
from tqdm import tqdm

import numpy as np
import torch
from omegaconf import DictConfig

from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter


# remove the short silence
def change_zeros_to_ones(inputs, min_silence, threshold, frame_len):
    res = []
    num_0 = 0
    thr = int(min_silence // frame_len)
    for i in inputs:
        if i >= threshold:
            if num_0 != 0:
                if num_0 > thr:
                    res.extend([0] * num_0)
                else:
                    res.extend([1] * num_0)
                num_0 = 0
            res.extend([1])
        else:
            num_0 += 1
    if num_0 > thr:
        res.extend([0] * num_0)
    else:
        res.extend([1] * num_0)
    return res


# Combine the short speech segments
def change_ones_to_zeros(inputs, min_speech, threshold, frame_len):
    res = []
    num_1 = 0
    thr = int(min_speech // frame_len)
    for i in inputs:
        if i < threshold:
            if num_1 != 0:
                if num_1 > thr:
                    res.extend([1] * num_1)
                else:
                    res.extend([0] * num_1)
                num_1 = 0
            res.extend([0])
        else:
            num_1 += 1
    if num_1 > thr:
        res.extend([1] * num_1)
    else:
        res.extend([0] * num_1)
    return res


def main(cfg: DictConfig):

    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    assert cfg.common_eval.path is not None, "--path required for generation!"
    assert (
        not cfg.generation.sampling or cfg.generation.nbest == cfg.generation.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        cfg.generation.replace_unk is None or cfg.dataset.dataset_impl == "raw"
    ), "--replace-unk requires a raw text dataset (--dataset-impl=raw)"

    if cfg.common_eval.results_path is not None:
        os.makedirs(cfg.common_eval.results_path, exist_ok=True)
        output_path = os.path.join(
            cfg.common_eval.results_path,
            "generate-{}.txt".format(cfg.dataset.gen_subset),
        )
        with open(output_path, "w", buffering=1, encoding="utf-8") as h:
            return _main(cfg, h)
    else:
        return _main(cfg, sys.stdout)


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos}


def _main(cfg: DictConfig, output_file):
    logging.basicConfig(
        format="%(asctime)s | (%(module)s:%(lineno)d)| %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )
    logger = logging.getLogger("fairseq_cli.generate")
    utils.import_user_module(cfg.common)

    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.max_tokens = 12000
    logger.info(cfg)

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    # Load dataset splits
    task = tasks.setup_task(cfg.task)

    overrides = ast.literal_eval(cfg.common_eval.model_overrides)

    # Load ensemble
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, saved_cfg = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path),
        arg_overrides=overrides,
        task=task,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
    )

    # Optimize ensemble for generation
    for model in models:
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # loading the dataset should happen after the checkpoint has been loaded so we can give it the saved task config
    task.load_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg.task)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(cfg.dataset.gen_subset),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), *[m.max_positions() for m in models]
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )

    # Initialize generator
    gen_timer = StopwatchMeter()

    assert cfg.common_eval.results_path is not None
    eval_dir = f"{cfg.common_eval.results_path}/{cfg.dataset.gen_subset}"
    os.makedirs(eval_dir, exist_ok=True)
    der_write = open(f"{eval_dir}/der_result", "a")
    rttm_path = eval_dir + "/res_rttm"
    rttms = {}
    for threshold in [0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8]:
        rttms[threshold] = open(f"{rttm_path}_{threshold}", "w")
    DER = []
    ACC = []
    res_dict_all = defaultdict(lambda: defaultdict(list))
    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if "net_input" not in sample:
            continue

        gen_timer.start()
        result, res_dict = task.inference_step(
            models,
            sample,
        )

        DER.append(result["DER"])
        ACC.append(result["ACC"])

        for filename in res_dict:
            for time_step in res_dict[filename]:
                res_dict_all[filename][time_step].extend(res_dict[filename][time_step])

    print("Model DER: ", sum(DER) / len(DER))
    print("Model ACC: ", sum(ACC) / len(ACC))

    if task.cfg.embed_input:
        frame_len = task.cfg.embed_shift
    else:
        frame_len = 1 / task.cfg.label_rate

    if task.cfg.rttm_name is not None:
        rttm_name = task.cfg.rttm_name
    else:
        rttm_name = task.cfg.dataset_name

    for filename in tqdm(res_dict_all):
        speaker_id = filename.split("-")[-1]
        name = filename[: -len(speaker_id) - 1]
        labels = res_dict_all[filename]
        labels = dict(sorted(labels.items()))
        ave_labels = []
        for key in labels:
            ave_labels.append(np.mean(labels[key]))
        labels = signal.medfilt(ave_labels, task.cfg.med_filter)
        for threshold in rttms:
            labels_threshold = change_zeros_to_ones(
                labels, task.cfg.min_silence, threshold, frame_len
            )
            labels_threshold = change_ones_to_zeros(
                labels_threshold, task.cfg.min_speech, threshold, frame_len
            )
            start, duration = 0, 0
            for i, label in enumerate(labels_threshold):
                if label == 1:
                    duration += frame_len
                else:
                    if duration != 0:
                        line = (
                            "SPEAKER "
                            + str(name)
                            + " 1 %.3f" % (start)
                            + " %.3f " % (duration)
                            + "<NA> <NA> "
                            + str(speaker_id)
                            + " <NA> <NA>\n"
                        )
                        rttms[threshold].write(line)
                        duration = 0
                    start = i * frame_len
            if duration != 0:
                line = (
                    "SPEAKER "
                    + str(name)
                    + " 1 %.3f" % (start)
                    + " %.3f " % (duration)
                    + "<NA> <NA> "
                    + str(speaker_id)
                    + " <NA> <NA>\n"
                )
                rttms[threshold].write(line)
    for threshold in rttms:
        rttms[threshold].close()

    for threshold in rttms:
        out = subprocess.check_output(
            [
                "spyder",
                f"-c {task.cfg.collar}",
                "-r",f"{task.cfg.regions}",
                "-p",
                f"{task.cfg.rttm_dir}/{rttm_name.lower()}.rttm",
                f"{rttm_path}_{threshold}",
            ]
        )  # command: spyder -c {task.cfg.collar} -r {task.cfg.regions} ref_rttm hyp_rttm
        out = out.decode("utf-8")

        print(
                f"Eval for threshold {threshold} : {out} \n"
        )
        print(
                f"Eval for threshold {threshold} : {out}\n",
            file=der_write,
        )
    der_write.close()


def cli_main():
    parser = options.get_generation_parser()
    # TODO: replace this workaround with refactoring of `AudioPretraining`
    parser.add_argument(
        "--arch",
        "-a",
        metavar="ARCH",
        default="wav2vec2",
        help="Model architecture. For constructing tasks that rely on "
        "model args (e.g. `AudioPretraining`)",
    )
    parser.add_argument(
        "--rttm_dir", default="SCTK-2.4.12", help="specify reference rttm folder"
    )
    parser.add_argument("--rttm_name", default=None, help="rttm name")
    parser.add_argument(
        "--speaker_embedding_name_dir",
        default="ecapa_feature_dir",
        help="specify speaker embedding directory name",
    )
    parser.add_argument(
        "--speaker_embed_dim", default=192, help="speaker embedding dimension"
    )
    parser.add_argument(
        "--regions",
        default="all",
        help="""it is from spyder package help --regions [all|single|overlap|nonoverlap]
                                  Only evaluate on the selected region type.
                                  Default is all.  - all: all regions.  -
                                  single: only single-speaker regions (ignore
                                  silence and multiple speaker).  - overlap:
                                  only regions with multiple speakers in the
                                  reference.  - nonoverlap: only regions
                                  without multiple speakers in the reference.
                                  [default: all]""",
    )
    parser.add_argument("--collar",default=0.25, type=float, help="Collar size for spyder")
    args = options.parse_args_and_arch(
        parser
    )  # this function will rewrite task/model relative parameter, so these parameter names must occur task/model cfg .
    main(args)


if __name__ == "__main__":
    cli_main()
