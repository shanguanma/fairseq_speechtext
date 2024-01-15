#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys

from copy import deepcopy
from scipy.signal import lfilter

import numpy as np
from tqdm import tqdm
import soundfile as sf
import os.path as osp
import os
import logging
import torch
# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
#torch.set_num_threads(1)
#torch.set_num_interop_threads(1)

def get_parser():
    parser = argparse.ArgumentParser(description="compute vad segments")
    parser.add_argument(
        "--rvad-home",
        "-r",
        help="path to rvad home (see https://github.com/zhenghuatan/rVADfast)",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        default="tests",
    )
    parser.add_argument(
        "--wav_scp",
        default="wav.scp",
    )
    return parser


def rvad(speechproc, path):
    winlen, ovrlen, pre_coef, nfilter, nftt = 0.025, 0.01, 0.97, 20, 512
    ftThres = 0.5
    vadThres = 0.4
    opts = 1

    data, fs = sf.read(path)
    assert fs == 16_000, "sample rate must be 16khz"
    ft, flen, fsh10, nfr10 = speechproc.sflux(data, fs, winlen, ovrlen, nftt)

    # --spectral flatness --
    pv01 = np.zeros(ft.shape[0])
    pv01[np.less_equal(ft, ftThres)] = 1
    pitch = deepcopy(ft)

    pvblk = speechproc.pitchblockdetect(pv01, pitch, nfr10, opts)

    # --filtering--
    ENERGYFLOOR = np.exp(-50)
    b = np.array([0.9770, -0.9770])
    a = np.array([1.0000, -0.9540])
    fdata = lfilter(b, a, data, axis=0)

    # --pass 1--
    noise_samp, noise_seg, n_noise_samp = speechproc.snre_highenergy(
        fdata, nfr10, flen, fsh10, ENERGYFLOOR, pv01, pvblk
    )

    # sets noisy segments to zero
    for j in range(n_noise_samp):
        fdata[range(int(noise_samp[j, 0]), int(noise_samp[j, 1]) + 1)] = 0

    vad_seg = speechproc.snre_vad(
        fdata, nfr10, flen, fsh10, ENERGYFLOOR, pv01, pvblk, vadThres
    )
    return vad_seg, data


def main():
    parser = get_parser()
    args = parser.parse_args()

    sys.path.append(args.rvad_home)
    import speechproc

    stride = 160
    #lines = sys.stdin.readlines()
    #root = lines[0].rstrip()
    #for fpath in tqdm(lines):

    rank = int(os.environ['LOCAL_RANK'])        ## processing id
    threads_num = int(os.environ['WORLD_SIZE']) ## cpu numbers, is setted by --nproc_per_node 
    logging.info("rank {}/{}.".format(
        rank, threads_num,
    ))

    all_recs = []
    if args.wav_scp is not None and len(args.wav_scp) > 0:
        for one_line in open(args.wav_scp, "rt", encoding="utf-8"):
            path = one_line.strip()
            #logging.info(f"path: {path}")
            key, path = path.split("\t", maxsplit=1)
            all_recs.append((key, path))
    all_recs.sort(key=lambda x: x[0])
    local_all_recs = all_recs[rank::threads_num]
    meeting_count = 0
    output=open(os.path.join(args.output_dir, f"train.{rank}.vads"),"wt")
    #for i, (uttid, wav_path) in enumerate(local_all_recs):
    for i, (uttid, wav_path) in tqdm(enumerate(local_all_recs), total=len(local_all_recs), ascii=True):
        vads, wav = rvad(speechproc, wav_path)
        start = None
        vad_segs = []
        for i, v in enumerate(vads):
            if start is None and v == 1:
                start = i * stride
            elif start is not None and v == 0:
                vad_segs.append((start, i * stride))
                start = None
        if start is not None:
            vad_segs.append((start, len(wav)))
        file_o = " ".join(f"{v[0]}:{v[1]}" for v in vad_segs)
        #print(" ".join(f"{v[0]}:{v[1]}" for v in vad_segs))
        output.write(f"{uttid} {file_o}\n")

        if i % 50 == 0:
            logging.info("{}/{}: process {}.".format(rank, threads_num, uttid))
            output.flush()


        meeting_count += 1
    logging.info("{}/{}: Complete {} records.".format(rank, threads_num, meeting_count))
    #output.close()

if __name__ == "__main__":
    import logging
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()



"""
## 
## note: tests/wav5.scp contain 20 lines, every line contains uttid path of audio wavform.
## note: it consums 27 mins for processing 20 utterances
torchrun --nproc_per_node=5 --master_port=12345 codebase/fairseq_speechtext/examples/wav2vec/unsupervised/scripts/vads_for_wavscp_parallel.py -r codebase/rVADfast --output_dir tests --wav_scp tests/wav5.scp
WARNING:torch.distributed.run:
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
*****************************************
2024-01-09 16:53:55,383 INFO [vads_for_wavscp_parallel.py:98] rank 2/5.
  0%|                                                                                                                                                                       | 0/4 [00:00<?, ?it/s]2024-01-09 16:53:55,396 INFO [vads_for_wavscp_parallel.py:98] rank 0/5.
  0%|                                                                                                                                                                       | 0/4 [00:00<?, ?it/s]2024-01-09 16:53:55,494 INFO [vads_for_wavscp_parallel.py:98] rank 1/5.
  0%|                                                                                                                                                                       | 0/4 [00:00<?, ?it/s]2024-01-09 16:53:55,511 INFO [vads_for_wavscp_parallel.py:98] rank 4/5.
2024-01-09 16:53:55,513 INFO [vads_for_wavscp_parallel.py:98] rank 3/5.
100%|##############################################################################################################################################################| 4/4 [26:37<00:00, 399.41s/it]
2024-01-09 17:20:33,029 INFO [vads_for_wavscp_parallel.py:136] 2/5: Complete 4 records.
100%|##############################################################################################################################################################| 4/4 [26:43<00:00, 400.94s/it]
2024-01-09 17:20:39,267 INFO [vads_for_wavscp_parallel.py:136] 4/5: Complete 4 records.
100%|##############################################################################################################################################################| 4/4 [26:43<00:00, 400.98s/it]
2024-01-09 17:20:39,432 INFO [vads_for_wavscp_parallel.py:136] 1/5: Complete 4 records.
100%|##############################################################################################################################################################| 4/4 [26:46<00:00, 401.58s/it]
2024-01-09 17:20:41,818 INFO [vads_for_wavscp_parallel.py:136] 3/5: Complete 4 records.
100%|##############################################################################################################################################################| 4/4 [26:48<00:00, 402.24s/it]
2024-01-09 17:20:44,346 INFO [vads_for_wavscp_parallel.py:136] 0/5: Complete 4 records.

"""
