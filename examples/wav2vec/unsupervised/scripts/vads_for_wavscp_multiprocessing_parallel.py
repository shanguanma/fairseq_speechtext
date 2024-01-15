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
    parser.add_argument(
        "--num_workers",
        type=int,
        default=5,
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

def write_vads(args,line):
    sys.path.append(args.rvad_home)
    import speechproc
    #logging.info(f"line: {line}")
    stride = 160
    with open(os.path.join(args.output_dir, f"train.concurrent.vads"),"a") as file:
        logging.info(f"wav: {line[1]}!!!")
        vads, wav = rvad(speechproc,line[1])
        logging.info(f"vads: {vads}")
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
        file.write(f"{line[0]} {file_o}\n")

import multiprocessing

# 创建一个进程锁
lock = multiprocessing.Lock()

# 共享的计数器
counter = multiprocessing.Value('i', 0)


def write_vads_for_multiprocessing(args,line):
    with lock:
        # 获取锁，保证只有一个进程可以执行下面的代码
        sys.path.append(args.rvad_home)
        import speechproc
        #logging.info(f"line: {line}")
        stride = 160
        with open(os.path.join(args.output_dir, f"train.multiprocessing.vads"),"a") as file:
            logging.info(f"wav: {line[1]}!!!")
            vads, wav = rvad(speechproc,line[1])
            logging.info(f"vads: {vads}")
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
            file.write(f"{line[0]} {file_o}\n")

        # 增加计数器
        with counter.get_lock():
            counter.value += 1





#def write_data_process_parallel(args,data_list):
#    # 创建一个进程池，最大进程数为CPU核心数
#    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
#        # 使用submit方法提交任务，返回Future对象列表
#        futures = [executor.submit(write_vads, args,data) for data in data_list]
        
        # 等待所有任务完成
#        concurrent.futures.wait(futures)
#    logging.info("process_parallel finish!!!!!!!!!")


#def write_data_thread_parallel(args,data_list):
#    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
#        # 使用submit方法提交任务，返回Future对象列表
#        futures = [executor.submit(write_vads, args,data) for data in data_list]
#
#        # 等待所有任务完成
#        concurrent.futures.wait(futures)
#    logging.info("thread_parallel finish!!!!!!!!!")




def main():
    parser = get_parser()
    args = parser.parse_args()
     
    data_list = []
    if args.wav_scp is not None and len(args.wav_scp) > 0:
        for one_line in open(args.wav_scp, "rt", encoding="utf-8"):
            path = one_line.strip()
            #logging.info(f"path: {path}")
            key, path = path.split("\t", maxsplit=1)
            data_list.append((key, path))

    pool = multiprocessing.Pool(processes=args.num_workers)
    pool.starmap(write_vads_for_multiprocessing, [(args, data) for data in data_list])
    pool.close()
    pool.join()

    # 打印计数器的值
    print('Counter:', counter.value)  


if __name__ == "__main__":
    import logging
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()

    """
    note: tests/wav5.scp contain 20 lines, every line contains uttid path of audio wavform.
    note: it consums 221mins for processing 20 utterances.
    python3 codebase/fairseq_speechtext/examples/wav2vec/unsupervised/scripts/vads_for_wavscp_multiprocessing_parallel.py  -r codebase/rVADfast --output_dir tests --wav_scp tests/wav5.scp --num_workers 5
2024-01-09 17:44:32,487 INFO [vads_for_wavscp_multiprocessing_parallel.py:128] wav: /mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000000_--5llN02F84.opus!!!
2024-01-09 18:21:58,267 INFO [vads_for_wavscp_multiprocessing_parallel.py:130] vads: [0 1 1 ... 0 0 0]
2024-01-09 18:21:58,501 INFO [vads_for_wavscp_multiprocessing_parallel.py:128] wav: /mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000006_-0IBbmCokvY.opus!!!
2024-01-09 18:58:21,190 INFO [vads_for_wavscp_multiprocessing_parallel.py:130] vads: [0 0 1 ... 0 0 0]
2024-01-09 18:58:21,572 INFO [vads_for_wavscp_multiprocessing_parallel.py:128] wav: /mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000002_--s1SMM6PBU.opus!!!
2024-01-09 19:32:58,154 INFO [vads_for_wavscp_multiprocessing_parallel.py:130] vads: [1 1 1 ... 0 0 0]
2024-01-09 19:32:58,422 INFO [vads_for_wavscp_multiprocessing_parallel.py:128] wav: /mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000003_--tTfmNRKHU.opus!!!
2024-01-09 19:45:33,370 INFO [vads_for_wavscp_multiprocessing_parallel.py:130] vads: [0 0 0 ... 0 0 0]
2024-01-09 19:45:33,475 INFO [vads_for_wavscp_multiprocessing_parallel.py:128] wav: /mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000005_-0CH_WrGBKY.opus!!!
2024-01-09 19:52:03,949 INFO [vads_for_wavscp_multiprocessing_parallel.py:130] vads: [0 0 0 ... 0 0 0]
2024-01-09 19:52:04,049 INFO [vads_for_wavscp_multiprocessing_parallel.py:128] wav: /mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000001_--TiI-YRU38.opus!!!
2024-01-09 19:54:55,023 INFO [vads_for_wavscp_multiprocessing_parallel.py:130] vads: [0 0 0 ... 0 0 0]
2024-01-09 19:54:55,041 INFO [vads_for_wavscp_multiprocessing_parallel.py:128] wav: /mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000007_-0P6CYbKnsQ.opus!!!
2024-01-09 20:01:20,786 INFO [vads_for_wavscp_multiprocessing_parallel.py:130] vads: [1 1 1 ... 0 0 0]
2024-01-09 20:01:20,823 INFO [vads_for_wavscp_multiprocessing_parallel.py:128] wav: /mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000008_-0RZxs2c3X8.opus!!!
2024-01-09 20:07:54,885 INFO [vads_for_wavscp_multiprocessing_parallel.py:130] vads: [0 0 0 ... 0 0 0]
2024-01-09 20:07:54,922 INFO [vads_for_wavscp_multiprocessing_parallel.py:128] wav: /mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000009_-0p8pYdlfjY.opus!!!
2024-01-09 20:16:24,906 INFO [vads_for_wavscp_multiprocessing_parallel.py:130] vads: [0 0 0 ... 0 0 0]
2024-01-09 20:16:24,956 INFO [vads_for_wavscp_multiprocessing_parallel.py:128] wav: /mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000010_-0zkH4K38Z0.opus!!!
2024-01-09 20:21:21,295 INFO [vads_for_wavscp_multiprocessing_parallel.py:130] vads: [0 0 0 ... 0 0 0]
2024-01-09 20:21:21,445 INFO [vads_for_wavscp_multiprocessing_parallel.py:128] wav: /mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000011_-1800ATDh64.opus!!!
2024-01-09 20:27:51,380 INFO [vads_for_wavscp_multiprocessing_parallel.py:130] vads: [0 0 0 ... 0 0 0]
2024-01-09 20:27:51,417 INFO [vads_for_wavscp_multiprocessing_parallel.py:128] wav: /mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000012_-1EdlcTjw8k.opus!!!
2024-01-09 20:34:32,982 INFO [vads_for_wavscp_multiprocessing_parallel.py:130] vads: [0 0 0 ... 1 1 0]
2024-01-09 20:34:33,021 INFO [vads_for_wavscp_multiprocessing_parallel.py:128] wav: /mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000013_-1T8BwkquEI.opus!!!
2024-01-09 20:40:31,675 INFO [vads_for_wavscp_multiprocessing_parallel.py:130] vads: [0 0 0 ... 1 1 0]
2024-01-09 20:40:31,709 INFO [vads_for_wavscp_multiprocessing_parallel.py:128] wav: /mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000014_-1_DSSMnDCE.opus!!!
2024-01-09 20:47:12,155 INFO [vads_for_wavscp_multiprocessing_parallel.py:130] vads: [0 0 0 ... 0 0 0]
2024-01-09 20:47:12,193 INFO [vads_for_wavscp_multiprocessing_parallel.py:128] wav: /mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000015_-1b96ICVDV4.opus!!!
2024-01-09 20:53:57,745 INFO [vads_for_wavscp_multiprocessing_parallel.py:130] vads: [0 0 0 ... 0 0 0]
2024-01-09 20:53:57,783 INFO [vads_for_wavscp_multiprocessing_parallel.py:128] wav: /mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000016_-1f3NNSInzU.opus!!!
2024-01-09 20:59:45,048 INFO [vads_for_wavscp_multiprocessing_parallel.py:130] vads: [0 0 0 ... 0 0 0]
2024-01-09 20:59:45,082 INFO [vads_for_wavscp_multiprocessing_parallel.py:128] wav: /mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000017_-1fY8qw-VrM.opus!!!
2024-01-09 21:06:16,342 INFO [vads_for_wavscp_multiprocessing_parallel.py:130] vads: [0 0 0 ... 0 0 0]
2024-01-09 21:06:16,381 INFO [vads_for_wavscp_multiprocessing_parallel.py:128] wav: /mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000018_-1pX3Y3M0GI.opus!!!
2024-01-09 21:12:52,095 INFO [vads_for_wavscp_multiprocessing_parallel.py:130] vads: [0 0 0 ... 0 0 0]
2024-01-09 21:12:52,134 INFO [vads_for_wavscp_multiprocessing_parallel.py:128] wav: /mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000019_-1rYBgd8XXs.opus!!!
2024-01-09 21:19:19,330 INFO [vads_for_wavscp_multiprocessing_parallel.py:130] vads: [0 0 0 ... 0 0 0]
2024-01-09 21:19:19,369 INFO [vads_for_wavscp_multiprocessing_parallel.py:128] wav: /mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000020_-1rpxhaKHHg.opus!!!
2024-01-09 21:25:36,036 INFO [vads_for_wavscp_multiprocessing_parallel.py:130] vads: [0 0 0 ... 0 0 0]
Counter: 20
    """
