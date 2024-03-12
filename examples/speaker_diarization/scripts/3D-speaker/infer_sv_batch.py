# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""
This script will download the pretrained models from modelscope (https://www.modelscope.cn/models)
based on the given model id, and extract the embeddings from the given audio files. 
Please pre-install "modelscope".
Usage:
    1. extract the embedding from the wav file.
        `python infer_sv.py --model_id $model_id --wavs $wav_path `
    2. extract embeddings from two wav files and compute the similarity score.
        `python infer_sv.py --model_id $model_id --wavs $wav_path1 $wav_path2 `
    3. extract embeddings from the wav list.
        `python infer_sv.py --model_id $model_id --wavs $wav_list `
"""

import os
import sys
import json
import glob
import tqdm
import re
import pathlib
import numpy as np
import argparse
import torch
import torchaudio

try:
    from speakerlab.process.processor import FBank
except ImportError:
    sys.path.append("%s/../.." % os.path.dirname(__file__))
    from speakerlab.process.processor import FBank

from speakerlab.utils.builder import dynamic_import

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.pipelines.util import is_official_hub_path

parser = argparse.ArgumentParser(description="Extract speaker embeddings.")
parser.add_argument("--model_id", default="", type=str, help="Model id in modelscope")
parser.add_argument(
    "--target_embedding_path", help="the path for the output embeddings"
)
parser.add_argument("--wavs", nargs="+", type=str, help="Wavs")
parser.add_argument(
    "--local_model_dir", default="pretrained", type=str, help="Local model dir"
)
parser.add_argument(
    "--length_embedding", type=float, default=6, help="length of embeddings, seconds"
)
parser.add_argument(
    "--step_embedding", type=float, default=1, help="step of embeddings, seconds"
)
parser.add_argument(
    "--batch_size", type=int, default=96, help="step of embeddings, seconds"
)

CAMPPLUS_VOX = {
    "obj": "speakerlab.models.campplus.DTDNN.CAMPPlus",
    "args": {
        "feat_dim": 80,
        "embedding_size": 512,
    },
}

CAMPPLUS_COMMON = {
    "obj": "speakerlab.models.campplus.DTDNN.CAMPPlus",
    "args": {
        "feat_dim": 80,
        "embedding_size": 192,
    },
}

ERes2Net_VOX = {
    "obj": "speakerlab.models.eres2net.ResNet.ERes2Net",
    "args": {
        "feat_dim": 80,
        "embedding_size": 192,
    },
}

supports = {
    "damo/speech_campplus_sv_en_voxceleb_16k": {
        "revision": "v1.0.2",
        "model": CAMPPLUS_VOX,
        "model_pt": "campplus_voxceleb.bin",
    },
    "damo/speech_campplus_sv_zh-cn_16k-common": {
        "revision": "v1.0.0",
        "model": CAMPPLUS_COMMON,
        "model_pt": "campplus_cn_common.bin",
    },
    "damo/speech_eres2net_sv_en_voxceleb_16k": {
        "revision": "v1.0.2",
        "model": ERes2Net_VOX,
        "model_pt": "pretrained_eres2net.ckpt",
    },
}


def main():
    args = parser.parse_args()
    assert isinstance(args.model_id, str) and is_official_hub_path(
        args.model_id
    ), "Invalid modelscope model id."
    assert args.model_id in supports, "Model id not currently supported."
    save_dir = os.path.join(args.local_model_dir, args.model_id.split("/")[1])
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    conf = supports[args.model_id]
    # download models from modelscope according to model_id
    cache_dir = snapshot_download(
        args.model_id,
        revision=conf["revision"],
    )
    cache_dir = pathlib.Path(cache_dir)

    embedding_dir = save_dir / "embeddings"
    embedding_dir.mkdir(exist_ok=True, parents=True)

    # link
    download_files = ["examples", conf["model_pt"]]
    for src in cache_dir.glob("*"):
        if re.search("|".join(download_files), src.name):
            dst = save_dir / src.name
            try:
                dst.unlink()
            except FileNotFoundError:
                pass
            dst.symlink_to(src)

    pretrained_model = save_dir / conf["model_pt"]
    pretrained_state = torch.load(pretrained_model, map_location="cpu")

    # load model
    model = conf["model"]
    embedding_model = dynamic_import(model["obj"])(**model["args"]).cuda()
    embedding_model.load_state_dict(pretrained_state)
    embedding_model.eval()

    def load_wav(wav_file, obj_fs=16000):
        wav, fs = torchaudio.load(wav_file)
        if fs != obj_fs:
            print(
                f"[WARNING]: The sample rate of {wav_file} is not {obj_fs}, resample it."
            )
            wav, fs = torchaudio.sox_effects.apply_effects_tensor(
                wav, fs, effects=[["rate", str(obj_fs)]]
            )
            if wav.shape[0] > 1:
                wav = wav[0, :].unsqueeze(0)
        return wav

    feature_extractor = FBank(80, sample_rate=16000, mean_nor=True)

    def compute_embedding(feat):
        # compute embedding
        with torch.no_grad():
            embedding = embedding_model(feat).detach().cpu()

        return embedding

    # extract embeddings
    print(f"[INFO]: Extracting embeddings...")

    # input one wav file
    files = sorted(glob.glob(args.wavs[0] + "/*/*.wav"))

    # input is wav path
    for file in tqdm.tqdm(files):
        if "all" not in file:
            wav_path = file
            output_file = (
                args.target_embedding_path
                + "/"
                + file.split("/")[-2]
                + "/"
                + file.split("/")[-1].replace(".wav", ".pt")
            )
            # load wav
            wav = load_wav(wav_path)
            batch = []
            embeds = []
            for start in range(
                0,
                wav.size(1) - int(args.length_embedding * 16000),
                int(args.step_embedding * 16000),
            ):
                stop = start + int(args.length_embedding * 16000)
                batch.append(feature_extractor(wav[:, start:stop]))
                if len(batch) == args.batch_size:
                    model_input = torch.stack(batch)
                    embedding = compute_embedding(model_input.cuda())
                    embeds.append(embedding)
                    batch = []
            if len(batch) != 0:
                model_input = torch.stack(batch)
                embedding = compute_embedding(model_input.cuda())
                embeds.append(embedding)
            embeds = torch.cat(embeds, dim=0)
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            # torch.save(embeds, output_file)
        # print(f'[INFO]: The extracted embedding from {wav_path} is saved to {save_path}.')


if __name__ == "__main__":
    main()
