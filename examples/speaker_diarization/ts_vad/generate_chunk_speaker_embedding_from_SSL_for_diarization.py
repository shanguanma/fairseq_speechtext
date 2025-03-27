#!/usr/bin/env bash

import os
import sys
import re
import pathlib
import numpy as np
import soundfile
import wave
import argparse
import torch
import logging
import torchaudio
import torchaudio.compliance.kaldi as Kaldi
import torch.nn as nn
from typing import Optional

from examples.speaker_diarization.ts_vad.models.modules.ecapa_tdnn_unispeech import ECAPA_TDNN_SMALL

def get_args():
    parser = argparse.ArgumentParser(description="Extract speaker embeddings.")
    parser.add_argument(
        "--pretrained_speaker_model", default="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm_ft/wavlm_large_finetune.pth", type=str, help="Model  in unispeech"
    )
    parser.add_argument(
        "--pretrained_model", default="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Large.pt", type=str, help="Model  in unilm"
    )
    parser.add_argument(
        "--model_name", default="wavlm_large_ft", type=str, help="Model name  in unispeech"
    )

    parser.add_argument("--wavs", nargs="+", type=str, help="Wavs")
    # parser.add_argument('--local_model_dir', default='pretrained', type=str, help='Local model dir')
    parser.add_argument(
        "--save_dir",
        type=str,
        default="model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding/Train/cam++_en_zh_feature_dir",
        help="speaker embedding dir",
    )
    parser.add_argument(
        "--length_embedding",
        type=float,
        default=6,
        help="length of embeddings, seconds",
    )
    parser.add_argument(
        "--step_embedding", type=float, default=1, help="step of embeddings, seconds"
    )
    parser.add_argument(
        "--batch_size", type=int, default=96, help="step of embeddings, seconds"
    )
    args = parser.parse_args()
    return args


def extract_embeddings(args, batch):

    if torch.cuda.is_available():
        msg = "Using gpu for inference."
        logging.info(f"{msg}")
        device = torch.device("cuda")
    else:
        msg = "No cuda device is detected. Using cpu."
        logging.info(f"{msg}")
        device = torch.device("cpu")


    #Instantiate model(TODO) maduo add model choice
    model: Optional[nn.Module] = None
    pretrained_state=None
    if args.model_name=="wavlm_base_plus_ft":
        model=ECAPA_TDNN_SMALL(feat_dim=768, emb_dim=256, feat_type='wavlm_base_plus',
                              update_extract=False,pretrain_ckpt=args.pretrained_model)
        pretrained_state = torch.load(args.pretrained_speaker_model, map_location=device)

    elif args.model_name=="wavlm_large_ft":
        model=ECAPA_TDNN_SMALL(feat_dim=1024, emb_dim=256, feat_type='wavlm_large',
                              update_extract=False,pretrain_ckpt=args.pretrained_model)
        pretrained_state = torch.load(args.pretrained_speaker_model, map_location=device)

    # load weight of model
    model.load_state_dict(pretrained_state,strict=False)
    model.to(device)
    model.eval()

    batch = torch.stack(batch)  # expect B,T
    # compute embedding
    embeddings = model.forward(batch.to(device))  # (B,D)
    if isinstance(embeddings, tuple): # for Resnet* model
        embeddings = embeddings[-1]

    embeddings = (
        embeddings.detach()
        )  ## it will remove requires_grad=True of output of model
    assert embeddings.requires_grad == False
    logging.info(f"embeddings shape: {embeddings.shape} !")
    return embeddings


def extract_embed(args, file):
    batch = []
    embeddings = []
    wav_length = wave.open(file, "rb").getnframes()  # entire length for target speech
    if wav_length > int(args.length_embedding * 16000):
        for start in range(
            0,
            wav_length - int(args.length_embedding * 16000),
            int(args.step_embedding * 16000),
        ):
            stop = start + int(args.length_embedding * 16000)
            target_speech, _ = soundfile.read(file, start=start, stop=stop)
            target_speech = torch.FloatTensor(np.array(target_speech))

            # because unispeech offer speaker models which are have feature extractor, it is base SSL model.
            # so I don't need to compute feat
            # compute embedding
            batch.append(target_speech)  # [(T,),(T,),...]
            if len(batch) == args.batch_size:
                embeddings.extend(extract_embeddings(args, batch))
                batch = []
    else:
        embeddings.extend(
            extract_embeddings(
                args, [torch.FloatTensor(np.array(soundfile.read(file)[0]))]
            )
        )
    if len(batch) != 0:
        embeddings.extend(extract_embeddings(args, batch))

    embeddings = torch.stack(embeddings)
    return embeddings


def main():
    args = get_args()
    logging.info(f"Extracting embeddings...")
    # input is wav list
    wav_list_file = args.wavs[0]
    with open(wav_list_file, "r") as f:
        for line in f:
            wav_path = line.strip()

            embedding = extract_embed(args, wav_path)

            dest_dir = os.path.join(
                args.save_dir, os.path.dirname(wav_path).split("/")[-1]
            )
            dest_dir = pathlib.Path(dest_dir)
            dest_dir.mkdir(exist_ok=True, parents=True)
            embedding_name = os.path.basename(wav_path).rsplit(".", 1)[0]
            save_path = dest_dir / f"{embedding_name}.pt"
            torch.save(embedding, save_path)
            logging.info(
                f"The extracted embedding from {wav_path} is saved to {save_path}."
            )


def setup_logging(verbose=1):
    """Make logging setup with a given log level."""
    if verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")


if __name__ == "__main__":
    setup_logging(verbose=1)
    main()
