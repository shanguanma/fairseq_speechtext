#!/usr/local/env python
# coding: utf-8
# Copyright 2022  Johns Hopkins University (author: Desh Raj)

import argparse
from pathlib import Path
from copy import deepcopy
import logging

import pytorch_lightning as pl
from pyannote.database import get_protocol
from pyannote.audio import Model
from pyannote.audio.tasks import Segmentation

#Since pyannote.database version 5.0, configuration files must be loaded into the registry like that:
from pyannote.database import registry
registry.load_database("clustering_based/pyannote/database.yml")
from pyannote.database import FileFinder


def read_args():
    parser = argparse.ArgumentParser(description="Fine-tune Pyannote model on data")
    parser.add_argument(
        "--dataset",
        type=str,
        help="Name of dataset of finetune vad model",
        choices=["AMI", "AISHELL-4", "AliMeeting"],
    )
    parser.add_argument("--exp_dir", type=str, help="Experiment directory")
    parser.add_argument("--pyannote_segmentation_model",type=str, help="pyannote/segmentation model directory")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = read_args()

    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(exist_ok=True, parents=True)

    # Create dataset
    logging.info("Loading dataset")
    ami = get_protocol(f"{args.dataset}.SpeakerDiarization.only_words",preprocessors={"audio": FileFinder()})

    # Get pretrained segmentation model
    logging.info("Loading pretrained model")
    pretrained = Model.from_pretrained(args.pyannote_segmentation_model,use_auth_token=None)

    # Create new segmentation task for dataset
    logging.info("Creating new segmentation task")
    seg_task = Segmentation(ami, duration=5.0, max_num_speakers=4, num_workers=4)

    # Copy pretrained model and override task
    logging.info("Copying pretrained model")
    finetuned = deepcopy(pretrained)
    finetuned.task = seg_task

    # Create trainer
    logging.info("Creating trainer")
    trainer = pl.Trainer(accelerator="cpu", max_epochs=1, default_root_dir=exp_dir)

    # Trainer model
    logging.info("Training model")
    trainer.fit(finetuned)
