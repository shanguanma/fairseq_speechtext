#!/bin/bash

python3 scripts/prepare_callhome_sim.py \
    --path_rttm /workspace2/junyi/datasets/callhome_sim/--no-use-rirs--use-noises/train/data/rttm \
    --path_wav /workspace2/junyi/datasets/callhome_sim/--no-use-rirs--use-noises/train/wavs \
    --out_text /workspace2/junyi/datasets/callhome_sim/--no-use-rirs--use-noises/train \
    --type train \
    --label_rate 25
