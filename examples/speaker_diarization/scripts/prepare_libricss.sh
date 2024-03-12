#!/bin/bash
split=test

python3 scripts/prepare_callhome_sim.py \
    --path_rttm /mlx_devbox/users/ajy/playground/mnt/bn/junyi-nas-hl2/datasets/jsalt2020_simulate/data/SimLibriCSS-${split}/rttm \
    --path_wav /mlx_devbox/users/ajy/playground/mnt/bn/junyi-nas-hl2/datasets/jsalt2020_simulate/data/SimLibriCSS-${split}/wav \
    --out_text /mlx_devbox/users/ajy/playground/mnt/bn/junyi-nas-hl2/datasets/jsalt2020_simulate/data/SimLibriCSS-${split} \
    --type ${split} \
    --label_rate 100
