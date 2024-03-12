split=train
libricss_path=/mlx_devbox/users/ajy/playground/mnt/bn/junyi-nas-hl2/datasets/jsalt2020_simulate/data/SimLibriCSS-${split}

python scripts/extract_target_speech_libricss.py \
    --rttm_path ${libricss_path}/rttm \
    --orig_audio_path ${libricss_path}/wav \
    --target_audio_path ${libricss_path}/target_audio \
