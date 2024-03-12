# gid=$1

# echo "Get target speech"
# # if [ -d "${target_audio_path}" ]; then
# #     rm -r ${target_audio_path}
# # fi
# python3 scripts/extract_target_speech_ami.py \
#     --rttm_path ${rttm_path} \
#     --orig_audio_path ${audio_dir} \
#     --target_audio_path ${target_audio_path}

echo "Get target embeddings"
# if [ -d "${target_embedding_path}" ]; then
#     rm -r ${target_embedding_path}
# fi
python3 scripts/extract_target_embedding_libricss.py \
    --target_audio_path /mnt/bn/junyi-nas-hl2/datasets/jsalt2020_simulate/data/SimLibriCSS-test/target_audio \
    --target_embedding_path /mnt/bn/junyi-nas-hl2/datasets/jsalt2020_simulate/data/SimLibriCSS-test/embed \
    --source pretrained_models/pretrain.model
