rttm_path=/mnt/bn/junyi-nas2/codebase/kaldi/egs/ami/s5c/data/train/rttm.annotation
audio_dir=/mnt/bn/junyi-nas2/ami/wav_db
target_audio_path=/mnt/bn/junyi-nas2/ami/tgt_wav_db/train
target_embedding_path=/mnt/bn/junyi-nas2/ami/embed_db/train

echo "Get target speech"
# if [ -d "${target_audio_path}" ]; then
#     rm -r ${target_audio_path}
# fi
python3 scripts/extract_target_speech_ami.py \
    --rttm_path ${rttm_path} \
    --orig_audio_path ${audio_dir} \
    --target_audio_path ${target_audio_path}

echo "Get target embeddings"
# if [ -d "${target_embedding_path}" ]; then
#     rm -r ${target_embedding_path}
# fi
python3 scripts/extract_target_embedding_ami.py \
    --target_audio_path ${target_audio_path} \
    --target_embedding_path ${target_embedding_path} \
    --source pretrained_models/pretrain.model
