rttm_path=/mnt/bn/junyi-nas2/codebase/kaldi/egs/ami/s5c/data/train/rttm.annotation
target_audio_path=/opt/tiger/unit_fairseq/examples/decoder_only/librimix/manifest/new_tsv/train.tsv
target_embedding_path=/mnt/bd/alimeeting3/librimix/Libri2Mix/embed

echo "Get target embeddings"
# if [ -d "${target_embedding_path}" ]; then
#     rm -r ${target_embedding_path}
# fi
python3 scripts/extract_target_embedding_librimix.py \
    --target_audio_path ${target_audio_path} \
    --target_embedding_path ${target_embedding_path} \
    --source pretrained_models/pretrain.model
