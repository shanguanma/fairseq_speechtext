echo " Process dataset: Train/Eval dataset, get json files"
python3 scripts/prepare_data_ami.py \
    --path_grid /opt/tiger/unit_fairseq/examples/decoder_only/junyi-nas2/codebase/kaldi/egs/ami/s5c/data/test/rttm.annotation \
    --path_wav /opt/tiger/unit_fairseq/examples/decoder_only/junyi-nas2/ami/wav_db \
    --out_text /opt/tiger/unit_fairseq/examples/decoder_only/junyi-nas2/ami/tgt_wav_db/test \
    --type test \
