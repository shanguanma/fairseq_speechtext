echo " Process dataset: Train/Eval dataset, get json files"
python3 scripts/prepare_data_librimix.py \
    --path_grid /opt/tiger/unit_fairseq/examples/decoder_only/alimeeting3/librimix/rttm_dev_new \
    --path_wav /opt/tiger/unit_fairseq/examples/decoder_only/librimix/Libri2Mix/wav8k/min/train-100/mix_both \
    --out_text /opt/tiger/unit_fairseq/examples/decoder_only/alimeeting3/librimix \
    --tsv_path /opt/tiger/unit_fairseq/examples/decoder_only/librimix/manifest/new_tsv/dev.tsv \
    --type dev \
