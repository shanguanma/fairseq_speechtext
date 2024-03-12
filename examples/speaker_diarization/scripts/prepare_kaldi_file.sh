# python /opt/tiger/unit_fairseq/examples/decoder_only/junyi-nas2/codebase/joint-optimization/scripts/prepare_kaldifiles.py \
#                   --source_dir /opt/tiger/unit_fairseq/examples/decoder_only/bd/librimix/Libri3Mix/wav16k/min/metadata \
#                   --rttm_dir /opt/tiger/unit_fairseq/examples/decoder_only/bd/librimix/Libri3Mix/min_rttm \
#                   --fs 16000 \
#                   --num_spk 3 \
#                   --target_dir Datasets/Libri2Mix/wav8k/min/diar

python3 /opt/tiger/unit_fairseq/examples/decoder_only/junyi-nas2/codebase/joint-optimization/scripts/prepare_kaldifiles.py \
                  --source_dir /opt/tiger/unit_fairseq/examples/decoder_only/bd/librimix/Libri3Mix/wav16k/min/metadata \
                  --rttm_dir /opt/tiger/unit_fairseq/examples/decoder_only/junyi-nas2/codebase/LibriMix/metadata/LibriSpeech \
                  --fs 16000 \
                  --num_spk 3 \
                  --target_dir /opt/tiger/unit_fairseq/examples/decoder_only/bd/alimeeting3/librimix
