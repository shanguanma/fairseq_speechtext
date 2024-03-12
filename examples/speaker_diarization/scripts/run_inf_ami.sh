gpu=0
exp_name=baseline_ami
rs_len=4
segment_shift=1
gen_subset=test
rttm_name=ami

code_path=/mnt/bn/junyi-nas2/codebase/joint-optimization
data_path=/mnt/bd/alimeeting3/ami/tgt_wav_db
spk_path=/mnt/bd/alimeeting3/ami/embed_db

. ./parse_options.sh

export CUDA_VISIBLE_DEVICES=$gpu

exp_dir=${code_path}/exp_fairseq
ts_vad_path=${code_path}/ts_vad

results_path=${exp_dir}/${exp_name}/inf

python3 ${ts_vad_path}/generate.py ${data_path} \
  --user-dir ${ts_vad_path} \
  --results-path ${results_path} \
  --path ${exp_dir}/${exp_name}/checkpoints/checkpoint_best.pt \
  --task ts_vad_task \
  --spk-path ${spk_path} \
  --rs-len ${rs_len} \
  --segment-shift ${segment_shift} \
  --gen-subset ${gen_subset} \
  --batch-size 64 \
  --sample-rate 16000 \
  --max-num-speaker 5 \
  --inference \
  --dataset-name ami \
  --med-filter 21 \
  --collar 0.0 \
  --rttm-name ${rttm_name} \