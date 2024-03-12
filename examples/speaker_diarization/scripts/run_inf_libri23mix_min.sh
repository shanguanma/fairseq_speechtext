gpu=0
exp_name=baseline_libri23mix_min
rs_len=4
segment_shift=1
gen_subset=test
med_filter=11
collar=0.0
sample_rate=16000
mode=min
rttm_name=libri23mix

code_path=/mnt/bn/junyi-nas2/codebase/joint-optimization
data_path=/mnt/bd/librimix

. ./parse_options.sh

export CUDA_VISIBLE_DEVICES=$gpu

exp_dir=${code_path}/exp_fairseq
ts_vad_path=${code_path}/ts_vad
spk_path=${data_path}/Libri23Mix/embed

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
  --sample-rate ${sample_rate} \
  --max-num-speaker 3 \
  --inference \
  --dataset-name libri23mix \
  --med-filter ${med_filter} \
  --collar ${collar} \
  --librimix-mode ${mode} \
  --rttm-name ${rttm_name} \
