gpu=0
exp_name=baseline_librimix
rs_len=4
segment_shift=1
gen_subset=test
med_filter=11
collar=0.0
sample_rate=16000
mode=min
ovr_ratio=0.2


code_path=/workspace2/sinan/joint-optimization
data_path=/workspace/sinan/Datasets

. ./parse_options.sh

if [[ -z "$rttm_name" ]]; then
  rttm_name=sparse_2_${ovr_ratio}
fi
 

export CUDA_VISIBLE_DEVICES=$gpu

exp_dir=${code_path}/exp_fairseq
ts_vad_path=${code_path}/ts_vad
spk_path=${data_path}/SparseLibri2Mix/embed

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
  --max-num-speaker 2 \
  --inference \
  --dataset-name SparseLibri2Mix \
  --med-filter ${med_filter} \
  --collar ${collar} \
  --librimix-mode ${mode} \
  --rttm-name ${rttm_name} \
  --ovr-ratio ${ovr_ratio} \
