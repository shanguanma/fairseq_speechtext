gpu=0
exp_name=baseline_se_min_6layer
gen_subset=test

code_path=/mnt/bn/junyi-nas2/codebase/joint-optimization
data_path=/mnt/bd/librimix/manifest/libri2mix_16k

. ./parse_options.sh

export CUDA_VISIBLE_DEVICES=$gpu

exp_dir=${code_path}/exp_fairseq/se
ts_vad_path=${code_path}/ts_vad

results_path=${exp_dir}/${exp_name}/inf

python3 ${ts_vad_path}/generate_spex.py ${data_path} \
  --user-dir ${ts_vad_path} \
  --results-path ${results_path} \
  --path ${exp_dir}/${exp_name}/checkpoints/checkpoint_best.pt \
  --task ts_vad_task \
  --gen-subset ${gen_subset} \
  --batch-size 1 \
  --task-type extraction \
  --inference \
  --segment-shift 4
