gpu=0
exp_name=onlyse_layer2
rs_len=4
segment_shift=4
gen_subset=test
med_filter=11
collar=0.0
sample_rate=16000
label_rate=100
mode=max
max_num_speaker=3
min_silence=0.0
dataset_name=Libri23Mix
rttm_name=libri23mix_max

code_path=/workspace/junyi/codebase/joint-optimization
data_path=/workspace/junyi/datasets/librimix
checkpoint_name=checkpoint_10_99000

. ./parse_options.sh

export CUDA_VISIBLE_DEVICES=$gpu

exp_dir=/workspace2/junyi/codebase/joint-optimization/exp_fairseq/joint
ts_vad_path=${code_path}/ts_vad
spk_path=${data_path}/embed
aux_path=${data_path}/ref_speech

results_path=${exp_dir}/${exp_name}/inf

python3 ${ts_vad_path}/generate_spex_joint.py ${data_path} \
  --user-dir ${ts_vad_path} \
  --results-path ${results_path} \
  --path ${exp_dir}/${exp_name}/checkpoints/${checkpoint_name}.pt \
  --task ts_vad_task \
  --task-type joint \
  --spk-path ${spk_path} \
  --aux-path ${aux_path} \
  --rs-len ${rs_len} \
  --segment-shift ${segment_shift} \
  --gen-subset ${gen_subset} \
  --batch-size 1 \
  --sample-rate ${sample_rate} \
  --max-num-speaker ${max_num_speaker} \
  --inference \
  --dataset-name ${dataset_name} \
  --med-filter ${med_filter} \
  --collar ${collar} \
  --librimix-mode ${mode} \
  --label-rate ${label_rate} \
  --min-silence ${min_silence} \
  --rttm-name ${rttm_name} \
  --required-batch-size-multiple 1 \
