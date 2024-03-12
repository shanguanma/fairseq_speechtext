gpu=0,1,2,3
config_name=base_librimix_joint_new
exp_name=spk_diar_input_onlydiar_2layer
other_args=
sample_rate=16000
label_rate=100
code_path=/workspace/junyi/codebase/joint-optimization
data_path=/workspace/junyi/datasets/librimix

. ./parse_options.sh

export CUDA_VISIBLE_DEVICES=$gpu

exp_dir=${code_path}/exp_fairseq/joint
ts_vad_path=${code_path}/ts_vad
aux_path=${data_path}/ref_speech
spk_path=${data_path}/embed

mkdir -p ${exp_dir}/${exp_name}

python3 ${code_path}/fairseq/fairseq_cli/hydra_train.py \
  --config-dir ${ts_vad_path}/config \
  --config-name ${config_name} \
  hydra.run.dir=${exp_dir}/${exp_name} \
  hydra.job.name=${exp_name} \
  hydra.sweep.dir=${exp_dir}/${exp_name} \
  task.data=${data_path} \
  common.user_dir=${ts_vad_path} \
  task.sample_rate=${sample_rate} \
  +task.spk_path=${spk_path} \
  +task.aux_path=${aux_path} \
  task.label_rate=${label_rate} \
  ${other_args}
