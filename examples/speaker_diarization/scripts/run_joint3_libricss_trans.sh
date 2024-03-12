gpu=0,1,2,3,4,5,6,7
config_name=joint_libricss_trans
exp_name=joint_libricss
other_args=
sample_rate=16000
label_rate=100
code_path=/mnt/bn/junyi-nas-hl2/codebase/joint-optimization
data_path=/mnt/bn/junyi-nas-hl2/datasets/jsalt2020_simulate/data

. ./parse_options.sh

export CUDA_VISIBLE_DEVICES=$gpu

exp_dir=${code_path}/exp_fairseq/joint
ts_vad_path=${code_path}/ts_vad
spk_path=${data_path}

mkdir -p ${exp_dir}/${exp_name}

fairseq-hydra-train \
  --config-dir ${ts_vad_path}/config \
  --config-name ${config_name} \
  hydra.run.dir=${exp_dir}/${exp_name} \
  hydra.job.name=${exp_name} \
  hydra.sweep.dir=${exp_dir}/${exp_name} \
  task.data=${data_path} \
  common.user_dir=${ts_vad_path} \
  task.sample_rate=${sample_rate} \
  +task.spk_path=${spk_path} \
  task.label_rate=${label_rate} \
  ${other_args}
  # +model.ts_vad_path=${tsvad_path} \
