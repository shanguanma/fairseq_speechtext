gpu=2,3
config_name=base_se_ecapa_noise
exp_name=baseline_se_min_ecapa
other_args=

. ./parse_options.sh

export CUDA_VISIBLE_DEVICES=$gpu

exp_dir=/mnt/bn/junyi-nas2/codebase/joint-optimization/exp_fairseq/se
ts_vad_path=/mnt/bn/junyi-nas2/codebase/joint-optimization/ts_vad
speaker_encoder_path=/mnt/bn/junyi-nas2/codebase/joint-optimization/pretrained_models/pretrain.model # Speaker encoder path

data_path=/mnt/bd/librimix/manifest/libri2mix_16k

mkdir -p ${exp_dir}/${exp_name}

fairseq-hydra-train \
  --config-dir ${ts_vad_path}/config \
  --config-name ${config_name} \
  hydra.run.dir=${exp_dir}/${exp_name} \
  hydra.job.name=${exp_name} \
  hydra.sweep.dir=${exp_dir}/${exp_name} \
  task.data=${data_path} \
  common.user_dir=${ts_vad_path} \
  +model.speaker_encoder_path=${speaker_encoder_path} \
  ${other_args}
