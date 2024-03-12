gpu=0
config_name=base_librimix
exp_name=baseline_librimix
other_args=
code_path=/mnt/bn/junyi-nas2/codebase/joint-optimization
data_path=/mnt/bd/librimix

. ./parse_options.sh

export CUDA_VISIBLE_DEVICES=$gpu

exp_dir=${code_path}/exp_fairseq
ts_vad_path=${code_path}/ts_vad
speech_encoder_path=${code_path}/pretrained_models/pretrain.model # Speaker encoder path
spk_path=${data_path}/Libri2Mix/embed

mkdir -p ${exp_dir}/${exp_name}

fairseq-hydra-train \
  --config-dir ${ts_vad_path}/config \
  --config-name ${config_name} \
  hydra.run.dir=${exp_dir}/${exp_name} \
  hydra.job.name=${exp_name} \
  hydra.sweep.dir=${exp_dir}/${exp_name} \
  task.data=${data_path} \
  common.user_dir=${ts_vad_path} \
  +task.spk_path=${spk_path} \
  +model.speech_encoder_path=${speech_encoder_path} \
  ${other_args}
