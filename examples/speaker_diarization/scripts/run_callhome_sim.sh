gpu=0
config_name=base_callhome
exp_name=baseline_callhome_sim
other_args=
code_path=/workspace/junyi/codebase/joint-optimization
data_path=/workspace2/junyi/datasets/callhome_sim/--no-use-rirs--use-noises

. ./parse_options.sh

export CUDA_VISIBLE_DEVICES=$gpu

exp_dir=${code_path}/exp_fairseq
ts_vad_path=${code_path}/ts_vad
speech_encoder_path=${code_path}/pretrain_models/pretrain.model # Speaker encoder path
spk_path=${data_path}

mkdir -p ${exp_dir}/${exp_name}

python ${code_path}/fairseq/fairseq_cli/hydra_train.py \
  --config-dir ${ts_vad_path}/config \
  --config-name ${config_name} \
  hydra.run.dir=${exp_dir}/${exp_name} \
  hydra.job.name=${exp_name} \
  hydra.sweep.dir=${exp_dir}/${exp_name} \
  task.data=${data_path} \
  common.user_dir=${ts_vad_path} \
  +task.spk_path=${spk_path} \
  +model.speech_encoder_path=${speech_encoder_path} \
  checkpoint.save_interval_updates=10000 \
  ${other_args}
