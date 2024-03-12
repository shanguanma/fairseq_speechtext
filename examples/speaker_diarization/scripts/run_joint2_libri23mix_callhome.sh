gpu=0,1,2,3,4,5,6,7
config_name=base_librimix_joint_new
exp_name=joint_callhome_460_combine_fromscratch_energy005_rep_all_clean_noseonreal
other_args=
sample_rate=16000
code_path=/mnt/bn/junyi-nas2/codebase/joint-optimization
data_path=/mnt/bn/junyi-nas2/librimix

. ./parse_options.sh

export CUDA_VISIBLE_DEVICES=$gpu

exp_dir=${code_path}/exp_fairseq/joint
ts_vad_path=${code_path}/ts_vad
aux_path=${data_path}/ref_speech
spk_path=${data_path}/Libri23Mix/embed

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
  dataset.train_subset=train-460-callhome-all \
  dataset.valid_subset=callhome2-all \
  +task.spk_path=${spk_path} \
  +task.aux_path=${aux_path} \
  task.librimix_mode=max \
  model.use_usev_loss=true \
  model.combine_results=true \
  checkpoint.save_interval_updates=2500 \
  optimization.max_update=200000 \
  lr_scheduler.warmup_updates=20000 \
  dataset.max_tokens=75000 \
  checkpoint.best_checkpoint_metric=after_DER1 \
  optimization.update_freq=[2] \
  distributed_training.distributed_world_size=8 \
  +model.combine_after_updates=200000 \
  criterion.loss_weights.energy=0.005 \
  +task.librimix_type='clean' \
  model.use_usev_loss=false \
  ${other_args}
