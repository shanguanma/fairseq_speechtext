#!/bin/bash



stage=0
stop_stage=1000
. utils/parse_options.sh
#. path_for_fairseq_speechtext.sh
#. path_for_fsq_speechtext.sh
. path_for_fsq_sptt.sh
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1


## this script is display how to train Chinese GAN in SPL paper 
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
  fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
  des_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence

  ## it stores hubert large model 15 layer representation of raw librispeech speech,
  ## it  removes  silence. it offers feature of speech.
  TASK_DATA=$des_dir/hubert_large_feat_dir_no_silence
   ## offer mfcc pesudo label from without silence unpair audio
   cp -r /mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir/mfcc_no_silence/mfcc_lab//* $TASK_DATA 
  # Unpaired text input
  text_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq/
  TEXT_DATA=$text_dir/unpair_text 
  KENLM_PATH=$text_dir/lm.phones.filtered.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)
  config_dir=$fairseq_dir/examples/wav2vec/unsupervised

  dir=/mntnfs/lee_data1/maduo/exp/
  model_name=w2v_unsup_gan_xp_4gpu_8update_with_unpair_text_1M
  exp_dir=$dir/wav2vec-u2_gan_from_scratch/${model_name}
  mkdir -p $exp_dir
  export PYTHONPATH=$fairseq_dir:$PYTHONPATH
  world_size=1
  update_freq=1
  #CUDA_VISIBLE_DEVICES=7  python $fairseq_dir/fairseq_cli/hydra_train.py  --multirun \
  python $fairseq_dir/fairseq_cli/hydra_train.py\
       --config-dir $config_dir/config/gan \
     --config-name w2vu2_local3_md \
     task.data=${TASK_DATA} \
     task.text_data=${TEXT_DATA} \
     task.kenlm_path=${KENLM_PATH} \
     dataset.train_subset=train_m\
     dataset.valid_subset=dev\
     dataset.batch_size=160\
     dataset.num_workers=6\
     common.user_dir=${fairseq_dir}/examples/wav2vec/unsupervised \
     distributed_training.distributed_world_size=${world_size}\
     distributed_training.distributed_port=-1\
     distributed_training.ddp_backend=legacy_ddp\
     optimization.update_freq=[${update_freq}]\
     common.tensorboard_logdir=$exp_dir\
     hydra.run.dir=$fairseq_dir/examples/wav2vec/unsupervised\
     hydra.job.name=$exp_dir/w2v_unsup_gan_xp
fi
