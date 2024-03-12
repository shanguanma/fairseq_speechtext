#!/usr/bin/env bash


stage=0
stop_stage=1000
. utils/parse_options.sh

. path_for_fsq_sptt.sh
pip install https://github.com/kpu/kenlm/archive/master.zip --force-reinstall
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

## this script is display how to train Chinese GAN in SPL paper
##  i have remove tone from phone sequence, final the number of phone set is 59 include <SIL>
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
  fairseq_dir=/home/maduo/codebase/fairseq_speechtext
  des_dir=/home/maduo/dataset/format/Chinese/wenetspeech_no_silence

  ## it stores hubert large model 15 layer representation of raw librispeech speech,
  ## it  removes  silence. it offers feature of speech.
  TASK_DATA=$des_dir/hubert_large_feat_dir_no_silence
   ## offer mfcc pesudo label from without silence unpair audio
  #cp -r /mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir/mfcc_no_silence/mfcc_lab//* $TASK_DATA


  # Unpaired text input
  #text_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_again/
  text_dir=/home/maduo/dataset/format/Chinese/aishell-2_norm_phn_seq_remove_tone
  cp -r $text_dir/unpair_text/dict.phn.txt  $TASK_DATA
  TEXT_DATA=$text_dir/unpair_text
  ## in order to match unpair speech train set name
  mv $TEXT_DATA/train.bin  $TEXT_DATA/train_m.bin
  mv $TEXT_DATA/train.idx $TEXT_DATA/train_m.idx
  #KENLM_PATH=$text_dir/lm.phones.filtered.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)
  KENLM_PATH=$text_dir/lm.phones.filtered.04.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)
  config_dir=$fairseq_dir/examples/wav2vec/unsupervised

  dir=/home/maduo/exp/
  model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_md_setting_remove_tone
  exp_dir=$dir/wav2vec-u2_gan_from_scratch/${model_name}
  mkdir -p $exp_dir
  export PYTHONPATH=$fairseq_dir:$PYTHONPATH
  world_size=1
  update_freq=1
  #CUDA_VISIBLE_DEVICES=7  python $fairseq_dir/fairseq_cli/hydra_train.py  --multirun \
  #export TORCH_CUDNN_V8_API_DISABLED=1
  #export TORCH_CUDNN_V8_API_DISABLED=1
  #python $fairseq_dir/fairseq_cli/hydra_train.py\
  #CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=$world_size --master_port=12345  $fairseq_dir/fairseq_cli/hydra_train.py \
  python $fairseq_dir/fairseq_cli/hydra_train.py \
       --config-dir $config_dir/config/gan \
       --config-name w2vu2_local2_md \
       task.data=${TASK_DATA} \
       task.text_data=${TEXT_DATA} \
       task.kenlm_path=${KENLM_PATH} \
       dataset.train_subset=train_m\
       dataset.valid_subset=dev\
       dataset.batch_size=160\
        dataset.num_workers=6\
       common.user_dir=${fairseq_dir}/examples/wav2vec/unsupervised \
       common.seed=1\
       model.code_penalty=2 \
       model.gradient_penalty=1.5 \
       model.smoothness_weight=0.5\
        distributed_training.distributed_world_size=${world_size}\
       distributed_training.distributed_port=-1\
       distributed_training.ddp_backend=legacy_ddp\
       optimization.update_freq=[${update_freq}]\
       common.tensorboard_logdir=$exp_dir\
       hydra.run.dir=$exp_dir\
       hydra.job.name=$exp_dir/w2v_unsup_gan_xp
## 2024-2-2 : hltsz cluster
# one RTX3090
# about 3mins for 100 steps
# about 3.125 days for 150k steps 


fi
