#!/usr/bin/env bash

stage=0

stop_stage=1000
.  utils/parse_options.sh
. path_for_fairseq.sh

export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
   echo "iter1: pretrain hubert on mfcc pesudo label "
   echo "training on 250k steps for train-960 of librispeech"
   fairseq_dir=/workspace2/maduo/fairseq
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   label_dir=$tsv_dir/mfcc/mfcc_lab # ##postfix .km files folder
   config_dir=/workspace2/maduo/fairseq/examples/hubert/
   dir=/workspace2/maduo/exp
   model_name=pretrain_on_hubert_4gpu_8update_960h_mfcc_250k_update
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_dir
   world_size=4
   update_freq=8
   CUDA_VISIBLE_DEVICES=4,5,6,7 python $fairseq_dir/fairseq_cli/hydra_train.py \
            --config-dir $config_dir/config/pretrain \
            --config-name hubert_base_librispeech_mfcc \
            task.data=$tsv_dir\
            task.label_dir=$label_dir\
            task.labels='["km"]' \
            model.label_rate=100\
            common.user_dir=$fairseq_dir/examples/hubert\
            dataset.train_subset=train-960\
            dataset.valid_subset=\'dev-other,dev-clean\'\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_dir\
            checkpoint.save_dir=$exp_dir\
            hydra.run.dir=$fairseq_dir/examples/hubert\
            hydra.job.name=$exp_dir/pretrain
fi
