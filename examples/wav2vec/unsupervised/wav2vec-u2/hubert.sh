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
   model_name=pretrain_on_hubert_4gpu_8update_960h_mfcc_250k_update   ## it actual runs 400k steps
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_dir
   world_size=4
   update_freq=8
   CUDA_VISIBLE_DEVICES=4,5,6,7 python $fairseq_dir/fairseq_cli/hydra_train.py \
            --config-dir $config_dir/config/pretrain \
            --config-name hubert_base_librispeech \
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

### i want to see only iter1(400k) pretrain model performance.
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
   echo "finetune on 80k steps on iter1 hubert model for train-100 of  librispeech"
   world_size=4  # total gpu number
   update_freq=1
   fairseq_dir=/workspace2/maduo/fairseq
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   label_dir=$tsv_dir/hubert_ltrlab ##postfix .ltr files folder, the utterances are coverted word into letter.
   config_dir=/workspace2/maduo/fairseq/examples/hubert/
   dir=/workspace2/maduo/exp
   model_name=pretrain_on_hubert_8gpu_4update_960h_mfcc_250k_update ##
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune
   mkdir -p $exp_finetune_dir
   python $fairseq_dir/fairseq_cli/hydra_train.py \
            --config-dir $config_dir/config/finetune \
            --config-name base_100h \
            task.data=$tsv_dir\
            task.label_dir=$label_dir\
            task.labels='["ltr"]' \
            model.w2v_path=$exp_dir/checkpoint_289_400000.pt\
            common.user_dir=$fairseq_dir/examples/hubert\
            dataset.train_subset=train-clean-100\
            dataset.valid_subset=dev-other\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_finetune_dir\
            checkpoint.save_dir=$exp_finetune_dir\
            hydra.run.dir=$fairseq_dir/examples\
            hydra.job.name=$exp_finetune_dir/finetune

fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
   echo "inference dev-clean dev-other test-clean test-other of librispeech using above finetune iter1 hubert"
   fairseq_dir=/workspace2/maduo/fairseq
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   label_dir=$tsv_dir/hubert_ltrlab
   config_dir=/workspace2/maduo/fairseq/examples/hubert/
   dir=/workspace2/maduo/exp
   model_name=pretrain_on_hubert_8gpu_4update_960h_mfcc_250k_update
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune
   mkdir -p $exp_finetune_dir/decode
   testsets='dev-clean dev-other test-clean test-other'
   for name in $testsets;do
     python $fairseq_dir/examples/speech_recognition/new/infer.py\
           --config-dir $config_dir/config/decode\
           --config-name infer_viterbi\
           task.data=$tsv_dir\
           task.label_dir=$label_dir\
           task.normalize=true\
           common_eval.results_path=$exp_finetune_dir/decode\
           common_eval.path=$exp_finetune_dir/checkpoint_best.pt\
           dataset.gen_subset=$name

   done
fi



if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
   echo "iter2: pretrain hubert on 6th layer transformer of hubert  pesudo label "
   echo "training on 400k steps for train-960 of librispeech"
   fairseq_dir=/workspace2/maduo/fairseq
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   label_dir=$tsv_dir/hubert_lab ##postfix .km files folder 
   config_dir=/workspace2/maduo/fairseq/examples/hubert/
   dir=/workspace2/maduo/exp
   model_name=pretrain_on_hubert_4gpu_8update_960h_hubert_400k_update
   exp_dir=$dir/pretrain/${model_name}
   world_size=4
   update_freq=8
   mkdir -p $exp_dir
   python $fairseq_dir/fairseq_cli/hydra_train.py \
            --config-dir $config_dir/config/pretrain \
            --config-name hubert_base_librispeech \
            task.data=$tsv_dir\
            task.label_dir=$label_dir\
            task.labels='["km"]' \
            model.label_rate=50\
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

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then
   echo "finetune on 80k steps on iter2 hubert model for train-100 of  librispeech"
   world_size=8  # total gpu number
   update_freq=1
   fairseq_dir=/workspace2/maduo/fairseq
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   label_dir=$tsv_dir/hubert_ltrlab ##postfix .ltr files folder, the utterances are coverted word into letter.
   config_dir=/workspace2/maduo/fairseq/examples/hubert/
   dir=/workspace2/maduo/exp
   model_name=pretrain_on_hubert_8gpu_4update_960h_mfcc_250k_update
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune
   mkdir -p $exp_finetune_dir
   python $fairseq_dir/fairseq_cli/hydra_train.py \
            --config-dir $config_dir/config/finetune \
            --config-name base_100h \
            task.data=$tsv_dir\
            task.label_dir=$label_dir\
            task.labels='["ltr"]' \
            model.w2v_path=$exp_dir/checkpoint_289_400000.pt\
            common.user_dir=$fairseq_dir/examples/hubert\
            dataset.train_subset=train-clean-100\
            dataset.valid_subset=dev-other\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_finetune_dir\
            checkpoint.save_dir=$exp_finetune_dir\
            hydra.run.dir=$fairseq_dir/examples\
            hydra.job.name=$exp_finetune_dir/finetune
            
fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ];then
   echo "inference dev-clean dev-other test-clean test-other of librispeech using above finetune iter2 hubert"
   fairseq_dir=/workspace2/maduo/fairseq
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   label_dir=$tsv_dir/hubert_ltrlab
   config_dir=/workspace2/maduo/fairseq/examples/hubert/
   dir=/workspace2/maduo/exp
   model_name=pretrain_on_hubert_8gpu_4update_960h_mfcc_250k_update
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune
   mkdir -p $exp_finetune_dir/decode
   testsets='dev-clean dev-other test-clean test-other'
   for name in $testsets;do
     python $fairseq_dir/examples/speech_recognition/new/infer.py\
           --config-dir $config_dir/config/decode\
           --config-name infer_viterbi\
           task.data=$tsv_dir\
           task.label_dir=$label_dir\
           task.normalize=true\
           common_eval.results_path=$exp_finetune_dir/decode\
           common_eval.path=$exp_finetune_dir/checkpoint_best.pt\
           dataset.gen_subset=$name
  
   done
fi


# in order to get iter2 hubert label
if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ];then
   echo "iter1: pretrain hubert on mfcc pesudo label "
   echo "training on 250k steps for train-960 of librispeech"
   fairseq_dir=/workspace2/maduo/fairseq
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   label_dir=$tsv_dir/mfcc/mfcc_lab # ##postfix .km files folder
   config_dir=/workspace2/maduo/fairseq/examples/hubert/
   dir=/workspace2/maduo/exp
   model_name=pretrain_on_hubert_4gpu_8update_960h_mfcc_250k_update_offical   ## offical setting
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_dir
   world_size=4
   update_freq=8
   CUDA_VISIBLE_DEVICES=0,5,6,7 python $fairseq_dir/fairseq_cli/hydra_train.py \
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
#RTX3090: 250K steps: about 5.2 days pretraining
##         every 200steps : about 6 minites

fi

