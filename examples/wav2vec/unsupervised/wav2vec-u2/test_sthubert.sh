#!/usr/bin/env bash

stage=0

stop_stage=1000
.  utils/parse_options.sh
. path_for_fairseq_speechtext.sh

export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
#export NCCL_DEBUG=INFO 
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
   echo "iter1: pretrain hubert on mfcc pesudo label "
   echo "training on 250k steps for train-960 of librispeech"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   label_dir=$tsv_dir/unpair_phncode_unpair_mfcc_km_code # ##postfix *.km *.phncode files folder
   config_dir=$fairseq_dir/examples/sthubert/
   dir=/workspace2/maduo/exp
   model_name=pretrain_on_sthubert_4gpu_8update_960h_mfcc_250k_update
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_dir
   world_size=4
   update_freq=8
   CUDA_VISIBLE_DEVICES=0,1,2,3 python $fairseq_dir/fairseq_cli/hydra_train.py \
            --config-dir $config_dir/config/pretrain \
            --config-name sthubert_small_base_librispeech \
            task.data=$tsv_dir\
            task.label_dir=$label_dir\
            task.labels='["km","phncode"]' \
            model.label_rate=100\
            common.user_dir=$fairseq_dir/examples/sthubert\
            dataset.train_subset=train-960\
            dataset.valid_subset=\'dev-other,dev-clean\'\
            dataset.max_tokens=1100000\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_dir\
            checkpoint.save_dir=$exp_dir\
            hydra.run.dir=$fairseq_dir/examples/hubert\
            hydra.job.name=$exp_dir/pretrain
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
   echo "fine tune sthubert model using train-clean-100 supervision data"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   config_dir=source_md/wav2vec-u2/
   dir=/workspace2/maduo/exp
   model_name=pretrain_on_sthubert_4gpu_8update_960h_mfcc_250k_update
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_finetune_dir
   world_size=4
   update_freq=2
   CUDA_VISIBLE_DEVICES=0,1,2,3 python $fairseq_dir/fairseq_cli/hydra_train.py \
       --config-dir $config_dir/config/finetune \
            --config-name sthubert_base_100h \
            task.data=$tsv_dir\
            task.label_dir=$tsv_dir\
            task.labels='["ltr"]' \
            model.w2v_path=$exp_dir/checkpoint_226_400000.pt\
            common.user_dir=$fairseq_dir/examples/sthubert\
            dataset.train_subset=train-clean-100\
            dataset.valid_subset=dev-other\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_finetune_dir\
            checkpoint.save_dir=$exp_finetune_dir\
            hydra.run.dir=$fairseq_dir/examples/sthubert\
            hydra.job.name=$exp_dir/finetune

fi



if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
   echo "inference sthubert model on dev-other, dev-clean, test-other, test-clean of librispeech"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   #config_dir=/workspace2/maduo/source_md/wav2vec-u2
   config_dir=$fairseq_dir/examples/sthubert/
   dir=/workspace2/maduo/exp
   model_name=pretrain_on_sthubert_4gpu_8update_960h_mfcc_250k_update
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune
   mkdir -p $exp_finetune_dir/decode_on_100h
   testsets="dev-clean dev-other test-clean test-other"
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   #cd $fairseq_dir
   for name in $testsets;do
   CUDA_VISIBLE_DEVICES=7       python $fairseq_dir/examples/speech_recognition/new/infer.py \
                --config-dir $config_dir/config/decode\
                --config-name sthubert_infer_viterbi\
                task.data=$tsv_dir\
                task.label_dir=$tsv_dir\
                task.normalize=true\
                common_eval.results_path=$exp_finetune_dir/decode_on_100h\
                common_eval.path=$exp_finetune_dir/checkpoint_best.pt\
                dataset.gen_subset=$name
    
   done 
   #result: mfcc iter:400k@226epochs, finetune:80k@222epchs
   # WER% is from $exp_finetune_dir/decode_on_100/viterbi/infer.log
   #  dev-clean   dev-other   test-clean   test-other
   #  12.71        26.39        12.99        26.44

fi


#### random from text: dataset_version2 of sthubert
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then
   echo "iter1: pretrain hubert on mfcc pesudo label "
   echo "training on 250k steps for train-960 of librispeech"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   label_dir=$tsv_dir/unpair_phncode_unpair_mfcc_km_code # ##postfix *.km *.phncode files folder
   config_dir=$fairseq_dir/examples/sthubert/
   dir=/workspace2/maduo/exp
   model_name=pretrain_on_sthubert_4gpu_8update_960h_mfcc_250k_update2
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_dir
   world_size=4
   update_freq=8
   #export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 ## every 200steps: about 15mins
   #export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 ## every 200steps: about 7.5mins,Tried to allocate 58.00 MiB  
   #export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:768
   CUDA_VISIBLE_DEVICES=0,1,2,3 python $fairseq_dir/fairseq_cli/hydra_train.py \
            --config-dir $config_dir/config/pretrain \
            --config-name sthubert_small_base_librispeech2_mfcc \
            task.data=$tsv_dir\
            task.label_dir=$label_dir\
            task.labels='["km","phncode"]' \
            model.label_rate=100\
            common.user_dir=$fairseq_dir/examples/sthubert\
            dataset.train_subset=train-960\
            dataset.valid_subset=\'dev-other,dev-clean\'\
            dataset.max_tokens=1400000\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_dir\
            checkpoint.save_dir=$exp_dir\
            hydra.run.dir=$fairseq_dir/examples/sthubert\
            hydra.job.name=$exp_dir/pretrain
#RTX3090: 250K steps: about 4.3 days pretraining
##         every 200steps : about 5 minites
fi
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ];then
   echo "fine tune sthubert model using train-clean-100 supervision data"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   config_dir=source_md/wav2vec-u2/
   dir=/workspace2/maduo/exp
   model_name=pretrain_on_sthubert_4gpu_8update_960h_mfcc_250k_update2
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_finetune_dir
   world_size=4
   update_freq=2
   CUDA_VISIBLE_DEVICES=0,1,2,3 python $fairseq_dir/fairseq_cli/hydra_train.py \
            --config-dir $config_dir/config/finetune \
            --config-name sthubert_base_100h \
            task.data=$tsv_dir\
            task.label_dir=$tsv_dir\
            task.labels='["ltr"]' \
            model.w2v_path=$exp_dir/checkpoint_193_250000.pt\
            common.user_dir=$fairseq_dir/examples/sthubert\
            dataset.train_subset=train-clean-100\
            dataset.valid_subset=dev-other\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_finetune_dir\
            checkpoint.save_dir=$exp_finetune_dir\
            hydra.run.dir=$fairseq_dir/examples/sthubert\
            hydra.job.name=$exp_dir/finetune
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ];then
   echo "inference sthubert model on dev-other, dev-clean, test-other, test-clean of librispeech"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   #config_dir=/workspace2/maduo/source_md/wav2vec-u2
   config_dir=$fairseq_dir/examples/sthubert/
   dir=/workspace2/maduo/exp
   model_name=pretrain_on_sthubert_4gpu_8update_960h_mfcc_250k_update2
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune
   mkdir -p $exp_finetune_dir/decode_on_100h
   testsets="dev-clean dev-other test-clean test-other"
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   #cd $fairseq_dir
   for name in $testsets;do
       python $fairseq_dir/examples/speech_recognition/new/infer.py \
                --config-dir $config_dir/config/decode\
                --config-name sthubert_infer_viterbi\
                task.data=$tsv_dir\
                task.label_dir=$tsv_dir\
                task.normalize=true\
                common_eval.results_path=$exp_finetune_dir/decode_on_100h\
                common_eval.path=$exp_finetune_dir/checkpoint_best.pt\
                dataset.gen_subset=$name

   done
   #result: mfcc iter:250k@181epochs, finetune:80k@222epchs
   # WER% is from $exp_finetune_dir/decode_on_100/viterbi/infer.log
   #  dev-clean   dev-other   test-clean   test-other
   # 14.40          29.42        14.75       29.87
 
fi

#
#### fixed from text and keep speech length of hubert offical settting: dataset_version3 of sthubert
### it is stoping, its result is not very useful
if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ];then
   echo "iter1: pretrain hubert on mfcc pesudo label "
   echo "training on 250k steps for train-960 of librispeech"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   label_dir=$tsv_dir/unpair_phncode_unpair_mfcc_km_code # ##postfix *.km *.phncode files folder
   config_dir=$fairseq_dir/examples/sthubert/
   dir=/workspace2/maduo/exp
   model_name=pretrain_on_sthubert_4gpu_8update_960h_mfcc_250k_update3
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_dir
   world_size=4
   update_freq=8
   CUDA_VISIBLE_DEVICES=0,5,6,7 python $fairseq_dir/fairseq_cli/hydra_train.py \
            --config-dir $config_dir/config/pretrain \
            --config-name sthubert_small_base_librispeech3_mfcc\
            task.data=$tsv_dir\
            task.label_dir=$label_dir\
            task.labels='["km","phncode"]' \
            model.label_rate=100\
            common.user_dir=$fairseq_dir/examples/sthubert\
            dataset.train_subset=train-960\
            dataset.valid_subset=\'dev-other,dev-clean\'\
            dataset.max_tokens=1400000\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_dir\
            checkpoint.save_dir=$exp_dir\
            hydra.run.dir=$fairseq_dir/examples/sthubert\
            hydra.job.name=$exp_dir/pretrain
fi

