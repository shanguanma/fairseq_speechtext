#!/usr/bin/env bash
  
stage=0

stop_stage=1000
.  utils/parse_options.sh
. path_for_fairseq_speechtext.sh ## pytorch 2.0 fairseq flashnight-text flashnight sequence 

export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
### for debug one stage0-5
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
   echo "iter1: pretrain ILS-SSL on mfcc pesudo label "
   echo "training on 250k steps for train-960 of librispeech"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   label_dir=$tsv_dir/mfcc/mfcc_lab # ##postfix .km files folder
   config_dir=$fairseq_dir/examples/ILS-SSL
   dir=/workspace2/maduo/exp
   model_name=pretrain_on_small_ils-ssl_4gpu_8update_960h_mfcc_250k_update
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_dir
   world_size=4
   update_freq=8
   CUDA_VISIBLE_DEVICES=4,5,6,7 python $fairseq_dir/fairseq_cli/hydra_train.py \
            --config-dir $config_dir/config/pretrain \
            --config-name ILS-SSL_small_base_librispeech_mfcc \
            task.data=$tsv_dir\
            task.label_dir=$label_dir\
            task.labels='["km"]' \
            model.label_rate=100\
            common.user_dir=$fairseq_dir/examples/ILS-SSL\
            dataset.train_subset=train-960\
            dataset.valid_subset=\'dev-other,dev-clean\'\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_dir\
            checkpoint.save_dir=$exp_dir\
            hydra.run.dir=$fairseq_dir/examples/ILS-SSL\
            hydra.job.name=$exp_dir/pretrain
### 4RTX3090: training about 11.5 day
###           400steps: about 8 minites
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
   echo "fine tune small ils-ssl model(trained on mfcc) using train-clean-100 supervision data"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   config_dir=$fairseq_dir/examples/ILS-SSL
   dir=/workspace2/maduo/exp
   model_name=pretrain_on_small_ils-ssl_4gpu_8update_960h_mfcc_250k_update
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_finetune_dir
   world_size=4
   update_freq=2
   CUDA_VISIBLE_DEVICES=4,5,6,7 python $fairseq_dir/fairseq_cli/hydra_train.py \
       --config-dir $config_dir/config/finetune \
            --config-name ILS-SSL_base_100h \
            task.data=$tsv_dir\
            task.label_dir=$tsv_dir\
            task.labels='["ltr"]' \
            model.w2v_path=$exp_dir/checkpoint_last.pt\
            common.user_dir=$fairseq_dir/examples/ILS-SSL\
            dataset.train_subset=train-clean-100\
            dataset.valid_subset=dev-other\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_finetune_dir\
            checkpoint.save_dir=$exp_finetune_dir\
            hydra.run.dir=$fairseq_dir/examples/ILS-SSL\
            hydra.job.name=$exp_dir/finetune
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then
   echo "inference ils-ssl  model on dev-other, dev-clean, test-other, test-clean of librispeech"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   config_dir=$fairseq_dir/examples/hubert/
   dir=/workspace2/maduo/exp
   model_name=pretrain_on_small_hubert_4gpu_8update_960h_mfcc_250k_update
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune
   mkdir -p $exp_finetune_dir/decode_on_100h
   testsets="dev-clean dev-other test-clean test-other"
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH

   for name in $testsets;do
       python $fairseq_dir/examples/speech_recognition/new/infer.py \
                --config-dir $config_dir/config/decode\
                --config-name infer_viterbi\
                task.data=$tsv_dir\
                task.label_dir=$tsv_dir\
                task.normalize=true\
                common_eval.results_path=$exp_finetune_dir/decode_on_100h\
                common_eval.path=$exp_finetune_dir/checkpoint_best.pt\
                dataset.gen_subset=$name

   done
   #result: mfcc iter:400k@289epochs, finetune:80k@222epchs
   # WER% is from $exp_finetune_dir/decode_on_100/viterbi/infer.log
   #  dev-clean   dev-other   test-clean   test-other


fi

### using mfcc feature to train base model 
if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ];then
   echo "iter1: pretrain ILS-SSL on mfcc pesudo label "
   echo "training on 250k steps for train-960 of librispeech"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   label_dir=$tsv_dir/mfcc/mfcc_lab # ##postfix .km files folder
   config_dir=$fairseq_dir/examples/ILS-SSL
   dir=/workspace2/maduo/exp
   model_name=pretrain_on_base_ils-ssl_4gpu_8update_960h_mfcc_250k_update
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_dir
   world_size=4
   update_freq=8
   CUDA_VISIBLE_DEVICES=4,5,6,7 python $fairseq_dir/fairseq_cli/hydra_train.py \
            --config-dir $config_dir/config/pretrain \
            --config-name ILS-SSL_base_librispeech_mfcc \
            task.data=$tsv_dir\
            task.label_dir=$label_dir\
            task.labels='["km"]' \
            model.label_rate=100\
            common.user_dir=$fairseq_dir/examples/ILS-SSL\
            dataset.train_subset=train-960\
            dataset.valid_subset=\'dev-other,dev-clean\'\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_dir\
            checkpoint.save_dir=$exp_dir\
            hydra.run.dir=$fairseq_dir/examples/ILS-SSL\
            hydra.job.name=$exp_dir/pretrain

fi




if [ ${stage} -le 16 ] && [ ${stop_stage} -ge 16 ];then
   echo "iter2: pretrain ILS-SSL on ILS-SSL pesudo label "
   echo "training on 400k steps for train-960 of librispeech"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   label_dir=$tsv_dir/ils-ssl/ils-ssl_lab # # get 6 layer representation of ils-ssl model 
                                          ### and k-means them ,then store postfix .km files folder
   config_dir=$fairseq_dir/examples/ILS-SSL
   dir=/workspace2/maduo/exp
   model_name=pretrain_on_base_ils-ssl_4gpu_8update_960h_ils-ssl_400k_update
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_dir
   world_size=4
   update_freq=8
   CUDA_VISIBLE_DEVICES=4,5,6,7 python $fairseq_dir/fairseq_cli/hydra_train.py \
            --config-dir $config_dir/config/pretrain \
            --config-name ILS-SSL_base_librispeech \
            task.data=$tsv_dir\
            task.label_dir=$label_dir\
            task.labels='["km"]' \
            model.label_rate=50\
            common.user_dir=$fairseq_dir/examples/ILS-SSL\
            dataset.train_subset=train-960\
            dataset.valid_subset=\'dev-other,dev-clean\'\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_dir\
            checkpoint.save_dir=$exp_dir\
            hydra.run.dir=$fairseq_dir/examples/ILS-SSL\
            hydra.job.name=$exp_dir/pretrain
### 4RTX3090: training about 11.5 day
###           400steps: about 8 minites
fi



if [ ${stage} -le 17 ] && [ ${stop_stage} -ge 17 ];then
   echo "fine tune small ils-ssl model(trained on ils-ssl specify layer representation) using train-clean-100 supervision data"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   config_dir=$fairseq_dir/examples/ILS-SSL
   dir=/workspace2/maduo/exp
   model_name=pretrain_on_base_ils-ssl_4gpu_8update_960h_ils-ssl_400k_update
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_finetune_dir
   world_size=4
   update_freq=2
   CUDA_VISIBLE_DEVICES=4,5,6,7 python $fairseq_dir/fairseq_cli/hydra_train.py \
       --config-dir $config_dir/config/finetune \
            --config-name ILS-SSL_base_100h \
            task.data=$tsv_dir\
            task.label_dir=$tsv_dir\
            task.labels='["ltr"]' \
            model.w2v_path=$exp_dir/checkpoint_last.pt\
            common.user_dir=$fairseq_dir/examples/ILS-SSL\
            dataset.train_subset=train-clean-100\
            dataset.valid_subset=dev-other\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_finetune_dir\
            checkpoint.save_dir=$exp_finetune_dir\
            hydra.run.dir=$fairseq_dir/examples/ILS-SSL\
            hydra.job.name=$exp_dir/finetune
fi

if [ ${stage} -le 18 ] && [ ${stop_stage} -ge 18 ];then
   echo "inference ils-ssl  model on dev-other, dev-clean, test-other, test-clean of librispeech"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   config_dir=$fairseq_dir/examples/hubert/
   dir=/workspace2/maduo/exp
   model_name=pretrain_on_base_ils-ssl_4gpu_8update_960h_ils-ssl_400k_update
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune
   mkdir -p $exp_finetune_dir/decode_on_100h
   testsets="dev-clean dev-other test-clean test-other"
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH

   for name in $testsets;do
       python $fairseq_dir/examples/speech_recognition/new/infer.py \
                --config-dir $config_dir/config/decode\
                --config-name infer_viterbi\
                task.data=$tsv_dir\
                task.label_dir=$tsv_dir\
                task.normalize=true\
                common_eval.results_path=$exp_finetune_dir/decode_on_100h\
                common_eval.path=$exp_finetune_dir/checkpoint_best.pt\
                dataset.gen_subset=$name

   done
   #result: mfcc iter:400k@289epochs, finetune:80k@222epchs
   # WER% is from $exp_finetune_dir/decode_on_100/viterbi/infer.log
   #  dev-clean   dev-other   test-clean   test-other


fi


