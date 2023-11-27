#!/usr/bin/env bash

stage=0

stop_stage=1000
.  utils/parse_options.sh
#. path_for_fairseq_speechtext.sh ## pytorch 2.0 fairseq flashnight-text flashnight sequence
. path_for_fsq_speechtext.sh
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then

   echo "pretrain hubert with llama on 9layer of offical hubert base model, clusters=500, label_rate=50 from offical hubert base model checkpoint"
   echo "training on 100k steps for train-360 of librispeech speech"
   
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech
   label_dir=$tsv_dir/feat_dir/9layer_feat_from_offical_hubert_base_model/label_dir/ # ##postfix *.km files folder
   #tsv_dir=$label_dir
   config_dir=$fairseq_dir/examples/hubert_with_llama/
   dir=/mntnfs/lee_data1/maduo/exp
   model_name=continue_pretain_base_hubert_with_llama_on_train_360
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_dir
   llama_path=/mntcephfs/lab_data/maduo/model_hub/OPT-LLM/lit-llama/7b/7B/lit-llama.pth
   model_hub=/mntcephfs/lab_data/maduo/model_hub/librispeech/hubert_base_librispeech_offical_no_finetune/
   #cp -r $model_hub/hubert_base_ls960.pt $exp_dir
   # rename
   #mv $exp_dir/hubert_base_ls960.pt $exp_dir/checkpoint_last.pt
   world_size=1
   update_freq=32
   python $fairseq_dir/fairseq_cli/hydra_train.py \
            --config-dir $config_dir/config/pretrain \
            --config-name hubert_with_llama_base_librispeech\
            task.data=$tsv_dir\
            task.label_dir=$label_dir\
            task.labels='["km"]' \
            model.label_rate=50\
            model.llama_path=$llama_path\
            model.hubert_path=$model_hub/hubert_base_ls960.pt\
            optimization.max_update=100000\
            common.user_dir=$fairseq_dir/examples/hubert_with_llama\
            dataset.train_subset=train-clean-360\
            dataset.valid_subset=\'dev-other,dev-clean\'\
            dataset.max_tokens=1400000\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_dir\
            checkpoint.save_dir=$exp_dir\
            hydra.run.dir=$fairseq_dir/examples/hubert_with_llama\
            hydra.job.chdir=True\
            hydra.job.name=$exp_dir/pretrain
fi

##
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then

   echo "pretrain hubert with llama on 9layer of offical hubert base model, clusters=500, label_rate=50 from offical hubert base model checkpoint"
   echo "training on 100k steps for train-360 of librispeech speech"

   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech
   label_dir=$tsv_dir/feat_dir/9layer_feat_from_offical_hubert_base_model/label_dir/ # ##postfix *.km files folder
   #tsv_dir=$label_dir
   config_dir=$fairseq_dir/examples/hubert_with_llama/
   dir=/mntnfs/lee_data1/maduo/exp
   model_name=continue_pretain_on_hubert_iter2_with_llama_on_train_360_lr5e_5
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_dir
   llama_path=/mntcephfs/lab_data/maduo/model_hub/OPT-LLM/lit-llama/7b/7B/lit-llama.pth
   model_hub=/mntcephfs/lab_data/maduo/model_hub/librispeech/hubert_base_librispeech_offical_no_finetune/
   #cp -r $model_hub/hubert_base_ls960.pt $exp_dir
   # rename
   #mv $exp_dir/hubert_base_ls960.pt $exp_dir/checkpoint_last.pt
   world_size=1
   update_freq=32
   lr=0.00005
   python $fairseq_dir/fairseq_cli/hydra_train.py \
            --config-dir $config_dir/config/pretrain \
            --config-name hubert_with_llama_base_librispeech\
            task.data=$tsv_dir\
            task.label_dir=$label_dir\
            task.labels='["km"]' \
            model.label_rate=50\
            model.llama_path=$llama_path\
            model.hubert_path=$model_hub/hubert_iter2.pt\
            optimization.max_update=50000\
            optimization.lr=[$lr]\
            common.user_dir=$fairseq_dir/examples/hubert_with_llama\
            dataset.train_subset=train-clean-360\
            dataset.valid_subset=\'dev-other,dev-clean\'\
            dataset.max_tokens=1400000\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_dir\
            checkpoint.save_dir=$exp_dir\
            hydra.run.dir=$exp_dir\
            hydra.job.chdir=True\
            hydra.job.name=pretrain
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then

   echo "pretrain hubert with llama on 9layer of offical hubert base model, clusters=500, label_rate=50 from offical hubert base model checkpoint"
   echo "training on 100k steps for train-360 of librispeech speech"

   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech
   label_dir=$tsv_dir/feat_dir/9layer_feat_from_offical_hubert_base_model/label_dir/ # ##postfix *.km files folder
   #tsv_dir=$label_dir
   config_dir=$fairseq_dir/examples/hubert_with_llama/
   dir=/mntnfs/lee_data1/maduo/exp
   model_name=continue_pretain_on_hubert_iter2_with_llama_on_train_360_ft_style
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_dir
   llama_path=/mntcephfs/lab_data/maduo/model_hub/OPT-LLM/lit-llama/7b/7B/lit-llama.pth
   model_hub=/mntcephfs/lab_data/maduo/model_hub/librispeech/hubert_base_librispeech_offical_no_finetune/
   #cp -r $model_hub/hubert_base_ls960.pt $exp_dir
   # rename
   #mv $exp_dir/hubert_base_ls960.pt $exp_dir/checkpoint_last.pt
   world_size=4
   update_freq=8
   python $fairseq_dir/fairseq_cli/hydra_train_for_with_llama.py \
            --config-dir $config_dir/config/pretrain \
            --config-name hubert_with_llama_base_librispeech_ft_style\
            task.data=$tsv_dir\
            task.label_dir=$label_dir\
            task.labels='["km"]' \
            model.label_rate=50\
            model.llama_path=$llama_path\
            model.hubert_path=$model_hub/hubert_iter2.pt\
            optimization.max_update=100000\
            common.user_dir=$fairseq_dir/examples/hubert_with_llama\
            dataset.train_subset=train-clean-360\
            dataset.valid_subset=\'dev-other,dev-clean\'\
            dataset.max_tokens=1400000\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_dir\
            checkpoint.save_dir=$exp_dir\
            hydra.run.dir=$exp_dir\
            hydra.job.chdir=True\
            hydra.job.name=pretrain
fi

