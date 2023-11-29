#!/usr/bin/env bash

stage=0

stop_stage=1000
.  utils/parse_options.sh
#. path_for_fairseq_speechtext.sh ## pytorch 2.0 fairseq flashnight-text flashnight sequence, python=3.9, for infer stage
. path_for_fsq_speechtext.sh ## python3.11 , lightning for pretrain and finetune 
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
   model_name=continue_pretain_on_hubert_iter2_with_llama_on_train_360_lr5e_4
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_dir
   llama_path=/mntcephfs/lab_data/maduo/model_hub/OPT-LLM/lit-llama/7b/7B/lit-llama.pth
   model_hub=/mntcephfs/lab_data/maduo/model_hub/librispeech/hubert_base_librispeech_offical_no_finetune/
   #cp -r $model_hub/hubert_base_ls960.pt $exp_dir
   # rename
   #mv $exp_dir/hubert_base_ls960.pt $exp_dir/checkpoint_last.pt
   world_size=2
   update_freq=16
   lr=0.0005
   python $fairseq_dir/fairseq_cli/hydra_train.py \
            --config-dir $config_dir/config/pretrain \
            --config-name hubert_with_llama_base_librispeech\
            task.data=$tsv_dir\
            task.label_dir=$label_dir\
            task.labels='["km"]' \
            model.label_rate=50\
            model.llama_path=$llama_path\
            model.hubert_path=$model_hub/hubert_iter2.pt\
            optimization.max_update=100000\
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


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
   echo "finetune llamahubert on 10h on 25k steps in letter ctc loss mode"
   echo "freeze llama layer layer in finetune mode"

   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp
   config_dir=$fairseq_dir/examples/hubert_with_llama
   model_name=continue_pretain_on_hubert_iter2_with_llama_on_train_360_ft_style
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune_2gpus  ## name is error, it actually fine tune on 10hours not 100hours
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_finetune_dir
   world_size=2
   update_freq=4
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   python $fairseq_dir/fairseq_cli/hydra_train.py \
            --config-dir $config_dir/config/finetune \
            --config-name base_10h \
            task.data=$tsv_dir\
            task.label_dir=$tsv_dir\
            task.labels='["ltr"]' \
            model.w2v_path=$exp_dir/checkpoint_192_100000.pt\
            common.user_dir=$fairseq_dir/examples/hubert_with_llama\
            dataset.train_subset=train-10h\
            dataset.valid_subset=\'dev-other\'\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_finetune_dir\
            checkpoint.save_dir=$exp_finetune_dir\
            hydra.run.dir=$exp_finetune_dir\
            hydra.job.name=finetune

fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then
   echo "inference llamahubert  model on dev-other, dev-clean, test-other, test-clean of librispeech"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp

   #fairseq_dir=/workspace2/maduo/fairseq_speechtext
   #tsv_dir=/workspace2/maduo/dataset/format/librispeech
   config_dir=$fairseq_dir/examples/hubert_with_llama
   #dir=/workspace2/maduo/exp

   model_name=continue_pretain_on_hubert_iter2_with_llama_on_train_360_ft_style
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune_2gpus
   #results_path=$exp_finetune_dir/decode_on_100h
   results_path=$exp_finetune_dir/decode_normalize_false
   mkdir -p $results_path
   testsets="dev-clean dev-other test-clean test-other"
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH

   for name in $testsets;do
     python $fairseq_dir/examples/speech_recognition/new/infer.py \
                --config-dir $config_dir/config/decode\
                --config-name infer_viterbi_librispeech\
                task.data=$tsv_dir\
                task.label_dir=$tsv_dir\
                task.normalize=false\
                common_eval.results_path=$results_path\
                common_eval.path=$exp_finetune_dir/checkpoint_best.pt\
                dataset.gen_subset=$name

   done
   # (fairseq_speechtext) [maduo@pbcmlg01 maduo]$ grep -rn 'Word error rate' logs/hubert_with_llama_base_local_cluster_stage3_hubert_iter2_ft_style_infer.log
   # dev-clean dev-other test-clean  test-other
   # 9.4886    15.8154    9.4907       16.3349

fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ];then
   echo "inference llamahubert  model on dev-other, dev-clean, test-other, test-clean of librispeech with kenlm"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp
   #config_dir=$fairseq_dir/examples/voicelm/
   #fairseq_dir=/workspace2/maduo/fairseq_speechtext
   #tsv_dir=/workspace2/maduo/dataset/format/librispeech
   config_dir=$fairseq_dir/examples/hubert_with_llama
   #dir=/workspace2/maduo/exp
   model_name=continue_pretain_on_hubert_iter2_with_llama_on_train_360_ft_style
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune_2gpus

   results_path=$exp_finetune_dir/decode_with_kenlm
   mkdir -p $results_path
   path_to_lexicon=/mntcephfs/lab_data/maduo/datasets/librispeech/kenlm_files/librispeech_lexicon.lst #word2letter
   path_to_lm=/mntcephfs/lab_data/maduo/datasets/librispeech/kenlm_files/4-gram.arpa  ## word lm
   testsets="dev-clean dev-other test-clean test-other"
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH

   for name in $testsets;do
       python $fairseq_dir/examples/speech_recognition/new/infer.py \
                --config-dir $config_dir/config/decode\
                --config-name infer_kenlm_lirispeech\
                task.data=$tsv_dir\
                task.label_dir=$tsv_dir\
                task.normalize=false\
                common_eval.results_path=$results_path\
                common_eval.path=$exp_finetune_dir/checkpoint_best.pt\
                dataset.gen_subset=$name\
                decoding.type=kenlm\
                decoding.lexicon=$path_to_lexicon\
                decoding.lmpath=$path_to_lm\
                decoding.nbest=1\
                decoding.beam=1500 \
                decoding.lmweight=2 \
                decoding.wordscore=-1 \
                decoding.beamthreshold=100\
                common_eval.quiet=false

   done
   # grep -rn 'Word error rate'  logs/hubert_with_llama_base_local_cluster_stage3_hubert_iter2_ft_style_infer_kenlm.log
   # dev-clean dev-other test-clean test-other
   # 3.9318     8.9412    4.3240    9.4952 
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ];then

   echo "pretrain hubert with llama on 9layer of offical hubert base model, clusters=500, label_rate=50 from offical hubert base model checkpoint"
   echo "training on 100k steps for train-360 of librispeech speech"

   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech
   label_dir=$tsv_dir/feat_dir/9layer_feat_from_offical_hubert_base_model/label_dir/ # ##postfix *.km files folder
   #tsv_dir=$label_dir
   config_dir=$fairseq_dir/examples/hubert_with_llama/
   dir=/mntnfs/lee_data1/maduo/exp
   model_name=continue_pretain_on_hubert_iter2_with_llama_on_train_360_ft_style_freeze_first8layers
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
            +model.freeze_hubert_layer_nums=8\
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

