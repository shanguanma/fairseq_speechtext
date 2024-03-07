#!/usr/bin/env bash

stage=0

stop_stage=1000
.  utils/parse_options.sh
. path_for_fairseq_speechtext.sh ## pytorch 2.0 fairseq flashnight-text flashnight sequence, python=3.9, for infer stage
#. path_for_fsq_speechtext.sh ## python3.11 , lightning for pretrain and finetune 
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
export OC_CAUSE=1

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


if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ];then
   echo "finetune llamahubert on 10h on 25k steps in letter ctc loss mode"
   echo "freeze llama layer layer in finetune mode"

   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp
   config_dir=$fairseq_dir/examples/hubert_with_llama
   model_name=continue_pretain_on_hubert_iter2_with_llama_on_train_360_ft_style_freeze_first8layers
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
            model.w2v_path=$exp_dir/backup/checkpoint_48_50000.pt\
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

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ];then
   echo "inference llamahubert  model on dev-other, dev-clean, test-other, test-clean of librispeech"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp

   #fairseq_dir=/workspace2/maduo/fairseq_speechtext
   #tsv_dir=/workspace2/maduo/dataset/format/librispeech
   config_dir=$fairseq_dir/examples/hubert_with_llama
   #dir=/workspace2/maduo/exp

   model_name=continue_pretain_on_hubert_iter2_with_llama_on_train_360_ft_style_freeze_first8layers
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
   #  grep -rn 'Word error rate' logs/hubert_with_llama_base_local_cluster_stage3_hubert_iter2_ft_style_freeze_first8layers_infer.log
   # dev-clean dev-other test-clean test-other
   # 9.5125     15.6368   9.4356    16.2604

fi

if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ];then
   echo "finetune llamahubert on 10h on 25k steps in letter ctc loss mode"
   echo "freeze llama layer layer in finetune mode"

   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp
   config_dir=$fairseq_dir/examples/hubert_with_llama
   model_name=continue_pretain_on_hubert_iter2_with_llama_on_train_360_ft_style_freeze_first8layers
   exp_finetune_dir=$dir/finetune/${model_name}_10h_asr_finetune_2gpus_with_checkpoint_best  
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
            model.w2v_path=$exp_dir/checkpoint_best.pt\
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

if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ];then
   echo "inference llamahubert  model on dev-other, dev-clean, test-other, test-clean of librispeech"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp

   #fairseq_dir=/workspace2/maduo/fairseq_speechtext
   #tsv_dir=/workspace2/maduo/dataset/format/librispeech
   config_dir=$fairseq_dir/examples/hubert_with_llama
   #dir=/workspace2/maduo/exp

   model_name=continue_pretain_on_hubert_iter2_with_llama_on_train_360_ft_style_freeze_first8layers
   exp_finetune_dir=$dir/finetune/${model_name}_10h_asr_finetune_2gpus_with_checkpoint_best
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
   # grep -rn 'Word error rate' logs/hubert_with_llama_base_local_cluster_stage3_hubert_iter2_ft_style_freeze_first8layers_best_ckpt_infer.log
   # dev-clean dev-other test-clean test-other
   # 9.4059     15.4856   9.4527    16.0158
   
fi

if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ];then
   echo "finetune llamahubert on 10h on 25k steps in letter ctc loss mode"
   echo "freeze llama layer  and freeze first 8 layers of  hubert in finetune mode"

   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp
   config_dir=$fairseq_dir/examples/hubert_with_llama
   model_name=continue_pretain_on_hubert_iter2_with_llama_on_train_360_ft_style_freeze_first8layers
   exp_finetune_dir=$dir/finetune/${model_name}_10h_asr_finetune_2gpus_with_checkpoint_best_freeze_hubert_also
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_finetune_dir
   world_size=2
   update_freq=4
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   python $fairseq_dir/fairseq_cli/hydra_train_for_with_llama.py \
            --config-dir $config_dir/config/finetune \
            --config-name base_10h \
            task.data=$tsv_dir\
            task.label_dir=$tsv_dir\
            task.labels='["ltr"]' \
            model.w2v_path=$exp_dir/checkpoint_best.pt\
            +model.freeze_hubert_layer_nums=8\
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

if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ];then
   echo "inference llamahubert  model on dev-other, dev-clean, test-other, test-clean of librispeech"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp

   #fairseq_dir=/workspace2/maduo/fairseq_speechtext
   #tsv_dir=/workspace2/maduo/dataset/format/librispeech
   config_dir=$fairseq_dir/examples/hubert_with_llama
   #dir=/workspace2/maduo/exp

   model_name=continue_pretain_on_hubert_iter2_with_llama_on_train_360_ft_style_freeze_first8layers
   exp_finetune_dir=$dir/finetune/${model_name}_10h_asr_finetune_2gpus_with_checkpoint_best_freeze_hubert_also
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

   fi

## result is bad, so I will not run infer stage
if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ];then
   echo "finetune llamahubert on 10h on 25k steps in letter ctc loss mode"
   echo "freeze llama layer layer in finetune mode"

   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp
   config_dir=$fairseq_dir/examples/hubert_with_llama
   model_name=continue_pretain_on_hubert_iter2_with_llama_on_train_360_ft_style_freeze_first8layers
   exp_finetune_dir=$dir/finetune/${model_name}_10h_asr_finetune_2gpus_with_checkpoint_96_100000
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
            model.w2v_path=$exp_dir/checkpoint_96_100000.pt\
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

if [ ${stage} -le 20 ] && [ ${stop_stage} -ge 20 ];then

   echo "pretrain hubert with llama on 9layer of offical hubert base model, clusters=500, label_rate=50 from offical hubert base model checkpoint"
   echo "training on 100k steps for train-360 of librispeech speech"
   echo "using lora mlp of llama"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech
   label_dir=$tsv_dir/feat_dir/9layer_feat_from_offical_hubert_base_model/label_dir/ # ##postfix *.km files folder
   #tsv_dir=$label_dir
   config_dir=$fairseq_dir/examples/hubert_with_llama/
   dir=/mntnfs/lee_data1/maduo/exp
   model_name=continue_pretain_on_hubert_iter2_with_llama_on_train_360_ft_style_freeze_first8layers_lora_mlp
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
            +model.lora_r=16\
            +model.add_mlp_lora=true\
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
            hydra.job.name=pretrain
            #hydra.job.chdir=True\
            #hydra.job.name=pretrain
fi

if [ ${stage} -le 21 ] && [ ${stop_stage} -ge 21 ];then
   echo "finetune llamahubert on 10h on 25k steps in letter ctc loss mode"
   echo "only update llama lora mlp  layer in llama network in finetune mode"

   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp
   config_dir=$fairseq_dir/examples/hubert_with_llama
   model_name=continue_pretain_on_hubert_iter2_with_llama_on_train_360_ft_style_freeze_first8layers_lora_mlp
   exp_finetune_dir=$dir/finetune/${model_name}_10h_asr_finetune_2gpus  
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_finetune_dir
   world_size=2
   update_freq=4
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   python $fairseq_dir/fairseq_cli/hydra_train_for_with_llama.py \
            --config-dir $config_dir/config/finetune \
            --config-name base_10h \
            task.data=$tsv_dir\
            task.label_dir=$tsv_dir\
            task.labels='["ltr"]' \
            model.w2v_path=$exp_dir/checkpoint_192_100000.pt\
            +model.finetune_llama_model='lora_mlp'\
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


if [ ${stage} -le 22 ] && [ ${stop_stage} -ge 22 ];then
   echo "inference llamahubert  model on dev-other, dev-clean, test-other, test-clean of librispeech"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp

   #fairseq_dir=/workspace2/maduo/fairseq_speechtext
   #tsv_dir=/workspace2/maduo/dataset/format/librispeech
   config_dir=$fairseq_dir/examples/hubert_with_llama
   #dir=/workspace2/maduo/exp

   model_name=continue_pretain_on_hubert_iter2_with_llama_on_train_360_ft_style_freeze_first8layers_lora_mlp
   exp_finetune_dir=$dir/finetune/${model_name}_10h_asr_finetune_2gpus
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
   # grep -rn 'Word error rate:'  exp/finetune/continue_pretain_on_hubert_iter2_with_llama_on_train_360_ft_style_freeze_first8layers_lora_mlp_10h_asr_finetune_2gpus/decode_normalize_false/viterbi/infer.log 
#8361:[2024-01-21 20:26:03,323][__main__][INFO] - Word error rate: 10.5713
#17193:[2024-01-21 20:27:40,862][__main__][INFO] - Word error rate: 18.5655
#25290:[2024-01-21 20:29:22,419][__main__][INFO] - Word error rate: 10.5561
#34356:[2024-01-21 20:31:01,871][__main__][INFO] - Word error rate: 19.1070
fi


if [ ${stage} -le 23 ] && [ ${stop_stage} -ge 23 ];then
   echo "finetune llamahubert on 10h on 25k steps in letter ctc loss mode"
   echo "only update llama lora mlp  layer in llama network in finetune mode"

   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp
   config_dir=$fairseq_dir/examples/hubert_with_llama
   model_name=continue_pretain_on_hubert_iter2_with_llama_on_train_360_ft_style_freeze_first8layers_lora_mlp
   exp_finetune_dir=$dir/finetune/${model_name}_10h_asr_finetune_2gpus_checkpoint_best
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_finetune_dir
   world_size=2
   update_freq=4
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   python $fairseq_dir/fairseq_cli/hydra_train_for_with_llama.py \
            --config-dir $config_dir/config/finetune \
            --config-name base_10h \
            task.data=$tsv_dir\
            task.label_dir=$tsv_dir\
            task.labels='["ltr"]' \
            model.w2v_path=$exp_dir/checkpoint_best.pt\
            +model.finetune_llama_model='lora_mlp'\
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
if [ ${stage} -le 24 ] && [ ${stop_stage} -ge 24 ];then
   echo "inference llamahubert  model on dev-other, dev-clean, test-other, test-clean of librispeech"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp

   #fairseq_dir=/workspace2/maduo/fairseq_speechtext
   #tsv_dir=/workspace2/maduo/dataset/format/librispeech
   config_dir=$fairseq_dir/examples/hubert_with_llama
   #dir=/workspace2/maduo/exp

   model_name=continue_pretain_on_hubert_iter2_with_llama_on_train_360_ft_style_freeze_first8layers_lora_mlp
   exp_finetune_dir=$dir/finetune/${model_name}_10h_asr_finetune_2gpus_checkpoint_best
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
   # grep -rn 'Word error rate:' exp/finetune/continue_pretain_on_hubert_iter2_with_llama_on_train_360_ft_style_freeze_first8layers_lora_mlp_10h_asr_finetune_2gpus_checkpoint_best/decode_normalize_false/viterbi/infer.log 
#8361:[2024-01-22 09:17:24,645][__main__][INFO] - Word error rate: 10.2092
#17193:[2024-01-22 09:18:58,508][__main__][INFO] - Word error rate: 18.1376
#25290:[2024-01-22 09:20:37,885][__main__][INFO] - Word error rate: 10.2631
#34356:[2024-01-22 09:22:16,095][__main__][INFO] - Word error rate: 18.7364

fi



if [ ${stage} -le 30 ] && [ ${stop_stage} -ge 30 ];then
   echo "finetune llamahubert on 10h on 25k steps in letter ctc loss mode"
   echo "only update llama lora mlp  layer in llama network in finetune mode"
   echo "freeze first  8layer of hubert, this setting is same as pretrain setting"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp
   config_dir=$fairseq_dir/examples/hubert_with_llama
   model_name=continue_pretain_on_hubert_iter2_with_llama_on_train_360_ft_style_freeze_first8layers_lora_mlp
   exp_finetune_dir=$dir/finetune/${model_name}_10h_asr_finetune_1gpus_freeze_first8layer
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_finetune_dir
   world_size=1
   update_freq=8
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   python $fairseq_dir/fairseq_cli/hydra_train_for_with_llama.py \
            --config-dir $config_dir/config/finetune \
            --config-name base_10h \
            task.data=$tsv_dir\
            task.label_dir=$tsv_dir\
            task.labels='["ltr"]' \
            model.w2v_path=$exp_dir/checkpoint_best.pt\
            +model.freeze_hubert_layer_nums=8\
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

if [ ${stage} -le 31 ] && [ ${stop_stage} -ge 31 ];then
   echo "inference llamahubert  model on dev-other, dev-clean, test-other, test-clean of librispeech"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp

   #fairseq_dir=/workspace2/maduo/fairseq_speechtext
   #tsv_dir=/workspace2/maduo/dataset/format/librispeech
   config_dir=$fairseq_dir/examples/hubert_with_llama
   #dir=/workspace2/maduo/exp

   model_name=continue_pretain_on_hubert_iter2_with_llama_on_train_360_ft_style_freeze_first8layers_lora_mlp
   exp_finetune_dir=$dir/finetune/${model_name}_10h_asr_finetune_1gpus_freeze_first8layer
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
   #  grep -rn 'Word error rate:' logs/hubert_with_llama_base_local_cluster_stage3_hubert_iter2_ft_style_freeze_first8layers_four_gpus_lora_mlp_one_gpu_freeze_first_8_layer_hubert_on_infer.log
#8133:[2024-01-25 09:38:20,131][__main__][INFO] - Word error rate: 11.2588
#16736:[2024-01-25 09:39:59,014][__main__][INFO] - Word error rate: 19.3526
#24604:[2024-01-25 09:41:42,585][__main__][INFO] - Word error rate: 11.4064
#33441:[2024-01-25 09:43:24,766][__main__][INFO] - Word error rate: 19.6573
fi




if [ ${stage} -le 32 ] && [ ${stop_stage} -ge 32 ];then
    echo "inference  model on dev-other, dev-clean, test-other, test-clean of librispeech"
    fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp

   #fairseq_dir=/workspace2/maduo/fairseq_speechtext
   #tsv_dir=/workspace2/maduo/dataset/format/librispeech
   config_dir=$fairseq_dir/examples/hubert_with_llama
   #dir=/workspace2/maduo/exp

   model_name=continue_pretain_on_hubert_iter2_with_llama_on_train_360_ft_style_freeze_first8layers_lora_mlp
   exp_finetune_dir=$dir/finetune/${model_name}_10h_asr_finetune_1gpus_freeze_first8layer
   #results_path=$exp_finetune_dir/decode_on_100h
   results_path=$exp_finetune_dir/decode_normalize_false_audio_pretraining_again
   dict_path=/mntcephfs/lab_data/maduo/datasets/format/librispeech/dict.ltr.txt
   mkdir -p $results_path
   #testsets="test-clean10"
   #testsets="test-clean1"
   #testsets="dev-clean10"i
   testsets="dev-clean dev-other test-clean test-other"
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   #cd $fairseq_dir
   for name in $testsets;do
    python $fairseq_dir/examples/speech_recognition/new/infer_simple.py\
          $tsv_dir --task audio_pretraining\
          --nbest 1 --path $exp_finetune_dir/checkpoint_best.pt\
          --gen-subset ${name}\
          --results-path ${results_path} \
          --w2l-decoder viterbi \
          --word-score -1 --sil-weight 0 \
          --criterion ctc --max-tokens 1100000\
          --lexicon $dict_path \
          --label_dir $tsv_dir\
          --post-process letter
   done

   # grep -rn 'WER:' logs/hubert_with_llama_base_local_cluster_stage3_hubert_iter2_ft_style_freeze_first8layers_four_gpus_lora_mlp_one_gpu_freeze_first_8_layer_hubert_on_
#infer_simple.log
#8157:2024-01-25 10:20:56 | INFO | root | WER: 11.194441380831588
#16799:2024-01-25 10:23:38 | INFO | root | WER: 19.404098296302113
#24707:2024-01-25 10:26:21 | INFO | root | WER: 11.3549908703591
#33575:2024-01-25 10:29:08 | INFO | root | WER: 19.62057963815601
fi



## 2024-1-26 start move llama layer into medium layer of hubert 
if [ ${stage} -le 40 ] && [ ${stop_stage} -ge 40 ];then

   echo "pretrain hubert with llama on 9layer of offical hubert base model, clusters=500, label_rate=50 from offical hubert base model checkpoint"
   echo "training on 100k steps for train-360 of librispeech speech"
   echo "I will don't use lora mlp of llama"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech
   label_dir=$tsv_dir/feat_dir/9layer_feat_from_offical_hubert_base_model/label_dir/ # ##postfix *.km files folder
   #tsv_dir=$label_dir
   config_dir=$fairseq_dir/examples/hubert_with_llama/
   dir=/mntnfs/lee_data1/maduo/exp
   model_name=continue_pretain_on_hubert_iter2_with_llama_on_train_360_ft_style_freeze_first6hubertlayers_llama_in_medium_position
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_dir
   llama_path=/mntcephfs/lab_data/maduo/model_hub/OPT-LLM/lit-llama/7b/7B/lit-llama.pth
   model_hub=/mntcephfs/lab_data/maduo/model_hub/librispeech/hubert_base_librispeech_offical_no_finetune/
   #cp -r $model_hub/hubert_base_ls960.pt $exp_dir
   # rename
   #mv $exp_dir/hubert_base_ls960.pt $exp_dir/checkpoint_last.pt
   world_size=2
   update_freq=16
   
   ## debug
   #world_size=1
   #update_freq=32
   python $fairseq_dir/fairseq_cli/hydra_train_for_with_llama.py \
            --config-dir $config_dir/config/pretrain \
            --config-name hubert_with_llama_v2_base_librispeech_ft_style\
            task.data=$tsv_dir\
            task.label_dir=$label_dir\
            task.labels='["km"]' \
            model.label_rate=50\
            model.llama_path=$llama_path\
            model.hubert_path=$model_hub/hubert_iter2.pt\
            +model.freeze_hubert_layer_nums=6\
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
            hydra.job.name=pretrain
   fi



if [ ${stage} -le 41 ] && [ ${stop_stage} -ge 41 ];then
   echo "finetune llamahubert on 10h on 25k steps in letter ctc loss mode"
   echo "freeze llama layer  and freeze first 8 layers of  hubert in finetune mode"

   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp
   config_dir=$fairseq_dir/examples/hubert_with_llama
   model_name=continue_pretain_on_hubert_iter2_with_llama_on_train_360_ft_style_freeze_first6hubertlayers_llama_in_medium_position
   exp_finetune_dir=$dir/finetune/${model_name}_10h_asr_finetune_2gpus_with_checkpoint_best
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_finetune_dir
   world_size=2
   update_freq=4
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   python $fairseq_dir/fairseq_cli/hydra_train_for_with_llama.py \
            --config-dir $config_dir/config/finetune \
            --config-name base_10h \
            task.data=$tsv_dir\
            task.label_dir=$tsv_dir\
            task.labels='["ltr"]' \
            model.w2v_path=$exp_dir/checkpoint_best.pt\
            +model.freeze_hubert_layer_nums=6\
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

if [ ${stage} -le 42 ] && [ ${stop_stage} -ge 42 ];then
   echo "inference llamahubert  model on dev-other, dev-clean, test-other, test-clean of librispeech"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp

   #fairseq_dir=/workspace2/maduo/fairseq_speechtext
   #tsv_dir=/workspace2/maduo/dataset/format/librispeech
   config_dir=$fairseq_dir/examples/hubert_with_llama
   #dir=/workspace2/maduo/exp

   model_name=continue_pretain_on_hubert_iter2_with_llama_on_train_360_ft_style_freeze_first6hubertlayers_llama_in_medium_position
   exp_finetune_dir=$dir/finetune/${model_name}_10h_asr_finetune_2gpus_with_checkpoint_best
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
   # grep -rn 'Word error rate' logs/hubert_with_llama_v2_infer_1gpu.log
   # 8133:[2024-02-04 09:37:59,562][__main__][INFO] - Word error rate: 17.8560
   # 16736:[2024-02-04 09:39:31,435][__main__][INFO] - Word error rate: 29.5148
   # 24604:[2024-02-04 09:41:06,181][__main__][INFO] - Word error rate: 18.2947
   # 33441:[2024-02-04 09:42:40,419][__main__][INFO] - Word error rate: 30.2415
fi

if [ ${stage} -le 43 ] && [ ${stop_stage} -ge 43 ];then
   echo "inference llamahubert  model on dev-other, dev-clean, test-other, test-clean of librispeech with kenlm"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp
   #config_dir=$fairseq_dir/examples/voicelm/
   #fairseq_dir=/workspace2/maduo/fairseq_speechtext
   #tsv_dir=/workspace2/maduo/dataset/format/librispeech
   config_dir=$fairseq_dir/examples/hubert_with_llama
   #dir=/workspace2/maduo/exp
   model_name=continue_pretain_on_hubert_iter2_with_llama_on_train_360_ft_style_freeze_first6hubertlayers_llama_in_medium_position
   exp_finetune_dir=$dir/finetune/${model_name}_10h_asr_finetune_2gpus_with_checkpoint_best

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
                decoding.lmweight=3 \
                decoding.wordscore=-1 \
                decoding.beamthreshold=100\
                common_eval.quiet=false

   done
  
   # dev-clean dev-other test-clean test-other
   # 3.9318     8.9412    4.3240    9.4952
fi
