#!/usr/bin/env bash

stage=0
stop_stage=1000
. utils/parse_options.sh

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
   echo "pretrain base data2vec2.0  model for speech (e.g. librispeech)"
   tsv_dir=dataset/format/librispeech
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   config_dir=$fairseq_dir/examples/data2vec/config/v2
   dir=/workspace2/maduo/exp
   model_name=pretrain_on_data2vec2_4gpu_8update_960h_400k_update_offical_setting
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_dir
   world_size=4
   update_freq=8
   CUDA_VISIBLE_DEVICES=0,5,6,7 python $fairseq_dir/fairseq_cli/hydra_train.py -m \
            --config-dir $config_dir \
            --config-name base_audio_only_task_librispeech \
            task.data=$tsv_dir\
            common.user_dir=$fairseq_dir/examples/data2vec\
            dataset.train_subset=train-960\
            dataset.valid_subset=\'dev-other,dev-clean\'\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_dir\
            checkpoint.save_dir=$exp_dir\
            hydra.run.dir=$fairseq_dir/examples/data2vec\
            hydra.job.name=$exp_dir/pretrain
             
fi

#(TODO) check pretrain model name
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
   echo "finetune data2vec2 speech base model "
   world_size=4  # total gpu number
   update_freq=2
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   dir=/workspace2/maduo/exp
   model_name=pretrain_on_data2vec2_4gpu_8update_960h_400k_update_offical_setting ##
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune
   mkdir -p $exp_finetune_dir
   
   fairseq-hydra-train \
            --config-dir  $fairseq_dir/examples/wav2vec/config/finetuning \
            --config-name base_100h\
            task.data=$tsv_dir \
            model.w2v_path=$exp_dir/checkpoint_289_400000.pt\
            common.user_dir=$fairseq_dir/examples/data2vec\
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
   echo "inference dev-clean dev-other test-clean test-other of librispeech using above finetune data2vec2"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   dir=/workspace2/maduo/exp
   model_name=pretrain_on_data2vec2_4gpu_8update_960h_400k_update_offical_setting
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune
   mkdir -p $exp_finetune_dir/decode_100h
   testsets='dev-clean dev-other test-clean test-other'
   for name in $testsets;do
     python $fairseq_dir/examples/speech_recognition/new/infer.py\
           --config-dir $fairseq_dir/examples/speech_recognition/new/conf \
           --config-name infer \
           task=audio_finetuning
           task.data=$tsv_dir\
           task.label=ltr\
           common.user_dir=$fairseq_dir/examples/data2vec
           decoding.unique_wer_file=True \
           common_eval.results_path=$exp_finetune_dir/decode\
           common_eval.path=$exp_finetune_dir/checkpoint_best.pt\
           dataset.gen_subset=$name\
           decoding.beam=1500\
           decoding.type=viterbi \
           distributed_training.distributed_world_size=1

   done
fi


