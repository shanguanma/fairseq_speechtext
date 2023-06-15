#!/usr/bin/env bash
  
stage=0

stop_stage=1000
.  utils/parse_options.sh
. path_for_fairseq_speechtext.sh ## pytorch 2.0 fairseq flashnight-text flashnight sequence 

export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
   echo "iter1: pretrain hubert on mfcc pesudo label "
   echo "training on 250k steps for train-960 of librispeech"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   label_dir=$tsv_dir/mfcc/mfcc_lab # ##postfix .km files folder
   config_dir=source_md/wav2vec-u2/
   dir=/workspace2/maduo/exp
   model_name=pretrain_on_small_hubert_4gpu_8update_960h_mfcc_250k_update  
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_dir
   world_size=4
   update_freq=8
   CUDA_VISIBLE_DEVICES=0,1,2,3 python $fairseq_dir/fairseq_cli/hydra_train.py \
            --config-dir $config_dir/config/pretrain \
            --config-name hubert_small_base_librispeech_mfcc \
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
### 4RTX3090: training about 3day
###           200steps: about 3.5 minites
fi

### i want to see performance of  iter1 hubert  model(small model).
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
   echo "fine tune small hubert model(trained on mfcc) using train-clean-100 supervision data"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   config_dir=$fairseq_dir/examples/hubert
   dir=/workspace2/maduo/exp
   model_name=pretrain_on_small_hubert_4gpu_8update_960h_mfcc_250k_update
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_finetune_dir
   world_size=4
   update_freq=2
   CUDA_VISIBLE_DEVICES=0,1,2,3 python $fairseq_dir/fairseq_cli/hydra_train.py \
       --config-dir $config_dir/config/finetune \
            --config-name base_100h \
            task.data=$tsv_dir\
            task.label_dir=$tsv_dir\
            task.labels='["ltr"]' \
            model.w2v_path=$exp_dir/checkpoint_181_250000.pt\
            common.user_dir=$fairseq_dir/examples/hubert\
            dataset.train_subset=train-clean-100\
            dataset.valid_subset=dev-other\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_finetune_dir\
            checkpoint.save_dir=$exp_finetune_dir\
            hydra.run.dir=$fairseq_dir/examples/hubert\
            hydra.job.name=$exp_dir/finetune

fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
   echo "inference sthubert model on dev-other, dev-clean, test-other, test-clean of librispeech"
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
   #result: mfcc iter:250k@181epochs, finetune:80k@222epchs
   # WER% is from $exp_finetune_dir/decode_on_100/viterbi/infer.log
   #  dev-clean   dev-other   test-clean   test-other
  

fi
                    
