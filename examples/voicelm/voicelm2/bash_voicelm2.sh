#!/usr/bin/env bash

stage=0

stop_stage=1000
.  utils/parse_options.sh
. path_for_fairseq_speechtext.sh ## pytorch 2.0 fairseq flashnight-text flashnight sequence
#pip install -U flash-attn==2.0.8 

#pip install flash-attn --no-build-isolation
#cd codebase/flash-attention/
#python setup.py install

export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

### noly for sribd class
## build c++ part using CUDA, however in head node of this slurm server system hasn't cuda gpu.
#cd /mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
#pip install ninja ## using fast  distutils backend training for pytorch, it is very important
#pip install --editable ./  ## for python package, it can be installed at local environment
#python setup.py build_ext --inplace
## check ninja  it works correctly ?
##  if return exit code 0, it should works correctly, otherwise  it doesn't work
ninja --version  | echo $?
##  you can reinstall via pip uninstall -y ninja && pip install ninja


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
   echo "iter1: pretrain voicelm2 on librilm monophncode and librispeech monophncode from w2vu2-model "
   echo "training on 400k steps for train-960 of librispeech"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp
   label_dir=$tsv_dir/librispeech_lm_monophncode_using_monophn_dict_librispeech_frame_monophncode_using_wav2vec-u2_model
   #fairseq_dir=/workspace2/maduo/fairseq_speechtext
   #tsv_dir=/workspace2/maduo/dataset/format/librispeech/
   #dir=/workspace2/maduo/exp
   #label_dir=$tsv_dir/librispeech_lm_monophncode_using_monophn_dict_librispeech_frame_monophncode_using_wav2vec-u2_model
   config_dir=$fairseq_dir/examples/voicelm/voicelm2
   model_name=pretrain_on_base_voicelm2_4gpu_8update_960h_400k_update
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_dir
   world_size=4
   update_freq=8
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   python $fairseq_dir/fairseq_cli/hydra_train.py \
            --config-dir $config_dir/config/pretrain \
            --config-name voicelm2_base_librispeech \
            task.data=$tsv_dir\
            task.label_dir=$label_dir\
            task.labels='["speechphncode","textphncode"]' \
            model.label_rate=50\
            common.user_dir=$fairseq_dir/examples/voicelm/voicelm2\
            dataset.train_subset=train-960\
            dataset.valid_subset=\'dev-other,dev-clean\'\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_dir\
            checkpoint.save_dir=$exp_dir\
            hydra.run.dir=$fairseq_dir/examples/voicelm/voicelm2\
            hydra.job.name=$exp_dir/pretrain
### 4V100: training about  day
###           200steps: about 19 minites
fi
## voicelm2_base_librispeech_flash_attention.yaml
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
   echo "iter: pretrain voicelm2 on librilm monophncode and librispeech monophncode from w2vu2-model "
   echo "training on 400k steps for train-960 of librispeech"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp
   label_dir=$tsv_dir/librispeech_lm_monophncode_using_monophn_dict_librispeech_frame_monophncode_using_wav2vec-u2_model
   #fairseq_dir=/workspace2/maduo/fairseq_speechtext
   #tsv_dir=/workspace2/maduo/dataset/format/librispeech/
   #dir=/workspace2/maduo/exp
   #label_dir=$tsv_dir/librispeech_lm_monophncode_using_monophn_dict_librispeech_frame_monophncode_using_wav2vec-u2_model
   config_dir=$fairseq_dir/examples/voicelm/voicelm2
   model_name=pretrain_on_base_voicelm2_4gpu_8update_960h_400k_update_flash_attention
   #model_name=pretrain_on_base_voicelm2_2gpu_16update_960h_400k_update_flash_attention_debug
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_dir
   world_size=4
   update_freq=8
   #world_size=2
   #update_freq=16

   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   python $fairseq_dir/fairseq_cli/hydra_train.py \
            --config-dir $config_dir/config/pretrain \
            --config-name voicelm2_base_librispeech_flash_attention \
            task.data=$tsv_dir\
            task.label_dir=$label_dir\
            task.labels='["speechphncode","textphncode"]' \
            model.label_rate=50\
            common.user_dir=$fairseq_dir/examples/voicelm/voicelm2\
            dataset.train_subset=train-960\
            dataset.valid_subset=\'dev-other,dev-clean\'\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_dir\
            checkpoint.save_dir=$exp_dir\
            hydra.run.dir=$fairseq_dir/examples/voicelm/voicelm2\
            hydra.job.name=$exp_dir/pretrain
            common.tensorboard_logdir=$exp_finetune_dir\
            checkpoint.save_dir=$exp_finetune_dir\
            hydra.run.dir=$fairseq_dir/examples/voicelm/voicelm2\
            hydra.job.name=$exp_finetune_dir/finetune
### 4A100: training about  day
###           200steps: about 3 minites
fi
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
   echo "iter1: finetune voicelm2 on train-clean-100 on 80k steps in letter ctc loss mode"
   echo "without freeze feature_fuse layer in finetune mode"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp
   label_dir=$tsv_dir/librispeech_lm_monophncode_using_monophn_dict_librispeech_frame_monophncode_using_wav2vec-u2_model
   #fairseq_dir=/workspace2/maduo/fairseq_speechtext
   #tsv_dir=/workspace2/maduo/dataset/format/librispeech/
   #dir=/workspace2/maduo/exp
   #label_dir=$tsv_dir/librispeech_lm_monophncode_using_monophn_dict_librispeech_frame_monophncode_using_wav2vec-u2_model
   config_dir=$fairseq_dir/examples/voicelm/voicelm2
   
   model_name=pretrain_on_base_voicelm2_4gpu_8update_960h_400k_update_flash_attention
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune_debug
   #exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_finetune_dir
   #world_size=4
   #update_freq=2
   #debug
   world_size=2
   update_freq=4
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   python $fairseq_dir/fairseq_cli/hydra_train.py \
            --config-dir $config_dir/config/finetune \
            --config-name voicelm2_base_100h_ctc_ltr \
            task.data=$tsv_dir\
            task.label_dir=$label_dir\
            task.labels='["ltr","textphncode"]' \
            task.text_drop=true\
            model.w2v_path=$exp_dir/checkpoint_265_295000.pt\
            model.feature_fuse_freeze=false\
            common.user_dir=$fairseq_dir/examples/voicelm/voicelm2\
            dataset.train_subset=train-clean-100\
            dataset.valid_subset=\'dev-other\'\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_finetune_dir\
            checkpoint.save_dir=$exp_finetune_dir\
            hydra.run.dir=$fairseq_dir/examples/voicelm/voicelm2\
            hydra.job.name=$exp_finetune_dir/finetune

fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
   echo "inference voicelm2  model on dev-other, dev-clean, test-other, test-clean of librispeech"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp

   #config_dir=$fairseq_dir/examples/hubert/
   model_name=pretrain_on_base_voicelm2_4gpu_8update_960h_400k_update_flash_attention
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune_debug
   #results_path=$exp_finetune_dir/decode_on_100h
   results_path=$exp_finetune_dir/decode_on_100h_normalize_false
   mkdir -p $results_path
   testsets="dev-clean dev-other test-clean test-other"
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH

   for name in $testsets;do
       python $fairseq_dir/examples/speech_recognition/new/infer.py \
                --config-dir $fairseq_dir/examples/speech_recognition/new/conf\
                --config-name infer_viterbi_librispeech_for_voicelm2\
                task.data=$tsv_dir\
                task.label_dir=$tsv_dir\
                task.normalize=false\
                task._name=voicelm2_pretraining\
                +task.inference_mode=true\
                common.fp16=true\
                common_eval.results_path=$results_path\
                common_eval.path=$exp_finetune_dir/checkpoint_best.pt\
                dataset.gen_subset=$name

   done
   #  grep -rn "Word error rate" exp/finetune/pretrain_on_base_voicelm2_4gpu_8update_960h_400k_update_flash_attention_100h_asr_finetune_debug/decode_on_100h_normalize_false/viterbi/infer.log 
   # dev-clean  dev-other  test-clean  test-other
   # 5.2976     12.2703     5.4007      12.3935
fi



if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ];then
   echo "iter1: finetune voicelm2 on train-clean-100 on 80k steps in letter ctc loss mode"
   echo "freeze feature_fuse layer in finetune mode"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp
   label_dir=$tsv_dir/librispeech_lm_monophncode_using_monophn_dict_librispeech_frame_monophncode_using_wav2vec-u2_model
   #fairseq_dir=/workspace2/maduo/fairseq_speechtext
   #tsv_dir=/workspace2/maduo/dataset/format/librispeech/
   #dir=/workspace2/maduo/exp
   #label_dir=$tsv_dir/librispeech_lm_monophncode_using_monophn_dict_librispeech_frame_monophncode_using_wav2vec-u2_model
   config_dir=$fairseq_dir/examples/voicelm/voicelm2

   model_name=pretrain_on_base_voicelm2_4gpu_8update_960h_400k_update_flash_attention
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune_freeze_feature_fuse_debug
   #exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_finetune_dir
   #world_size=4
   #update_freq=2
   #debug
   world_size=2
   update_freq=4
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   python $fairseq_dir/fairseq_cli/hydra_train.py \
            --config-dir $config_dir/config/finetune \
            --config-name voicelm2_base_100h_ctc_ltr \
            task.data=$tsv_dir\
            task.label_dir=$label_dir\
            task.labels='["ltr","textphncode"]' \
            task.text_drop=true\
            model.w2v_path=$exp_dir/checkpoint_265_295000.pt\
            model.feature_fuse_freeze=true\
            common.user_dir=$fairseq_dir/examples/voicelm/voicelm2\
            dataset.train_subset=train-clean-100\
            dataset.valid_subset=\'dev-other\'\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_finetune_dir\
            checkpoint.save_dir=$exp_finetune_dir\
            hydra.run.dir=$fairseq_dir/examples/voicelm/voicelm2\
            hydra.job.name=$exp_finetune_dir/finetune

fi


if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ];then
   echo "inference voicelm2  model on dev-other, dev-clean, test-other, test-clean of librispeech"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp

   #config_dir=$fairseq_dir/examples/hubert/
   model_name=pretrain_on_base_voicelm2_4gpu_8update_960h_400k_update_flash_attention
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune_freeze_feature_fuse_debug
   #results_path=$exp_finetune_dir/decode_on_100h
   results_path=$exp_finetune_dir/decode_on_100h_normalize_false
   mkdir -p $results_path
   testsets="dev-clean dev-other test-clean test-other"
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH

   for name in $testsets;do
       python $fairseq_dir/examples/speech_recognition/new/infer.py \
                --config-dir $fairseq_dir/examples/speech_recognition/new/conf\
                --config-name infer_viterbi_librispeech\
                task.data=$tsv_dir\
                task.label_dir=$tsv_dir\
                task.normalize=false\
                task._name=voicelm2_pretraining\
                +task.inference_mode=true\
                common.fp16=true\
                common_eval.results_path=$results_path\
                common_eval.path=$exp_finetune_dir/checkpoint_best.pt\
                dataset.gen_subset=$name

   done
   # grep -rn "Word error rate" exp/finetune/pretrain_on_base_voicelm2_4gpu_8update_960h_400k_update_flash_attention_100h_asr_finetune_freeze_feature_fuse_debug/decode_on_100h_normalize_false/viterbi/infer.log 
   # dev-clean  dev-other test-clean   test-other
   # 5.3178      12.4431   5.5187       12.3534
fi

if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ];then
   echo "iter1: finetune voicelm2 on train-clean-100 on 80k steps in letter ctc loss mode"
   echo "without freeze feature_fuse layer and text_drop false in finetune mode"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp
   label_dir=$tsv_dir/librispeech_lm_monophncode_using_monophn_dict_librispeech_frame_monophncode_using_wav2vec-u2_model
   #fairseq_dir=/workspace2/maduo/fairseq_speechtext
   #tsv_dir=/workspace2/maduo/dataset/format/librispeech/
   #dir=/workspace2/maduo/exp
   #label_dir=$tsv_dir/librispeech_lm_monophncode_using_monophn_dict_librispeech_frame_monophncode_using_wav2vec-u2_model
   config_dir=$fairseq_dir/examples/voicelm/voicelm2

   model_name=pretrain_on_base_voicelm2_4gpu_8update_960h_400k_update_flash_attention
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune_text_drop_false_debug
   #exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_finetune_dir
   #world_size=4
   #update_freq=2
   #debug
   world_size=2
   update_freq=4
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   python $fairseq_dir/fairseq_cli/hydra_train.py \
            --config-dir $config_dir/config/finetune \
            --config-name voicelm2_base_100h_ctc_ltr \
            task.data=$tsv_dir\
            task.label_dir=$label_dir\
            task.labels='["ltr","textphncode"]' \
            task.text_drop=false\
            model.w2v_path=$exp_dir/checkpoint_265_295000.pt\
            model.feature_fuse_freeze=false\
            common.user_dir=$fairseq_dir/examples/voicelm/voicelm2\
            dataset.train_subset=train-clean-100\
            dataset.valid_subset=\'dev-other\'\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_finetune_dir\
            checkpoint.save_dir=$exp_finetune_dir\
            hydra.run.dir=$fairseq_dir/examples/voicelm/voicelm2\
            hydra.job.name=$exp_finetune_dir/finetune

fi


if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ];then
   echo "inference voicelm2  model on dev-other, dev-clean, test-other, test-clean of librispeech"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp

   #config_dir=$fairseq_dir/examples/hubert/
   model_name=pretrain_on_base_voicelm2_4gpu_8update_960h_400k_update_flash_attention
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune_text_drop_false_debug
   #results_path=$exp_finetune_dir/decode_on_100h
   results_path=$exp_finetune_dir/decode_on_100h_normalize_false
   mkdir -p $results_path
   testsets="dev-clean dev-other test-clean test-other"
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH

   for name in $testsets;do
       python $fairseq_dir/examples/speech_recognition/new/infer.py \
                --config-dir $fairseq_dir/examples/speech_recognition/new/conf\
                --config-name infer_viterbi_librispeech\
                task.data=$tsv_dir\
                task.label_dir=$tsv_dir\
                task.normalize=false\
                task._name=voicelm2_pretraining\
                +task.inference_mode=true\
                common.fp16=true\
                common_eval.results_path=$results_path\
                common_eval.path=$exp_finetune_dir/checkpoint_best.pt\
                dataset.gen_subset=$name

   done
   #grep -rn "Word error rate" exp/finetune/pretrain_on_base_voicelm2_4gpu_8update_960h_400k_update_flash_attention_100h_asr_finetune_text_drop_false_debug/decode_on_100h_normalize_false/viterbi/infer.log
   # dev-clean dev-other  test-clean test-other
   # 5.3674     12.1879    5.4369     12.3113
fi

if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ];then
   echo "iter1: finetune voicelm2 on train-clean-100 on 80k steps in letter ctc loss mode"
   echo "with freeze feature_fuse layer and text_drop false in finetune mode"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp
   label_dir=$tsv_dir/librispeech_lm_monophncode_using_monophn_dict_librispeech_frame_monophncode_using_wav2vec-u2_model
   #fairseq_dir=/workspace2/maduo/fairseq_speechtext
   #tsv_dir=/workspace2/maduo/dataset/format/librispeech/
   #dir=/workspace2/maduo/exp
   #label_dir=$tsv_dir/librispeech_lm_monophncode_using_monophn_dict_librispeech_frame_monophncode_using_wav2vec-u2_model
   config_dir=$fairseq_dir/examples/voicelm/voicelm2

   model_name=pretrain_on_base_voicelm2_4gpu_8update_960h_400k_update_flash_attention
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune_text_drop_false_feature_fuse_freeze_debug
   #exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_finetune_dir
   #world_size=4
   #update_freq=2
   #debug
   world_size=2
   update_freq=4
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   python $fairseq_dir/fairseq_cli/hydra_train.py \
            --config-dir $config_dir/config/finetune \
            --config-name voicelm2_base_100h_ctc_ltr \
            task.data=$tsv_dir\
            task.label_dir=$label_dir\
            task.labels='["ltr","textphncode"]' \
            task.text_drop=false\
            model.w2v_path=$exp_dir/checkpoint_265_295000.pt\
            model.feature_fuse_freeze=true\
            common.user_dir=$fairseq_dir/examples/voicelm/voicelm2\
            dataset.train_subset=train-clean-100\
            dataset.valid_subset=\'dev-other\'\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_finetune_dir\
            checkpoint.save_dir=$exp_finetune_dir\
            hydra.run.dir=$fairseq_dir/examples/voicelm/voicelm2\
            hydra.job.name=$exp_finetune_dir/finetune

fi

if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ];then
   echo "inference voicelm2  model on dev-other, dev-clean, test-other, test-clean of librispeech"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp

   #config_dir=$fairseq_dir/examples/hubert/
   model_name=pretrain_on_base_voicelm2_4gpu_8update_960h_400k_update_flash_attention
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune_text_drop_false_feature_fuse_freeze_debug
   #results_path=$exp_finetune_dir/decode_on_100h
   results_path=$exp_finetune_dir/decode_on_100h_normalize_false
   mkdir -p $results_path
   testsets="dev-clean dev-other test-clean test-other"
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH

   for name in $testsets;do
       python $fairseq_dir/examples/speech_recognition/new/infer.py \
                --config-dir $fairseq_dir/examples/speech_recognition/new/conf\
                --config-name infer_viterbi_librispeech\
                task.data=$tsv_dir\
                task.label_dir=$tsv_dir\
                task.normalize=false\
                task._name=voicelm2_pretraining\
                +task.inference_mode=true\
                common.fp16=true\
                common_eval.results_path=$results_path\
                common_eval.path=$exp_finetune_dir/checkpoint_best.pt\
                dataset.gen_subset=$name

   done
   #grep -rn "Word error rate" exp/finetune/pretrain_on_base_voicelm2_4gpu_8update_960h_400k_update_flash_attention_100h_asr_finetune_text_drop_false_feature_fuse_freeze_debug/decode_on_100h_normalize_false/viterbi/infer.log
   # dev-clean dev-other  test-clean test-other
   # 5.3583     12.1938     5.3855    12.3362
fi
## note:colusion: add unpaired text for finetune , It's converging more faster.

if [ ${stage} -le 20 ] && [ ${stop_stage} -ge 20 ];then
   echo "iter: pretrain voicelm2 on librilm monophncode and librispeech monophncode from w2vu2-model "
   echo "training on 400k steps for train-960 of librispeech"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp
   label_dir=$tsv_dir/librispeech_lm_monophncode_using_monophn_dict_librispeech_frame_monophncode_using_wav2vec-u2_model
   #fairseq_dir=/workspace2/maduo/fairseq_speechtext
   #tsv_dir=/workspace2/maduo/dataset/format/librispeech/
   #dir=/workspace2/maduo/exp
   #label_dir=$tsv_dir/librispeech_lm_monophncode_using_monophn_dict_librispeech_frame_monophncode_using_wav2vec-u2_model
   config_dir=$fairseq_dir/examples/voicelm/voicelm2
   model_name=pretrain_on_base_voicelm2_4gpu_8update_960h_400k_update_flash_attention_lower_lr
   #model_name=pretrain_on_base_voicelm2_2gpu_16update_960h_400k_update_flash_attention_debug
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_dir
   world_size=4
   update_freq=8
   #world_size=2
   #update_freq=16

    export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   python $fairseq_dir/fairseq_cli/hydra_train.py \
            --config-dir $config_dir/config/pretrain \
            --config-name voicelm2_base_librispeech_flash_attention_lower_lr \
            task.data=$tsv_dir\
            task.label_dir=$label_dir\
            task.labels='["speechphncode","textphncode"]' \
            model.label_rate=50\
            common.user_dir=$fairseq_dir/examples/voicelm/voicelm2\
            dataset.train_subset=train-960\
            dataset.valid_subset=\'dev-other,dev-clean\'\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_dir\
            checkpoint.save_dir=$exp_dir\
            hydra.run.dir=$fairseq_dir/examples/voicelm/voicelm2\
            hydra.job.name=$exp_dir/pretrain
### 4A100: training about  day
###           200steps: about 3 minites
fi

## this script is running huawei server without flash_attention
if [ ${stage} -le 25 ] && [ ${stop_stage} -ge 25 ];then
   echo "iter: pretrain voicelm2 on librilm monophncode and librispeech monophncode from w2vu2-model "
   echo "training on 400k steps for train-960 of librispeech, with multi  layer  and multi target "
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp
   label_dir=$tsv_dir/librispeech_lm_monophncode_using_monophn_dict_librispeech_frame_monophncode_using_wav2vec-u2_model
   #fairseq_dir=/workspace2/maduo/fairseq_speechtext
   #tsv_dir=/workspace2/maduo/dataset/format/librispeech/
   #dir=/workspace2/maduo/exp
   #label_dir=$tsv_dir/librispeech_lm_monophncode_using_monophn_dict_librispeech_frame_monophncode_using_wav2vec-u2_model
   config_dir=$fairseq_dir/examples/voicelm/voicelm2
   #model_name=pretrain_on_base_voicelm2_4gpu_8update_960h_400k_update_flash_attention_multi_label
   model_name=pretrain_on_base_voicelm2_4gpu_8update_960h_400k_update_multi_label
   #model_name=pretrain_on_base_voicelm2_2gpu_16update_960h_400k_update_flash_attention_debug
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_dir
   world_size=4
   update_freq=8
   #world_size=2
   #update_freq=16
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   python $fairseq_dir/fairseq_cli/hydra_train.py \
            --config-dir $config_dir/config/pretrain \
            --config-name voicelm2_base_librispeech_multi_label \
            task.data=$tsv_dir\
            task.label_dir=$label_dir\
            task.labels='["speechphncode","km","textphncode"]' \
            model.label_rate=50\
            common.user_dir=$fairseq_dir/examples/voicelm/voicelm2\
            dataset.train_subset=train-960\
            dataset.valid_subset=\'dev-other,dev-clean\'\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_dir\
            checkpoint.save_dir=$exp_dir\
            hydra.run.dir=$fairseq_dir/examples/voicelm/voicelm2\
            hydra.job.name=$exp_dir/pretrain
### 4A100: training about  day
###           200steps: about  minites
fi

if [ ${stage} -le 26 ] && [ ${stop_stage} -ge 26 ];then
   echo "iter: pretrain voicelm2 on librilm monophncode and librispeech monophncode from w2vu2-model "
   echo "training on 400k steps for train-960 of librispeech, with multi  layer  and multi target  and text mlm loss"
   #fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   #tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   #dir=/mntnfs/lee_data1/maduo/exp
   #label_dir=$tsv_dir/librispeech_lm_monophncode_using_monophn_dict_librispeech_frame_monophncode_using_wav2vec-u2_model
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech/
   dir=/workspace2/maduo/exp
   label_dir=$tsv_dir/librispeech_lm_monophncode_using_monophn_dict_librispeech_frame_monophncode_using_wav2vec-u2_model
   config_dir=$fairseq_dir/examples/voicelm/voicelm2
   #model_name=pretrain_on_base_voicelm2_4gpu_8update_960h_400k_update_flash_attention_multi_label
   #model_name=pretrain_on_base_voicelm2_4gpu_8update_960h_400k_update_multi_label_text_mlm
   model_name=pretrain_on_base_voicelm2_2gpu_16update_960h_400k_update_multi_label_text_mlm_debug
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_dir
   #world_size=4
   #update_freq=8
   world_size=2
   update_freq=16
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   python $fairseq_dir/fairseq_cli/hydra_train.py \
            --config-dir $config_dir/config/pretrain \
            --config-name voicelm2_base_librispeech_multi_label_text_mlm \
            task.data=$tsv_dir\
            task.label_dir=$label_dir\
            task.labels='["speechphncode","km","textphncode"]' \
            model.label_rate=50\
            common.user_dir=$fairseq_dir/examples/voicelm/voicelm2\
            dataset.train_subset=train-960\
            dataset.valid_subset=\'dev-other,dev-clean\'\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_dir\
            checkpoint.save_dir=$exp_dir\
            hydra.run.dir=$fairseq_dir/examples/voicelm/voicelm2\
            hydra.job.name=$exp_dir/pretrain
### 4V100: training about  day
###           200steps: about  minites
fi
