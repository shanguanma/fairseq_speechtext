#!/usr/bin/env bash

stage=0

stop_stage=1000
.  utils/parse_options.sh
#. path_for_fairseq_speechtext.sh ## pytorch 2.0 fairseq flashnight-text flashnight sequence
. path_for_fsq_speechtext.sh
#pip install -U flash-attn==2.0.8

#pip install flash-attn --no-build-isolation
#cd codebase/flash-attention/
#python setup.py install

export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
   echo "iter: pretrain voicelm2 on librilm monophncode and librispeech monophncode from w2vu2-model "
   echo "training on 400k steps for train-960 of librispeech"
   #fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   #tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   #dir=/mntnfs/lee_data1/maduo/exp
   #label_dir=$tsv_dir/librispeech_lm_monophncode_using_monophn_dict_librispeech_frame_monophncode_using_wav2vec-u2_model
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech/
   dir=/workspace2/maduo/exp
   label_dir=$tsv_dir/librispeech_lm_monophncode_using_monophn_dict_librispeech_frame_monophncode_using_wav2vec-u2_model
   config_dir=$fairseq_dir/examples/voicelm/voicelm2
   model_name=pretrain_on_base_voicelm2_4gpu_8update_960h_400k_update_flash_attention_lr4_5e_4_big_bs
   #model_name=pretrain_on_base_voicelm2_2gpu_16update_960h_400k_update_flash_attention_debug
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_dir
   world_size=4
   update_freq=8
   #world_size=2
   #update_freq=16

    export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   CUDA_VISIBLE_DEVICES=4,5,6,7   python $fairseq_dir/fairseq_cli/hydra_train.py \
            --config-dir $config_dir/config/pretrain \
            --config-name voicelm2_base_librispeech_flash_attention_lr4_5e_4_big_bs \
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
### 4RTX3090: training about  day
###           200steps: about 6.5 minites
fi

### using best_checkpoint to finetune model.
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
   echo "iter1: finetune voicelm2 on train-clean-100 on 80k steps in letter ctc loss mode"
   echo "with freeze feature_fuse layer and text_drop false in finetune mode"
   #fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   #tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   #dir=/mntnfs/lee_data1/maduo/exp
   #label_dir=$tsv_dir/librispeech_lm_monophncode_using_monophn_dict_librispeech_frame_monophncode_using_wav2vec-u2_model
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech/
   dir=/workspace2/maduo/exp
   label_dir=$tsv_dir/librispeech_lm_monophncode_using_monophn_dict_librispeech_frame_monophncode_using_wav2vec-u2_model
   config_dir=$fairseq_dir/examples/voicelm/voicelm2

   model_name=pretrain_on_base_voicelm2_4gpu_8update_960h_400k_update_flash_attention_lr4_5e_4_big_bs
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune_use_best_checkpoint_text_drop_true_feature_fuse_wo_freeze
   #exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_finetune_dir
   #world_size=4
   #update_freq=2
   #debug
   world_size=2
   update_freq=4
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   CUDA_VISIBLE_DEVICES=5,6   python $fairseq_dir/fairseq_cli/hydra_train.py \
            --config-dir $config_dir/config/finetune \
            --config-name voicelm2_base_100h_ctc_ltr \
            task.data=$tsv_dir\
            task.label_dir=$label_dir\
            task.labels='["ltr","textphncode"]' \
            task.text_drop=true\
            model.w2v_path=$exp_dir/checkpoint_best.pt\
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

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
   echo "inference voicelm2  model on dev-other, dev-clean, test-other, test-clean of librispeech"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech/
   dir=/workspace2/maduo/exp
   #config_dir=$fairseq_dir/examples/hubert/
   model_name=pretrain_on_base_voicelm2_4gpu_8update_960h_400k_update_flash_attention_lr4_5e_4_big_bs
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune_use_best_checkpoint_text_drop_true_feature_fuse_wo_freeze
   #results_path=$exp_finetune_dir/decode_on_100h
   results_path=$exp_finetune_dir/decode_on_100h_normalize_false
   mkdir -p $results_path
   testsets="dev-clean dev-other test-clean test-other"
   #testsets="dev-clean"
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH

   for name in $testsets;do
   CUDA_VISIBLE_DEVICES=6    python $fairseq_dir/examples/speech_recognition/new/infer_md.py \
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
   # dev-clean dev-other  test-clean test-other
   # (fsq_speechtext) maduo@lthpc-SYS-4029GP-TRT-BA001:/workspace2/maduo$ grep -rn "Word error rate" logs/bash_voicelm2_local_cluster_flash_attention_lr4_5e_4_big_bs_infer.log
   #  4.7039   11.8356     5.0612     11.5794
fi

## for debug
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
   echo "iter: pretrain voicelm2 on librilm monophncode and librispeech monophncode from w2vu2-model "
   echo "training on 400k steps for train-960 of librispeech"
   #fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   #tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   #dir=/mntnfs/lee_data1/maduo/exp
   #label_dir=$tsv_dir/librispeech_lm_monophncode_using_monophn_dict_librispeech_frame_monophncode_using_wav2vec-u2_model
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech/
   dir=/workspace2/maduo/exp
   label_dir=$tsv_dir/40M_librispeech_lm_monophncode_librispeech_frame_monophncode_using_wav2vec-u2_model
   config_dir=$fairseq_dir/examples/voicelm/voicelm2
   model_name=pretrain_on_base_voicelm2_4gpu_8update_960h_400k_update_flash_attention_lr4e_4_40M_unpaired_text_debug
   #model_name=pretrain_on_base_voicelm2_2gpu_16update_960h_400k_update_flash_attention_debug
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_dir
   #world_size=4
   #update_freq=8
   world_size=2
   update_freq=16

    export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   CUDA_VISIBLE_DEVICES=5,6  python $fairseq_dir/fairseq_cli/hydra_train.py \
            --config-dir $config_dir/config/pretrain \
            --config-name voicelm2_base_librispeech_flash_attention_lr4e_4 \
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
### 4RTX3090: training about  day
###           200steps: about  minites
fi

## for debug
## 2023-10-26 From the ultra-large-scale unpaired text, we obtain the number of text sentences has  more than several time  the number of raw audio sentences
## and The number of text and audio sentences is equal in the model input, which means that the original audio sentence will be repeated many times.
##
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then
   echo "iter: pretrain voicelm2 on librilm monophncode and librispeech monophncode from w2vu2-model "
   echo "training on 400k steps for train-960 of librispeech"
   #fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   #tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   #dir=/mntnfs/lee_data1/maduo/exp
   #label_dir=$tsv_dir/40M_librispeech_lm_monophncode_librispeech_frame_monophncode_using_wav2vec-u2_model
   
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech/
   dir=/workspace2/maduo/exp
   label_dir=$tsv_dir/40M_librispeech_lm_monophncode_librispeech_frame_monophncode_using_wav2vec-u2_model

   config_dir=$fairseq_dir/examples/voicelm/voicelm2
   model_name=pretrain_on_base_voicelm2_4gpu_8update_960h_400k_update_flash_attention_lr4e_4_40M_unpaired_text_ratio5_bs2400k
   #model_name=pretrain_on_base_voicelm2_2gpu_16update_960h_400k_update_flash_attention_debug
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_dir
   world_size=4
   update_freq=8
   #world_size=2
   #update_freq=16

    export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   CUDA_VISIBLE_DEVICES=3,4,5,6  python $fairseq_dir/fairseq_cli/hydra_train.py \
            --config-dir $config_dir/config/pretrain \
            --config-name voicelm2_base_librispeech_flash_attention_lr4e_4_text_ratio \
            task.data=$label_dir\
            task.label_dir=$label_dir\
            task.labels='["speechphncode","textphncode"]' \
            task.text_ratio=5\
            task.max_phone_size=250\
            dataset.max_tokens=2400000\
            model.label_rate=50\
            common.user_dir=$fairseq_dir/examples/voicelm/voicelm2\
            dataset.train_subset=train-960\
            dataset.valid_subset=\'dev-other_2864,dev-clean\'\
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


## now I will try to training iter2 of voicelm2 via pesduo label.
## this pesduo label is getting from 500 clusters kmeans model of 12 layer representation of iter1 voicelm2.
if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ];then
   echo "iter: pretrain voicelm2 on librilm monophncode and librispeech monophncode from w2vu2-model "
   echo "training on 400k steps for train-960 of librispeech"
   #fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   #tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   #dir=/mntnfs/lee_data1/maduo/exp


   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech/
   dir=/workspace2/maduo/exp
   label_dir=$tsv_dir/40M_librispeech_lm_monophncode_librispeech_frame_monophncode_using_wav2vec-u2_model_withiter1_voicelm2_12layer_500_kms
   #label_dir=$tsv_dir/librispeech_lm_monophncode_using_monophn_dict_librispeech_frame_monophncode_using_wav2vec-u2_model

   config_dir=$fairseq_dir/examples/voicelm/voicelm2
   model_name=pretrain_on_base_voicelm2_4gpu_8update_960h_400k_update_flash_attention_lr4e_4_40M_unpaired_text_1to40_withiter1_voicelm2_12layer_500_kms
   #model_name=pretrain_on_base_voicelm2_2gpu_16update_960h_400k_update_flash_attention_debug
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_dir
   world_size=4
   update_freq=8
   #world_size=2
   #update_freq=16

    export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   CUDA_VISIBLE_DEVICES=3,4,5,6 python $fairseq_dir/fairseq_cli/hydra_train.py \
            --config-dir $config_dir/config/pretrain \
            --config-name voicelm2_base_librispeech_flash_attention_lr4e_4_1:40 \
            task.data=$tsv_dir\
            task.label_dir=$label_dir\
            task.labels='["km","speechphncode","textphncode"]' \
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



