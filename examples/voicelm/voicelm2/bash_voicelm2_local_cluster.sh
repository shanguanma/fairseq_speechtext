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
### 4A100: training about  day
###           200steps: about 3 minites
fi
