#!/bin/bash

stage=0

stop_stage=1000
.  utils/parse_options.sh
. path_for_fairseq_speechtext.sh ## pytorch 2.0 fairseq flashnight-text flashnight sequence
#. path_for_fsq_speechtext.sh
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

## build c++ part using CUDA, however in head node of this slurm server system hasn't cuda gpu.
cd /mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
pip install ninja ## using fast  distutils backend training for pytorch, it is very important
#pip install --editable ./  ## for python package, it can be installed at local environment
python setup.py build_ext --inplace


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
    echo "iter: pretrain voicelm on 15layer of hubert-large pesudo label and librispeech monophncode from GAN(w2vu2) Chinese model"
   echo "training on 400k steps for train_m of wenetspeech"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   #tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntcephfs/lab_data/maduo/exp/
   #label_dir=$tsv_dir/offical_hubert_codes_and_librispeech_frame_monophncode_using_wav2vec-u2_model
   #tsv_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech/m/no_segements_tsv_dir
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_middle_speech_aishell-2_unpair_text
   label_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_middle_speech_aishell-2_unpair_text
   config_dir=$fairseq_dir/examples/voicelm
   model_name=pretrain_on_base_T-hubert_Chinese_4gpu_8update_960h_400k_update
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_dir
   world_size=4
   update_freq=8
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   python $fairseq_dir/fairseq_cli/hydra_train.py \
            --config-dir $config_dir/config/pretrain \
            --config-name voicelm_base_librispeech \
            task.data=$tsv_dir\
            task.label_dir=$label_dir\
            task.labels='["unsupphncode1","km"]' \
            model.label_rate=50\
            common.user_dir=$fairseq_dir/examples/voicelm\
            dataset.train_subset=train_m\
            dataset.valid_subset=dev\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_dir\
            checkpoint.save_dir=$exp_dir\
            hydra.run.dir=$fairseq_dir/examples/voicelm\
            hydra.job.name=$exp_dir/pretrain
fi
