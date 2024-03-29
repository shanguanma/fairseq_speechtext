#!/usr/bin/env bash

stage=0
stop_stage=1000
nj=32
. utils/parse_options.sh
#. path_for_fairseq_speechtext.sh
#. path_for_fsq_speechtext.sh
. path_for_fsq_sptt.sh
#. path.sh # with kaldi env and fsq_sptt
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1


if [ ${stage} -le 0 ]&&[ ${stop_stage} -ge 0 ];then
 
    echo "Start training"
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    train_conf=$root_dir/eend/conf/base/mini_librispeech_train.yaml
    train_dir=$root_dir/data/simu/data/train_clean_5_ns2_beta2_500
    dev_dir=$root_dir/data/simu/data/dev_clean_2_ns2_beta2_500
    model_dir=$root_dir/exp/eend_base_100epoch
    max_epochs=100
    python eend/eend/bin/train.py -c $train_conf $train_dir $dev_dir $model_dir --max-epochs $max_epochs

fi


