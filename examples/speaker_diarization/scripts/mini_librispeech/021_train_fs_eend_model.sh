#!/usr/bin/env bash

stage=0
stop_stage=1000
nj=32
. utils/parse_options.sh
#. path_for_fairseq_speechtext.sh
#. path_for_fsq_speechtext.sh
. path_for_fsq_sptt.sh # with pytorch_lightning=2.1.2
#. path.sh # with kaldi env and fsq_sptt
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1


if [ ${stage} -le 0 ]&&[ ${stop_stage} -ge 0 ];then
 
    echo "Start training"
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    train_conf=$root_dir/fs_eend/config/spk_onl_tfm_enc_dec_nonautoreg.yaml
    ## (TODO) modify data dir at config
    train_dir=$root_dir/data/simu/data/train_clean_5_ns2_beta2_500
    dev_dir=$root_dir/data/simu/data/dev_clean_2_ns2_beta2_500
    model_dir=$root_dir/exp/fs_eend_base
    #model_dir=$root_dir/exp/fs_eend_base/
    mkdir -p $model_dir
    python fs_eend/train.py --configs $train_conf  --gpus 1
    

fi


