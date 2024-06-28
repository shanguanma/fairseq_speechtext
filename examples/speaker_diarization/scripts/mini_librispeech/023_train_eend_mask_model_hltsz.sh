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


if [ ${stage} -le 1 ]&&[ ${stop_stage} -ge 1 ];then

    echo "Start  eend_m2f training"
    exp_name=eend_m2f_model_10epoch_debug
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    train_conf=$root_dir/eend/conf/base/mini_librispeech_mask_train.yaml
    train_dir=$root_dir/data/simu/data/train_clean_5_ns2_beta2_500
    dev_dir=$root_dir/data/simu/data/dev_clean_2_ns2_beta2_500
    model_dir=$root_dir/exp/$exp_name
    max_epochs=100
    model_type="eend_m2f"
    num_classes=20
    batchsize=64
    subsampling=10
    lr=0.001
    python eend/eend/bin/train_mask_model.py -c $train_conf \
            --train-data-dir $train_dir \
            --valid-data-dir $dev_dir \
            --model-save-dir $model_dir \
            --max-epochs $max_epochs\
            --model-type $model_type\
	    --transformer-decoder-num-classes $num_classes\
	    --batchsize $batchsize\
	    --subsampling $subsampling\
	    --lr $lr
fi

if [ ${stage} -le 2 ]&&[ ${stop_stage} -ge 2 ];then

    echo "Start  eend_m2f training"
    exp_name=eend_m2f_model_10epoch_debug_1
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    train_conf=$root_dir/eend/conf/base/mini_librispeech_mask_train.yaml
    train_dir=$root_dir/data/simu/data/train_clean_5_ns2_beta2_500
    dev_dir=$root_dir/data/simu/data/dev_clean_2_ns2_beta2_500
    model_dir=$root_dir/exp/$exp_name
    max_epochs=10
    model_type="eend_m2f"
    num_classes=20
    batchsize=64
    subsampling=10
    lr=1.0
    python eend/eend/bin/train_mask_model.py -c $train_conf \
            --train-data-dir $train_dir \
            --valid-data-dir $dev_dir \
            --model-save-dir $model_dir \
            --max-epochs $max_epochs\
            --model-type $model_type\
            --transformer-decoder-num-classes $num_classes\
            --batchsize $batchsize\
            --subsampling $subsampling\
            --lr $lr
fi

if [ ${stage} -le 3 ]&&[ ${stop_stage} -ge 3 ];then
    echo "Start  eend_m2f training"
    exp_name=eend_m2f_model_10epoch_debug_subsample1
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    train_conf=$root_dir/eend/conf/base/mini_librispeech_mask_train.yaml
    train_dir=$root_dir/data/simu/data/train_clean_5_ns2_beta2_500
    dev_dir=$root_dir/data/simu/data/dev_clean_2_ns2_beta2_500
    model_dir=$root_dir/exp/$exp_name
    max_epochs=10
    model_type="eend_m2f"
    num_classes=20
    batchsize=64
    subsampling=1
    lr=1.0
    python eend/eend/bin/train_mask_model.py -c $train_conf \
            --train-data-dir $train_dir \
            --valid-data-dir $dev_dir \
            --model-save-dir $model_dir \
            --max-epochs $max_epochs\
            --model-type $model_type\
            --transformer-decoder-num-classes $num_classes\
            --batchsize $batchsize\
            --subsampling $subsampling\
            --lr $lr
fi

if [ ${stage} -le 4 ]&&[ ${stop_stage} -ge 4 ];then
    echo "Start  eend_m2f training"
    exp_name=eend_m2f_model_100epoch_debug_1
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    train_conf=$root_dir/eend/conf/base/mini_librispeech_mask_train.yaml
    train_dir=$root_dir/data/simu/data/train_clean_5_ns2_beta2_500
    dev_dir=$root_dir/data/simu/data/dev_clean_2_ns2_beta2_500
    model_dir=$root_dir/exp/$exp_name
    max_epochs=100
    model_type="eend_m2f"
    num_classes=20
    batchsize=64
    subsampling=10
    lr=1.0
    python eend/eend/bin/train_mask_model.py -c $train_conf \
            --train-data-dir $train_dir \
            --valid-data-dir $dev_dir \
            --model-save-dir $model_dir \
            --max-epochs $max_epochs\
            --model-type $model_type\
            --transformer-decoder-num-classes $num_classes\
            --batchsize $batchsize\
            --subsampling $subsampling\
            --lr $lr
fi

