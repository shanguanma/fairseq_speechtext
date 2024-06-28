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

    echo "Start eend_eda training"
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    train_conf=$root_dir/eend/conf/base/mini_librispeech_train.yaml
    train_dir=$root_dir/data/simu/data/train_clean_5_ns2_beta2_500
    dev_dir=$root_dir/data/simu/data/dev_clean_2_ns2_beta2_500
    model_dir=$root_dir/exp/eend_eda_base_100epoch
    max_epochs=100
    model_type="TransformerEda"
    python eend/eend/bin/train_eda.py -c $train_conf \
	    $train_dir \
	    $dev_dir \
	    $model_dir \
	    --max-epochs $max_epochs\
	    --model-type $model_type

fi


if [ ${stage} -le 2 ]&&[ ${stop_stage} -ge 2 ];then

    echo "Start eend_eda training"
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    train_conf=$root_dir/eend/conf/base/mini_librispeech_train.yaml
    train_dir=$root_dir/data/simu/data/train_clean_5_ns2_beta2_500
    dev_dir=$root_dir/data/simu/data/dev_clean_2_ns2_beta2_500
    model_dir=$root_dir/exp/eend_eda_base_100epoch_debug
    max_epochs=100
    model_type="TransformerEda"
    python eend/eend/bin/train_eda.py -c $train_conf \
            $train_dir \
            $dev_dir \
            $model_dir \
            --max-epochs $max_epochs\
            --model-type $model_type

fi

if [ ${stage} -le 3 ]&&[ ${stop_stage} -ge 3 ];then

    echo "Start  conformer eend_eda training"
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    train_conf=$root_dir/eend/conf/base/mini_librispeech_train.yaml
    train_dir=$root_dir/data/simu/data/train_clean_5_ns2_beta2_500
    dev_dir=$root_dir/data/simu/data/dev_clean_2_ns2_beta2_500
    model_dir=$root_dir/exp/eend_eda_base_100epoch_conformer
    max_epochs=100
    model_type="ConformerEda"
    python eend/eend/bin/train_eda.py -c $train_conf \
            $train_dir \
            $dev_dir \
            $model_dir \
            --max-epochs $max_epochs\
            --model-type $model_type

fi

if [ ${stage} -le 4 ]&&[ ${stop_stage} -ge 4 ];then

    echo "Start  conformer eend_eda training"
    exp_name=eend_eda_base_100epoch_conformer_4layer
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    train_conf=$root_dir/eend/conf/base/mini_librispeech_train.yaml
    train_dir=$root_dir/data/simu/data/train_clean_5_ns2_beta2_500
    dev_dir=$root_dir/data/simu/data/dev_clean_2_ns2_beta2_500
    model_dir=$root_dir/exp/$exp_name
    max_epochs=100
    model_type="ConformerEda"
    conformer_encoder_n_layers=4
    python eend/eend/bin/train_eda.py -c $train_conf \
            $train_dir \
            $dev_dir \
            $model_dir \
            --max-epochs $max_epochs\
            --model-type $model_type\
	    --transformer-encoder-n-layers $conformer_encoder_n_layers

fi


if [ ${stage} -le 5 ]&&[ ${stop_stage} -ge 5 ];then

    echo "Start  conformer eend_eda training"
    exp_name=eend_eda_base_100epoch_conformer_4layer_debug
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    train_conf=$root_dir/eend/conf/base/mini_librispeech_train.yaml
    train_dir=$root_dir/data/simu/data/train_clean_5_ns2_beta2_500
    dev_dir=$root_dir/data/simu/data/dev_clean_2_ns2_beta2_500
    model_dir=$root_dir/exp/$exp_name
    max_epochs=100
    model_type="ConformerEda"
    conformer_encoder_n_layers=4
    python eend/eend/bin/train_eda.py -c $train_conf \
            $train_dir \
            $dev_dir \
            $model_dir \
            --max-epochs $max_epochs\
            --model-type $model_type\
            --transformer-encoder-n-layers $conformer_encoder_n_layers

fi
