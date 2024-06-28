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

    echo "Start model averaging"
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    model_dir=$root_dir/exp/eend_eda_base_100epoch
    init_model=$model_dir/avg.th
    ifiles=`eval echo $model_dir/transformer{90..100}.th`
    python eend/eend/bin/model_averaging.py $init_model $ifiles
fi


# Inferring
if [ $stage -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Start inferring"
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    infer_conf=$root_dir/eend/conf/base/mini_librispeech_infer.yaml
    test_dir=$root_dir/data/simu/data/dev_clean_2_ns2_beta2_500
    test_model=$root_dir/exp/eend_eda_base_100epoch/avg.th
    infer_out_dir=$root_dir/exp/eend_eda_base_100epoch/infer_eda
    model_type="TransformerEda"
    python eend/eend/bin/infer_eda.py -c $infer_conf $test_dir $test_model $infer_out_dir --model_type $model_type
fi



# Scoring
if [ $stage -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Start scoring"
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    stck_dir=$root_dir/SCTK-2.4.12/src/md-eval/
    infer_out_dir=$root_dir/exp/eend_eda_base_100epoch/infer_eda
    work=$infer_out_dir/.work
    scoring_dir=$infer_out_dir/score/mini_librispeech_dev
    test_dir=data/simu/data/dev_clean_2_ns2_beta2_500/
    mkdir -p $work
    mkdir -p $scoring_dir
    find $infer_out_dir -iname "*.h5" > $work/file_list
      for med in 1 11; do
        for th in 0.3 0.4 0.5 0.6 0.7; do
        python eend/eend/bin/make_rttm.py --median=$med --threshold=$th \
                --frame_shift=80 --subsampling=10 --sampling_rate=8000 \
                $work/file_list $scoring_dir/hyp_${th}_$med.rttm
        $stck_dir/md-eval.pl -c 0.25 \
                -r $test_dir/rttm \
                -s $scoring_dir/hyp_${th}_$med.rttm > $scoring_dir/result_th${th}_med${med}_collar0.25 2>/dev/null || exit
        done
      done
   echo "result: DER,MS,FA,SC"
   head  $scoring_dir/result_th0.*
   # result:
   # # DER, MS, FA, SC
   #  head  exp/eend_base_100epoch/infer/score/mini_librispeech_dev/result_th0.*
   #Start scoring
#result: DER,MS,FA,SC
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch/infer_eda/score/mini_librispeech_dev/result_th0.3_med11_collar0.25 <==
#41.09/2.41/37.73/0.95
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch/infer_eda/score/mini_librispeech_dev/result_th0.3_med1_collar0.25 <==
#40.70/2.68/36.99/1.03
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch/infer_eda/score/mini_librispeech_dev/result_th0.4_med11_collar0.25 <==
#38.60/3.12/34.19/1.29
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch/infer_eda/score/mini_librispeech_dev/result_th0.4_med1_collar0.25 <==
#38.49/3.77/33.32/1.40
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch/infer_eda/score/mini_librispeech_dev/result_th0.5_med11_collar0.25 <==
#35.76/4.26/29.79/1.72
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch/infer_eda/score/mini_librispeech_dev/result_th0.5_med1_collar0.25 <==
#36.19/5.41/28.94/1.84
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch/infer_eda/score/mini_librispeech_dev/result_th0.6_med11_collar0.25 <==
#33.22/6.25/24.81/2.16
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch/infer_eda/score/mini_librispeech_dev/result_th0.6_med1_collar0.25 <==
#34.30/8.04/23.92/2.34
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch/infer_eda/score/mini_librispeech_dev/result_th0.7_med11_collar0.25 <==
#30.83/9.63/18.49/2.70 # as final result
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch/infer_eda/score/mini_librispeech_dev/result_th0.7_med1_collar0.25 <==
#33.31/12.43/17.97/2.91
fi

if [ ${stage} -le 4 ]&&[ ${stop_stage} -ge 4 ];then

    echo "Start model averaging"
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    model_dir=$root_dir/exp/eend_eda_base_100epoch_debug
    dst_model=$model_dir/best_avg.th
    avg_num=10
    remove_models='True'
    python eend/eend/bin/model_averaging2.py \
	    --dst_model $dst_model\
	    --src_path $model_dir\
	    --nums $avg_num\
	    --remove_models $remove_models
	    
fi


# Inferring
if [ $stage -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "Start inferring"
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    infer_conf=$root_dir/eend/conf/base/mini_librispeech_infer.yaml
    test_dir=$root_dir/data/simu/data/dev_clean_2_ns2_beta2_500
    test_model=$root_dir/exp/eend_eda_base_100epoch_debug/best_avg.th
    infer_out_dir=$root_dir/exp/eend_eda_base_100epoch_debug/infer_eda
    model_type="TransformerEda"
    python eend/eend/bin/infer_eda.py -c $infer_conf $test_dir $test_model $infer_out_dir --model_type $model_type
fi

# Scoring
if [ $stage -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "Start scoring"
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    stck_dir=$root_dir/SCTK-2.4.12/src/md-eval/
    infer_out_dir=$root_dir/exp/eend_eda_base_100epoch_debug/infer_eda
    work=$infer_out_dir/.work
    scoring_dir=$infer_out_dir/score/mini_librispeech_dev
    test_dir=data/simu/data/dev_clean_2_ns2_beta2_500/
    mkdir -p $work
    mkdir -p $scoring_dir
    find $infer_out_dir -iname "*.h5" > $work/file_list
      for med in 1 11; do
        for th in 0.3 0.4 0.5 0.6 0.7; do
        python eend/eend/bin/make_rttm.py --median=$med --threshold=$th \
                --frame_shift=80 --subsampling=10 --sampling_rate=8000 \
                $work/file_list $scoring_dir/hyp_${th}_$med.rttm
        $stck_dir/md-eval.pl -c 0.25 \
                -r $test_dir/rttm \
                -s $scoring_dir/hyp_${th}_$med.rttm > $scoring_dir/result_th${th}_med${med}_collar0.25 2>/dev/null || exit
        done
      done
   echo "result: DER,MS,FA,SC"
   head  $scoring_dir/result_th0.*

   #Start scoring
#result: DER,MS,FA,SC
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch_debug/infer_eda/score/mini_librispeech_dev/result_th0.3_med11_collar0.25 <==
#40.97/2.42/37.65/0.90
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch_debug/infer_eda/score/mini_librispeech_dev/result_th0.3_med1_collar0.25 <==
#40.66/2.76/36.87/1.03
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch_debug/infer_eda/score/mini_librispeech_dev/result_th0.4_med11_collar0.25 <==
#38.43/3.04/34.15/1.24
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch_debug/infer_eda/score/mini_librispeech_dev/result_th0.4_med1_collar0.25 <==
#38.28/3.80/33.07/1.41
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch_debug/infer_eda/score/mini_librispeech_dev/result_th0.5_med11_collar0.25 <==
#35.64/4.11/29.89/1.65
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch_debug/infer_eda/score/mini_librispeech_dev/result_th0.5_med1_collar0.25 <==
#35.92/5.52/28.50/1.89
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch_debug/infer_eda/score/mini_librispeech_dev/result_th0.6_med11_collar0.25 <==
#32.36/6.05/24.04/2.28
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch_debug/infer_eda/score/mini_librispeech_dev/result_th0.6_med1_collar0.25 <==
#33.88/8.48/22.92/2.48
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch_debug/infer_eda/score/mini_librispeech_dev/result_th0.7_med11_collar0.25 <==
#29.50/10.13/16.36/3.01
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch_debug/infer_eda/score/mini_librispeech_dev/result_th0.7_med1_collar0.25 <==
#33.13/13.65/16.40/3.08
fi

if [ ${stage} -le 7 ]&&[ ${stop_stage} -ge 7 ];then

    echo "Start model averaging"
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    model_dir=$root_dir/exp/eend_eda_base_100epoch_conformer
    dst_model=$model_dir/best_avg.th
    avg_num=10
    remove_models='True'
    python eend/eend/bin/model_averaging2.py \
            --dst_model $dst_model\
            --src_path $model_dir\
            --nums $avg_num\
            --remove_models $remove_models

fi


# Inferring
if [ $stage -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    echo "Start inferring"
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    infer_conf=$root_dir/eend/conf/base/mini_librispeech_infer.yaml
    test_dir=$root_dir/data/simu/data/dev_clean_2_ns2_beta2_500
    test_model=$root_dir/exp/eend_eda_base_100epoch_conformer/best_avg.th
    infer_out_dir=$root_dir/exp/eend_eda_base_100epoch_conformer/infer_eda
    model_type="ConformerEda"
    python eend/eend/bin/infer_eda.py -c $infer_conf $test_dir $test_model $infer_out_dir --model_type $model_type
fi

# Scoring
if [ $stage -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    echo "Start scoring"
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    stck_dir=$root_dir/SCTK-2.4.12/src/md-eval/
    infer_out_dir=$root_dir/exp/eend_eda_base_100epoch_conformer/infer_eda
    work=$infer_out_dir/.work
    scoring_dir=$infer_out_dir/score/mini_librispeech_dev
    test_dir=data/simu/data/dev_clean_2_ns2_beta2_500/
    mkdir -p $work
    mkdir -p $scoring_dir
    find $infer_out_dir -iname "*.h5" > $work/file_list
      for med in 1 11; do
        for th in 0.3 0.4 0.5 0.6 0.7; do
        python eend/eend/bin/make_rttm.py --median=$med --threshold=$th \
                --frame_shift=80 --subsampling=10 --sampling_rate=8000 \
                $work/file_list $scoring_dir/hyp_${th}_$med.rttm
        $stck_dir/md-eval.pl -c 0.25 \
                -r $test_dir/rttm \
                -s $scoring_dir/hyp_${th}_$med.rttm > $scoring_dir/result_th${th}_med${med}_collar0.25 2>/dev/null || exit
        done
      done
   echo "result: DER,MS,FA,SC"
   head  $scoring_dir/result_th0.*
   #Start scoring
#result: DER,MS,FA,SC
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch_conformer/infer_eda/score/mini_librispeech_dev/result_th0.3_med11_collar0.25 <==
#23.21/1.90/19.04/2.27
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch_conformer/infer_eda/score/mini_librispeech_dev/result_th0.3_med1_collar0.25 <==
#23.16/1.97/18.83/2.35
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch_conformer/infer_eda/score/mini_librispeech_dev/result_th0.4_med11_collar0.25 <==
#21.57/2.65/16.19/2.73
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch_conformer/infer_eda/score/mini_librispeech_dev/result_th0.4_med1_collar0.25 <==
#21.57/2.76/16.00/2.81
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch_conformer/infer_eda/score/mini_librispeech_dev/result_th0.5_med11_collar0.25 <==
#20.33/3.69/13.56/3.09
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch_conformer/infer_eda/score/mini_librispeech_dev/result_th0.5_med1_collar0.25 <==
#20.44/3.83/13.43/3.18
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch_conformer/infer_eda/score/mini_librispeech_dev/result_th0.6_med11_collar0.25 <==
#19.60/5.13/11.05/3.42
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch_conformer/infer_eda/score/mini_librispeech_dev/result_th0.6_med1_collar0.25 <==
#19.80/5.32/11.01/3.47
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch_conformer/infer_eda/score/mini_librispeech_dev/result_th0.7_med11_collar0.25 <==
#19.61/7.36/8.67/3.58 # as final result
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch_conformer/infer_eda/score/mini_librispeech_dev/result_th0.7_med1_collar0.25 <==
#19.86/7.57/8.65/3.63
fi


if [ ${stage} -le 10 ]&&[ ${stop_stage} -ge 10 ];then
    echo "Start model averaging"
    exp_name=eend_eda_base_100epoch_conformer_4layer
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    model_dir=$root_dir/exp/$exp_name
    dst_model=$model_dir/best_avg.th
    avg_num=10
    remove_models='True'
    python eend/eend/bin/model_averaging2.py \
            --dst_model $dst_model\
            --src_path $model_dir\
            --nums $avg_num\
            --remove_models $remove_models

fi


# Inferring
if [ $stage -le 11 ] && [ ${stop_stage} -ge 11 ]; then
    echo "Start inferring"
    exp_name=eend_eda_base_100epoch_conformer_4layer
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    infer_conf=$root_dir/eend/conf/base/mini_librispeech_infer.yaml
    test_dir=$root_dir/data/simu/data/dev_clean_2_ns2_beta2_500
    test_model=$root_dir/exp/$exp_name/best_avg.th
    infer_out_dir=$root_dir/exp/$exp_name/infer_eda
    model_type="ConformerEda"
    conformer_encoder_n_layers=4
    python eend/eend/bin/infer_eda.py \
	    -c $infer_conf \
	    $test_dir \
	    $test_model\
	    $infer_out_dir \
	    --model_type $model_type\
	    --transformer-encoder-n-layers $conformer_encoder_n_layers
fi

# Scoring
if [ $stage -le 12 ] && [ ${stop_stage} -ge 12 ]; then
    echo "Start scoring"
    exp_name=eend_eda_base_100epoch_conformer_4layer
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    stck_dir=$root_dir/SCTK-2.4.12/src/md-eval/
    infer_out_dir=$root_dir/exp/$exp_name/infer_eda
    work=$infer_out_dir/.work
    scoring_dir=$infer_out_dir/score/mini_librispeech_dev
    test_dir=data/simu/data/dev_clean_2_ns2_beta2_500/
    mkdir -p $work
    mkdir -p $scoring_dir
    find $infer_out_dir -iname "*.h5" > $work/file_list
      for med in 1 11; do
        for th in 0.3 0.4 0.5 0.6 0.7; do
        python eend/eend/bin/make_rttm.py --median=$med --threshold=$th \
                --frame_shift=80 --subsampling=10 --sampling_rate=8000 \
                $work/file_list $scoring_dir/hyp_${th}_$med.rttm
        $stck_dir/md-eval.pl -c 0.25 \
		-r $test_dir/rttm \
                -s $scoring_dir/hyp_${th}_$med.rttm > $scoring_dir/result_th${th}_med${med}_collar0.25 2>/dev/null || exit
        done
      done
   echo "result: DER,MS,FA,SC"
   head  $scoring_dir/result_th0.*
   #Start scoring
#result: DER,MS,FA,SC
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch_conformer_4layer/infer_eda/score/mini_librispeech_dev/result_th0.3_med11_collar0.25 <==
#23.25/1.39/19.39/2.48
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch_conformer_4layer/infer_eda/score/mini_librispeech_dev/result_th0.3_med1_collar0.25 <==
#23.21/1.46/19.18/2.57
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch_conformer_4layer/infer_eda/score/mini_librispeech_dev/result_th0.4_med11_collar0.25 <==
#21.74/2.12/16.59/3.04
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch_conformer_4layer/infer_eda/score/mini_librispeech_dev/result_th0.4_med1_collar0.25 <==
#21.77/2.24/16.44/3.09
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch_conformer_4layer/infer_eda/score/mini_librispeech_dev/result_th0.5_med11_collar0.25 <==
#20.76/3.17/14.14/3.45
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch_conformer_4layer/infer_eda/score/mini_librispeech_dev/result_th0.5_med1_collar0.25 <==
#20.85/3.33/14.03/3.49
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch_conformer_4layer/infer_eda/score/mini_librispeech_dev/result_th0.6_med11_collar0.25 <==
#20.19/4.66/11.80/3.73
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch_conformer_4layer/infer_eda/score/mini_librispeech_dev/result_th0.6_med1_collar0.25 <==
#20.37/4.85/11.75/3.77
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch_conformer_4layer/infer_eda/score/mini_librispeech_dev/result_th0.7_med11_collar0.25 <==
#20.20/6.85/9.49/3.86
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch_conformer_4layer/infer_eda/score/mini_librispeech_dev/result_th0.7_med1_collar0.25 <==
#20.44/7.07/9.49/3.89
fi

# Inferring
if [ $stage -le 13 ] && [ ${stop_stage} -ge 13 ]; then
    echo "Start inferring"
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    infer_conf=$root_dir/eend/conf/base/mini_librispeech_infer.yaml
    test_dir=$root_dir/data/simu/data/dev_clean_2_ns2_beta2_500
    test_model=$root_dir/exp/eend_eda_base_100epoch_conformer/best_avg.th
    infer_out_dir=$root_dir/exp/eend_eda_base_100epoch_conformer/infer_eda_debug
    model_type="ConformerEda"
    sample_rate=8000
    python eend/eend/bin/infer_eda.py -c $infer_conf $test_dir $test_model $infer_out_dir --model_type $model_type --sampling-rate $sample_rate
fi

# Scoring
if [ $stage -le 14 ] && [ ${stop_stage} -ge 14 ]; then
    echo "Start scoring"
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    stck_dir=$root_dir/SCTK-2.4.12/src/md-eval/
    infer_out_dir=$root_dir/exp/eend_eda_base_100epoch_conformer/infer_eda_debug
    work=$infer_out_dir/.work
    scoring_dir=$infer_out_dir/score/mini_librispeech_dev
    test_dir=data/simu/data/dev_clean_2_ns2_beta2_500/
    mkdir -p $work
    mkdir -p $scoring_dir
    find $infer_out_dir -iname "*.h5" > $work/file_list
      for med in 1 11; do
        for th in 0.3 0.4 0.5 0.6 0.7; do
        python eend/eend/bin/make_rttm.py --median=$med --threshold=$th \
                --frame_shift=80 --subsampling=10 --sampling_rate=8000 \
                $work/file_list $scoring_dir/hyp_${th}_$med.rttm
        $stck_dir/md-eval.pl -c 0.25 \
                -r $test_dir/rttm \
                -s $scoring_dir/hyp_${th}_$med.rttm > $scoring_dir/result_th${th}_med${med}_collar0.25 2>/dev/null || exit
        done
      done
   echo "result: DER,MS,FA,SC"
   head  $scoring_dir/result_th0.*
#Start scoring
#result: DER,MS,FA,SC
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch_conformer/infer_eda_debug/score/mini_librispeech_dev/result_th0.3_med11_collar0.25 <==
#20.97/1.40/17.21/2.36
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch_conformer/infer_eda_debug/score/mini_librispeech_dev/result_th0.3_med1_collar0.25 <==
#20.97/1.47/17.06/2.44
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch_conformer/infer_eda_debug/score/mini_librispeech_dev/result_th0.4_med11_collar0.25 <==
#19.05/2.12/14.07/2.86
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch_conformer/infer_eda_debug/score/mini_librispeech_dev/result_th0.4_med1_collar0.25 <==
#19.13/2.23/13.95/2.95
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch_conformer/infer_eda_debug/score/mini_librispeech_dev/result_th0.5_med11_collar0.25 <==
#17.76/3.15/11.34/3.28
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch_conformer/infer_eda_debug/score/mini_librispeech_dev/result_th0.5_med1_collar0.25 <==
#17.94/3.30/11.27/3.37
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch_conformer/infer_eda_debug/score/mini_librispeech_dev/result_th0.6_med11_collar0.25 <==
#17.10/4.67/8.80/3.63
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch_conformer/infer_eda_debug/score/mini_librispeech_dev/result_th0.6_med1_collar0.25 <==
#17.37/4.86/8.84/3.66
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch_conformer/infer_eda_debug/score/mini_librispeech_dev/result_th0.7_med11_collar0.25 <==
#17.29/7.02/6.53/3.74
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_eda_base_100epoch_conformer/infer_eda_debug/score/mini_librispeech_dev/result_th0.7_med1_collar0.25 <==
#17.59/7.24/6.58/3.78

fi
