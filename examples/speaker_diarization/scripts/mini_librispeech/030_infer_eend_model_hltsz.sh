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


# Model averaging
if [ ${stage} -le 0 ]&&[ ${stop_stage} -ge 0 ];then

    echo "Start model averaging"
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    model_dir=$root_dir/exp/eend_base
    init_model=$model_dir/avg.th
    ifiles=`eval echo $model_dir/transformer{7..9}.th`
    python eend/eend/bin/model_averaging.py $init_model $ifiles
fi


# Inferring
if [ $stage -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Start inferring"
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    infer_conf=$root_dir/eend/conf/base/mini_librispeech_infer.yaml
    test_dir=$root_dir/data/simu/data/dev_clean_2_ns2_beta2_500
    test_model=$root_dir/exp/eend_base/avg.th
    infer_out_dir=$root_dir/exp/eend_base/infer
    python eend/eend/bin/infer.py -c $infer_conf $test_dir $test_model $infer_out_dir
fi



# Scoring
if [ $stage -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Start scoring"
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    stck_dir=$root_dir/SCTK-2.4.12/src/md-eval/
    infer_out_dir=$root_dir/exp/eend_base/infer
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
	# result:
	# DER, MS, FA, SC
	# head exp/eend_base/infer/score/mini_librispeech_dev/result_*
#==> exp/eend_base/infer/score/mini_librispeech_dev/result_th0.3_med11_collar0.25 <==
#47.22/0.00/47.20/0.01
#
#==> exp/eend_base/infer/score/mini_librispeech_dev/result_th0.3_med1_collar0.25 <==
#46.36/0.13/46.08/0.15
#
#==> exp/eend_base/infer/score/mini_librispeech_dev/result_th0.4_med11_collar0.25 <==
#43.55/0.04/43.33/0.18
#
#==> exp/eend_base/infer/score/mini_librispeech_dev/result_th0.4_med1_collar0.25 <==
#42.47/0.56/41.34/0.57
#
#==> exp/eend_base/infer/score/mini_librispeech_dev/result_th0.5_med11_collar0.25 <==
#39.13/0.19/38.24/0.69
#
#==> exp/eend_base/infer/score/mini_librispeech_dev/result_th0.5_med1_collar0.25 <==
#38.37/1.64/35.52/1.21
#
#==> exp/eend_base/infer/score/mini_librispeech_dev/result_th0.6_med11_collar0.25 <==
#33.55/1.28/30.56/1.72
#
#==> exp/eend_base/infer/score/mini_librispeech_dev/result_th0.6_med1_collar0.25 <==
#35.17/4.56/28.65/1.96
#
#==> exp/eend_base/infer/score/mini_librispeech_dev/result_th0.7_med11_collar0.25 <== ## as final result
#29.33/8.54/17.15/3.64
#
#==> exp/eend_base/infer/score/mini_librispeech_dev/result_th0.7_med1_collar0.25 <==
#34.65/12.91/18.76/2.98
fi


if [ ${stage} -le 3 ]&&[ ${stop_stage} -ge 3 ];then

    echo "Start model averaging"
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    model_dir=$root_dir/exp/eend_base_100epoch
    init_model=$model_dir/avg.th
    ifiles=`eval echo $model_dir/transformer{90..100}.th`
    python eend/eend/bin/model_averaging.py $init_model $ifiles
fi


# Inferring
if [ $stage -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "Start inferring"
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    infer_conf=$root_dir/eend/conf/base/mini_librispeech_infer.yaml
    test_dir=$root_dir/data/simu/data/dev_clean_2_ns2_beta2_500
    test_model=$root_dir/exp/eend_base_100epoch/avg.th
    infer_out_dir=$root_dir/exp/eend_base_100epoch/infer
    python eend/eend/bin/infer.py -c $infer_conf $test_dir $test_model $infer_out_dir
fi



# Scoring
if [ $stage -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "Start scoring"
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    stck_dir=$root_dir/SCTK-2.4.12/src/md-eval/
    infer_out_dir=$root_dir/exp/eend_base_100epoch/infer
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
   # result:
   # # DER, MS, FA, SC
   #  head  exp/eend_base_100epoch/infer/score/mini_librispeech_dev/result_th0.*
#==> exp/eend_base_100epoch/infer/score/mini_librispeech_dev/result_th0.3_med11_collar0.25 <==
#37.84/2.54/34.00/1.30
#
#==> exp/eend_base_100epoch/infer/score/mini_librispeech_dev/result_th0.3_med1_collar0.25 <==
#37.66/3.19/33.07/1.40
#
#==> exp/eend_base_100epoch/infer/score/mini_librispeech_dev/result_th0.4_med11_collar0.25 <==
#35.06/3.48/29.90/1.68
#
#==> exp/eend_base_100epoch/infer/score/mini_librispeech_dev/result_th0.4_med1_collar0.25 <==
#35.33/4.63/28.93/1.77
#
#==> exp/eend_base_100epoch/infer/score/mini_librispeech_dev/result_th0.5_med11_collar0.25 <==
#32.37/4.85/25.41/2.11
#
#==> exp/eend_base_100epoch/infer/score/mini_librispeech_dev/result_th0.5_med1_collar0.25 <==
#33.40/6.63/24.60/2.17
#
#==> exp/eend_base_100epoch/infer/score/mini_librispeech_dev/result_th0.6_med11_collar0.25 <==
#29.94/6.90/20.47/2.56
#
#==> exp/eend_base_100epoch/infer/score/mini_librispeech_dev/result_th0.6_med1_collar0.25 <==
#32.05/9.46/19.97/2.61
#
#==> exp/eend_base_100epoch/infer/score/mini_librispeech_dev/result_th0.7_med11_collar0.25 <==  ## as final result:
#27.79/10.04/14.57/3.18
#
#==> exp/eend_base_100epoch/infer/score/mini_librispeech_dev/result_th0.7_med1_collar0.25 <==
#31.55/13.67/14.70/3.18
#
fi
