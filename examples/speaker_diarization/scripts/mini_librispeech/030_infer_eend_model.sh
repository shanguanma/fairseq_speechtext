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
    ifiles=`eval echo $model_dir/transformer{91..100}.th`
    python eend/eend/bin/model_averaging.py $init_model $ifiles
fi


# Inferring
if [ $stage -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Start inferring"
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    infer_conf=$root_dir/eend/conf/base/mini_librispeech_infer.yaml
    test_dir=$root_dir/data/simu/data/dev_clean_2_ns2_beta2_500
    test_model=$model_adapt_dir/avg.th
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
fi
