#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail
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
    exp_name=eend_m2f_model_10epoch_debug_1
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    model_dir=$root_dir/exp/$exp_name
    dst_model=$model_dir/best_avg.pt
    avg_num=2
    remove_models='True'
    python eend/eend/bin/model_averaging2.py \
            --dst_model $dst_model\
            --src_path $model_dir\
            --nums $avg_num\
            --remove_models $remove_models

fi

# Inferring
if [ $stage -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Start inferring"
    exp_name=eend_m2f_model_10epoch_debug_1
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    infer_conf=$root_dir/eend/conf/base/mini_librispeech_mask_infer.yaml
    test_dir=$root_dir/data/simu/data/dev_clean_2_ns2_beta2_500
    test_model=$root_dir/exp/$exp_name/best_avg.pt
    infer_out_dir=$root_dir/exp/$exp_name/infer
    model_type="eend_m2f"
    python eend/eend/bin/infer_mask_model.py -c $infer_conf \
	    $test_dir \
	    $test_model \
	    $infer_out_dir \
	    --model_type $model_type
fi

if [ $stage -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Start scoring"
    exp_name=eend_m2f_model_10epoch_debug_1
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    stck_dir=$root_dir/SCTK-2.4.12/src/md-eval/
    infer_out_dir=$root_dir/exp/$exp_name/infer
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
#
#result: DER,MS,FA,SC
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_m2f_model_10epoch_debug_1/infer/score/mini_librispeech_dev/result_th0.3_med11_collar0.25 <==
#48.61/2.56/45.70/0.36
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_m2f_model_10epoch_debug_1/infer/score/mini_librispeech_dev/result_th0.3_med1_collar0.25 <==
#56.38/19.16/36.59/0.62
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_m2f_model_10epoch_debug_1/infer/score/mini_librispeech_dev/result_th0.4_med11_collar0.25 <==
#59.31/34.65/22.69/1.97
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_m2f_model_10epoch_debug_1/infer/score/mini_librispeech_dev/result_th0.4_med1_collar0.25 <==
#65.61/38.75/26.16/0.70
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_m2f_model_10epoch_debug_1/infer/score/mini_librispeech_dev/result_th0.5_med11_collar0.25 <==
#81.28/71.62/9.06/0.61
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_m2f_model_10epoch_debug_1/infer/score/mini_librispeech_dev/result_th0.5_med1_collar0.25 <==
#73.55/54.79/17.99/0.77
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_m2f_model_10epoch_debug_1/infer/score/mini_librispeech_dev/result_th0.6_med11_collar0.25 <==
#88.15/83.38/4.47/0.29
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_m2f_model_10epoch_debug_1/infer/score/mini_librispeech_dev/result_th0.6_med1_collar0.25 <==
#81.45/70.99/9.58/0.88
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_m2f_model_10epoch_debug_1/infer/score/mini_librispeech_dev/result_th0.7_med11_collar0.25 <==
#94.36/93.09/1.00/0.27
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_m2f_model_10epoch_debug_1/infer/score/mini_librispeech_dev/result_th0.7_med1_collar0.25 <==
#90.83/87.02/3.23/0.57
fi

if [ ${stage} -le 11 ]&&[ ${stop_stage} -ge 11 ];then
    echo "Start model averaging"
    exp_name=eend_m2f_model_100epoch_debug_1
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    model_dir=$root_dir/exp/$exp_name
    dst_model=$model_dir/best_avg.pt
    avg_num=10
    remove_models='True'
    python eend/eend/bin/model_averaging2.py \
            --dst_model $dst_model\
            --src_path $model_dir\
            --nums $avg_num\
            --remove_models $remove_models

fi

# Inferring
if [ $stage -le 12 ] && [ ${stop_stage} -ge 12 ]; then
    echo "Start inferring"
    exp_name=eend_m2f_model_100epoch_debug_1
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    infer_conf=$root_dir/eend/conf/base/mini_librispeech_mask_infer.yaml
    test_dir=$root_dir/data/simu/data/dev_clean_2_ns2_beta2_500
    #test_model=$root_dir/exp/$exp_name/best_avg.pt
    test_model=$root_dir/exp/$exp_name/model_87.pt
    infer_out_dir=$root_dir/exp/$exp_name/infer
    model_type="eend_m2f"
    python eend/eend/bin/infer_mask_model.py -c $infer_conf \
            $test_dir \
            $test_model \
            $infer_out_dir \
            --model_type $model_type
fi


if [ $stage -le 13 ] && [ ${stop_stage} -ge 13 ]; then
    echo "Start scoring"
    exp_name=eend_m2f_model_100epoch_debug_1
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    stck_dir=$root_dir/SCTK-2.4.12/src/md-eval/
    infer_out_dir=$root_dir/exp/$exp_name/infer
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

fi

if [ ${stage} -le 14 ]&&[ ${stop_stage} -ge 14 ];then
    echo "Start model averaging"
    exp_name=eend_m2f_model_10epoch_debug_subsample1
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    model_dir=$root_dir/exp/$exp_name
    dst_model=$model_dir/best_avg.pt
    avg_num=2
    remove_models='True'
    python eend/eend/bin/model_averaging2.py \
            --dst_model $dst_model\
            --src_path $model_dir\
            --nums $avg_num\
            --remove_models $remove_models

fi

# Inferring
if [ $stage -le 15 ] && [ ${stop_stage} -ge 15 ]; then
    echo "Start inferring"
    exp_name=eend_m2f_model_10epoch_debug_subsample1
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    infer_conf=$root_dir/eend/conf/base/mini_librispeech_mask_infer.yaml
    test_dir=$root_dir/data/simu/data/dev_clean_2_ns2_beta2_500
    test_model=$root_dir/exp/$exp_name/best_avg.pt
    #test_model=$root_dir/exp/$exp_name/model_87.pt
    infer_out_dir=$root_dir/exp/$exp_name/infer
    model_type="eend_m2f"
    python eend/eend/bin/infer_mask_model.py -c $infer_conf \
            $test_dir \
            $test_model \
            $infer_out_dir \
            --model_type $model_type
fi
if [ $stage -le 16 ] && [ ${stop_stage} -ge 16 ]; then
    echo "Start scoring"
    exp_name=eend_m2f_model_10epoch_debug_subsample1
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    stck_dir=$root_dir/SCTK-2.4.12/src/md-eval/
    infer_out_dir=$root_dir/exp/$exp_name/infer
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
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_m2f_model_10epoch_debug_subsample1/infer/score/mini_librispeech_dev/result_th0.3_med11_collar0.25 <==
#49.04/0.00/49.04/0.00
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_m2f_model_10epoch_debug_subsample1/infer/score/mini_librispeech_dev/result_th0.3_med1_collar0.25 <==
#49.04/0.00/49.04/0.00
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_m2f_model_10epoch_debug_subsample1/infer/score/mini_librispeech_dev/result_th0.4_med11_collar0.25 <==
#49.04/0.00/49.04/0.00
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_m2f_model_10epoch_debug_subsample1/infer/score/mini_librispeech_dev/result_th0.4_med1_collar0.25 <==
#49.04/0.00/49.04/0.00
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_m2f_model_10epoch_debug_subsample1/infer/score/mini_librispeech_dev/result_th0.5_med11_collar0.25 <==
#49.04/0.00/49.04/0.00
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_m2f_model_10epoch_debug_subsample1/infer/score/mini_librispeech_dev/result_th0.5_med1_collar0.25 <==
#49.04/0.00/49.04/0.00
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_m2f_model_10epoch_debug_subsample1/infer/score/mini_librispeech_dev/result_th0.6_med11_collar0.25 <==
#49.04/0.00/49.04/0.00
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_m2f_model_10epoch_debug_subsample1/infer/score/mini_librispeech_dev/result_th0.6_med1_collar0.25 <==
#49.04/0.00/49.04/0.00
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_m2f_model_10epoch_debug_subsample1/infer/score/mini_librispeech_dev/result_th0.7_med11_collar0.25 <==
#49.04/0.00/49.04/0.00
#
#==> /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/eend_m2f_model_10epoch_debug_subsample1/infer/score/mini_librispeech_dev/result_th0.7_med1_collar0.25 <==
#49.04/0.00/49.04/0.00
fi

