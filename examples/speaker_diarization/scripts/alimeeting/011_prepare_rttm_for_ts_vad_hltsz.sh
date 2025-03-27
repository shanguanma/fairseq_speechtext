#!/usr/bin/env bash


stage=0
stop_stage=1000
. utils/parse_options.sh
#. path_for_fairseq_speechtext.sh
#. path_for_fsq_speechtext.sh
. path_for_fsq_sptt.sh
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

## it is modified from https://github.com/yufan-aslp/AliMeeting/blob/main/speaker/run.sh
if [ $stage -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "Process textgrid to obtain rttm label"
    audio_dir=/data/alimeeting/Eval_Ali/Eval_Ali_far/audio_dir
    textgrid_dir=/data/alimeeting/Eval_Ali/Eval_Ali_far/textgrid_dir
    work_dir=/home/maduo/tests/alimeeting/eval
    fairseq_dir=/home/maduo/codebase/fairseq_speechtext/
    mkdir -p $work_dir
    find -L $audio_dir -name "*.wav" > $work_dir/wavlist
    sort  $work_dir/wavlist > $work_dir/tmp
    cp $work_dir/tmp $work_dir/wavlist
    awk -F '/' '{print $NF}' $work_dir/wavlist | awk -F '.' '{print $1}' > $work_dir/uttid


    find -L $textgrid_dir -iname "*.TextGrid" >  $work_dir/textgrid.flist
    sort  $work_dir/textgrid.flist  > $work_dir/tmp
    cp $work_dir/tmp $work_dir/textgrid.flist
    paste $work_dir/uttid $work_dir/textgrid.flist > $work_dir/uttid_textgrid.flist
    while read line;do
    #for line in $(cat $work_dir/uttid_textgrid.flist) do
        text_grid=`echo $line | awk '{print $1}'`
        text_grid_path=`echo $line | awk '{print $2}'`
	echo "text_grid: $text_grid"
	echo "text_grid_path: ${text_grid_path}"
        python $fairseq_dir/examples/speaker_diarization/source_md/make_textgrid_rttm.py\
		--input_textgrid_file $text_grid_path \
		--uttid $text_grid \
		--output_rttm_file $work_dir/${text_grid}.rttm
    done < $work_dir/uttid_textgrid.flist
    cat $work_dir/*.rttm > $work_dir/all.rttm1
    #mv tests/alimeeting/all.rttm1 tests/alimeeting/alimeeting.rttm ## it is ground truth of alimeeting eval
    mv $work_dir/all.rttm1  $work_dir/alimeeting_eval.rttm1
    mv $work_dir/alimeeting_eval.rttm1 /home/maduo/model_hub/ts_vad/
    mv /home/maduo/model_hub/ts_vad/alimeeting_eval.rttm1  /home/maduo/model_hub/ts_vad/alimeeting_eval.rttm
fi


if [ $stage -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Process textgrid to obtain rttm label"
    audio_dir=/data/alimeeting/Test_Ali/Test_Ali_far/audio_dir
    textgrid_dir=/data/alimeeting/Test_Ali/Test_Ali_far/textgrid_dir
    work_dir=/home/maduo/tests/alimeeting/test
    fairseq_dir=/home/maduo/codebase/fairseq_speechtext/
    mkdir -p $work_dir
    find -L $audio_dir -name "*.wav" > $work_dir/wavlist
    sort  $work_dir/wavlist > $work_dir/tmp
    cp $work_dir/tmp $work_dir/wavlist
    awk -F '/' '{print $NF}' $work_dir/wavlist | awk -F '.' '{print $1}' > $work_dir/uttid


    find -L $textgrid_dir -iname "*.TextGrid" >  $work_dir/textgrid.flist
    sort  $work_dir/textgrid.flist  > $work_dir/tmp
    cp $work_dir/tmp $work_dir/textgrid.flist
    paste $work_dir/uttid $work_dir/textgrid.flist > $work_dir/uttid_textgrid.flist
    while read line;do
    #for line in $(cat $work_dir/uttid_textgrid.flist) do
        text_grid=`echo $line | awk '{print $1}'`
        text_grid_path=`echo $line | awk '{print $2}'`
        echo "text_grid: $text_grid"
        echo "text_grid_path: ${text_grid_path}"
        python $fairseq_dir/examples/speaker_diarization/source_md/make_textgrid_rttm.py\
                --input_textgrid_file $text_grid_path \
                --uttid $text_grid \
                --output_rttm_file $work_dir/${text_grid}.rttm
    done < $work_dir/uttid_textgrid.flist
    cat $work_dir/*.rttm > $work_dir/all.rttm1
    #mv tests/alimeeting/all.rttm1 tests/alimeeting/alimeeting.rttm ## it is ground truth of alimeeting eval
    mv $work_dir/all.rttm1  $work_dir/alimeeting_test.rttm1
    mv $work_dir/alimeeting_test.rttm1 /home/maduo/model_hub/ts_vad/
    mv /home/maduo/model_hub/ts_vad/alimeeting_test.rttm1 /home/maduo/model_hub/ts_vad/alimeeting_test.rttm
fi

