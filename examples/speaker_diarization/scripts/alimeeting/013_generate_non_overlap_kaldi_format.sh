#!/usr/bin/env bash


stage=0
stop_stage=1000
. utils/parse_options.sh
#. path_for_fairseq_speechtext.sh
#. path_for_fsq_speechtext.sh
. path_for_fsq_sptt.sh
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
   echo "prepare train target audio list"
   #fairseq_dir=/home/maduo/codebase/fairseq_speechtext
   #input_dir=/data/alimeeting/Train_Ali/Train_Ali_far/target_audio/
   #file=$input_dir/wavs.txt
   data_dir=/data/alimeeting
   for name in Eval Train;do
    python source_md/prepare_non_overlapped_single_speaker_speech_from_alimeeting.py\
            --data_path \
	    --type $name
   done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
  echo "prepare format Eval data for alimeeting non_overlap"
  data_dir=/data/alimeeting/Eval_Ali/Eval_Ali_far/non_overlap/
  [ -f ${data_dir}/wav.scp ] && rm ${data_dir}/wav.scp
  [ -f ${data_dir}/utt2spk ] && rm ${data_dir}/utt2spk
  [ -f ${data_dir}/spk2utt ] && rm ${data_dir}/spk2utt
  for f in `find ${data_dir}  -name "*.wav"`;do
      speaker_id=$(basename $f | sed s:.wav$::)
      last_fold=$(dirname $f | awk -F'/' '{print $(NF)}')
      wav_id=${last_fold}_${speaker_id}
      echo "$wav_id $f" | sort >> ${data_dir}/wav.scp
      echo "$wav_id $speaker_id" | sort >> ${data_dir}/utt2spk
  done       
 source_md/utt2spk_to_spk2utt.pl ${data_dir}/utt2spk >${data_dir}/spk2utt 
 
fi
