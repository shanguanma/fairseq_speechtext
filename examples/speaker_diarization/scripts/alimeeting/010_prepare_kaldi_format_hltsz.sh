#!/usr/bin/env bash


stage=0
stop_stage=1000
. utils/parse_options.sh
#. path_for_fairseq_speechtext.sh
#. path_for_fsq_speechtext.sh
. path_for_fsq_sptt.sh
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
## modified from https://github.com/yufan-aslp/AliMeeting/blob/main/speaker/run.sh
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
    echo "prepared alimeeting lhotse format mono data"
    corpus_dir=/data/alimeeting
    dest_dir=data/kaldi_alimeeting
    mkdir -p $corpus_dir
    lhotse prepare ali-meeting --mic far $corpus_dir $dest_dir 
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
   echo "export kaldi format from lhotse format"
   data_dir=data/kaldi_alimeeting
   for name in train eval test;do 
      output_dir=data/kaldi_alimeeting/$name
      lhotse kaldi export $data_dir/alimeeting-far_recordings_${name}.jsonl.gz $data_dir/alimeeting-far_supervisions_${name}.jsonl.gz $output_dir

      ## need to kaldi environment
      utils/utt2spk_to_spk2utt.pl $output_dir/utt2spk > $output_dir/spk2utt
      utils/fix_data_dir.sh $output_dir     
   done
fi

