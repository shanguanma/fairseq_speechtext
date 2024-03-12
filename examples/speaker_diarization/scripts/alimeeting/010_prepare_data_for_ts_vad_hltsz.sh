#!/usr/bin/env bash


stage=0
stop_stage=1000
. utils/parse_options.sh
#. path_for_fairseq_speechtext.sh
#. path_for_fsq_speechtext.sh
. path_for_fsq_sptt.sh
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
   data_path=/home/maduo/dataset/alimeeting
   fairseq_dir=/home/maduo/codebase/fairseq_speechtext
   echo " Process dataset: Eval dataset, get json files"
   python $fairseq_dir/examples/speaker_diarization/scripts/prepare_data.py \
    --data_path ${data_path} \
    --type Eval \

fi
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
   data_path=/home/maduo/dataset/alimeeting
   fairseq_dir=/home/maduo/codebase/fairseq_speechtext
    echo " Process dataset: Train dataset, get json files"
   python $fairseq_dir/examples/speaker_diarization/scripts/prepare_data.py \
    --data_path ${data_path} \
    --type Train  

fi

