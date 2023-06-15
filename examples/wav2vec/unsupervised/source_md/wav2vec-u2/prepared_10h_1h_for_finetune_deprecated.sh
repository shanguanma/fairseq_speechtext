#!/usr/bin/env bash

stage=0
stop_stage=100
. utils/parse_options.sh

if [ ${stage} -le 0 ] &&  [ ${stop_stage}  -ge 0 ];then
  echo "prepare 10h tsv file for speech"
  tsv_dir=dataset/format/librispeech/
  head -n 2780 $tsv_dir/train-clean-100.tsv > $tsv_dir/train-clean-10h_tmp1.tsv
  sed '1d' $tsv_dir/train-clean-10h_tmp1.tsv > $tsv_dir/train-clean-10h_tmp.tsv
  cat $tsv_dir/train-clean-10h_tmp.tsv | awk '{x+=$2} END{print x/16000/3600}'
  head -n 1 $tsv_dir/train-clean-100.tsv > $tsv_dir/train-clean-10h_head.tsv
  cat $tsv_dir/train-clean-10h_head.tsv  $tsv_dir/train-clean-10h_tmp.tsv >  $tsv_dir/train-clean-10h.tsv
  wc -l  $tsv_dir/train-clean-10h.tsv
  echo "prepare 10h ltr file for text"
  head -n 2779 $tsv_dir/train-clean-100.ltr  > $tsv_dir/train-clean-10h.ltr
  rm $tsv_dir/train-clean-10h_tmp1.tsv $tsv_dir/train-clean-10h_tmp.tsv $tsv_dir/train-clean-10h_head.tsv
  wc -l $tsv_dir/{train-clean-10h.tsv,train-clean-10h.ltr}

fi

if [ ${stage} -le 1 ] &&  [ ${stop_stage}  -ge 1 ];then
  echo "prepare 1h tsv file for speech"
  tsv_dir=dataset/format/librispeech/
  head -n 275 $tsv_dir/train-clean-100.tsv > $tsv_dir/train-clean-1h_tmp1.tsv
  sed '1d' $tsv_dir/train-clean-1h_tmp1.tsv > $tsv_dir/train-clean-1h_tmp.tsv
  cat $tsv_dir/train-clean-1h_tmp.tsv | awk '{x+=$2} END{print x/16000/3600}'
  head -n 1 $tsv_dir/train-clean-100.tsv > $tsv_dir/train-clean-1h_head.tsv
  cat $tsv_dir/train-clean-1h_head.tsv  $tsv_dir/train-clean-1h_tmp.tsv >  $tsv_dir/train-clean-1h.tsv
  wc -l  $tsv_dir/train-clean-1h.tsv
  echo "prepare 10h ltr file for text"
  head -n 274 $tsv_dir/train-clean-100.ltr  > $tsv_dir/train-clean-1h.ltr
  rm $tsv_dir/train-clean-1h_tmp1.tsv $tsv_dir/train-clean-1h_tmp.tsv $tsv_dir/train-clean-1h_head.tsv
  wc -l $tsv_dir/{train-clean-1h.tsv,train-clean-1h.ltr}

fi

