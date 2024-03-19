#!/usr/bin/env bash

stage=0

stop_stage=1000
.  utils/parse_options.sh
. path_for_fairseq.sh

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ];then
   echo "get final phone dictionary"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   dest_dir=dataset/format/librispeech/librispeech_lm_norm_phn_seq/
   python $fairseq_dir/fairseq_cli/preprocess.py\
           --dataset-impl mmap\
           --trainpref $dest_dir/phones/lm.phones.filtered.txt\
           --workers 70\
           --only-source \
           --destdir tests/\
           --srcdict $dest_dir/phones/dict.txt

   echo "<SIL> 0" >>  tests/dict.txt
   #cut -f1 -d ' ' $dest_dir/phoness/dict.txt | awk '{print $0 " " NR-1}' > $dest_dir/phoness/dict.phn.txt

   echo "finish get final phone dictionary !!!!!!!!!!"
fi
