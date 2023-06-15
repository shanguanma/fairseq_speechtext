#!/usr/bin/env bash

stage=0
stop_stage=1000
. utils/parse_options.sh

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
   #datasets="train-clean-100 train-clean-360 train-other-500 dev-clean dev-other test-clean test-other"
   datasets="dev-clean dev-other test-clean test-other"
   tsv_dir=dataset/format/librispeech/
   for name in $datasets;do
     ## it outputs *.phn file
    cat $tsv_dir/${name}.wrd |  python source_md/wav2vec-u2/text/g2p_wrd_to_phn.py >$tsv_dir/${name}.phn             
   done
fi
