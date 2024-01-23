#!/usr/bin/env bash

stage=0
stop_stage=1000

. utils/parse_options.sh
. path_for_fairseq.sh

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
   echo "normalize librispeech-lm-norm using fasttext model"
   lg=en
   lid_path=dataset/librispeech/lid.176.bin
   input_text_dir=dataset/librispeech/
   output_text_dir=dataset/format/librispeech/librispeech_lm_norm_phn_seq/
   for name in librispeech-lm-norm;do 
     python source_md/wav2vec-u2/text/normalize_and_filter_text.py\
              --lang $lg\
              --fasttext-model $lid_path\
              --text $input_text_dir/${name}.txt \
              --output $output_text_dir/${name}.lid.tmp.txt
     cat $input_text_dir/${name}.lid.tmp.txt | grep -v '\-\-\-'>$input_text_dir/${name}.lid.txt
   done
fi 
