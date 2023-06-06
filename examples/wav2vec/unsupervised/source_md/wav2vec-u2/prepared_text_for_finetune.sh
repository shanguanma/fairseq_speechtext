#!/usr/bin/env bash

stage=0
stop_stage=1000
. utils/parse_options.sh
. path_for_fairseq_speechtext.sh  ### it contain pytorch2.0  ,fairseq
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   
   for name in dev-clean dev-other test-clean test-other train-960;do
     python3 source_md/wav2vec-u2/text/libri_labels.py\
         $tsv_dir/${name}.tsv\
         --output-dir $tsv_dir\
         --output-name $name
    echo "finish $name !!!!" 
   done 

fi

## prepared  finetune stage dictionary
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
  echo  "prepare dict text file for finetune"
  # this file is from https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt
  tsv_dir=/workspace2/maduo/dataset/format/librispeech
  dict=$tsv_dir/dict.ltr.txt
  cat <<EOF>$dict
| 94802
E 51860
T 38431
A 33152
O 31495
N 28855
I 28794
H 27187
S 26071
R 23546
D 18289
L 16308
U 12400
M 10685
W 10317
C 9844
F 9062
G 8924
Y 8226
P 6890
B 6339
V 3936
K 3456
' 1023
X 636
J 598
Q 437
Z 213
EOF

fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
   tsv_dir=/workspace2/maduo/dataset/format/librispeech

   for name in train-clean-100;do
     python3 source_md/wav2vec-u2/text/libri_labels.py\
         $tsv_dir/${name}.tsv\
         --output-dir $tsv_dir\
         --output-name $name
    echo "finish $name !!!!"
   done

fi 
