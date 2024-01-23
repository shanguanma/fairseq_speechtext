#!/usr/bin/env bash


stage=0

stop_stage=1000

. utils/parse_options.sh
#. path_for_fairseq_speechtext.sh
. path_for_fsq_speechtext.sh
#FAIRSEQ_ROOT=/workspace2/maduo/fairseq

### so I will offline get iter1 hubert specify layer feature , then train kmeans model.
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
  echo "step1: get using 10% train-960.tsv hubert feature"
  root_dir=/workspace2/maduo/
  tsv_dir=$root_dir/dataset/format/librispeech
  feat_dir=$tsv_dir/feat_dir/huber_6layer_feat_of_0.1_960h_librispeech_250k_update_from_mfcc_offical_setting
  ckpt_path=$root_dir/exp/pretrain/pretrain_on_hubert_4gpu_8update_960h_mfcc_250k_update_offical/checkpoint_181_250000.pt 
  portion=0.1
  sample_rate=16000
  layer=6
  mkdir -p $feat_dir
  for name in  train-960;do
    python source_md/wav2vec-u2/dump_hubert_feature.py \
      --tsvfile $tsv_dir/${name}.tsv\
      --ckpt-path $ckpt_path\
      --feat-dir $feat_dir\
      --portion ${portion}\
      --sample-rate $sample_rate\
      --layer $layer 
  done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
    echo "learn  kmeans model on hubert feature"
    tsv_dir=/workspace2/maduo/dataset/format/librispeech
    feat_dir=$tsv_dir/feat_dir/huber_6layer_feat_of_0.1_960h_librispeech_250k_update_from_mfcc_offical_setting
    km_path=$feat_dir/librispeech-960h_0.1_portion_250k_update_hubert_6layer_km_500_clusters.mdl
    python source_md/wav2vec-u2/learn_kmeans_on_hubert.py \
        --n-cluster 500\
        --km-path $km_path\
        --feat-dir $feat_dir
         
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
  echo "step3: get k-means pesudo target from k-means model"
  tsv_dir=/workspace2/maduo/dataset/format/librispeech
  ckpt_path=/workspace2/maduo/exp/pretrain/pretrain_on_hubert_4gpu_8update_960h_mfcc_250k_update_offical/checkpoint_181_250000.pt
  feat_dir=$tsv_dir/feat_dir/huber_6layer_feat_of_0.1_960h_librispeech_250k_update_from_mfcc_offical_setting
  km_path=$feat_dir/librispeech-960h_0.1_portion_250k_update_hubert_6layer_km_500_clusters.mdl
  label_dir=$feat_dir/label_dir/

  sample_rate=16000
  layer=6
  mkdir -p $label_dir
  #for name in train-960 dev-clean dev-other ;do
  for name in train-clean-360;do 
    python source_md/wav2vec-u2/dump_pseudo_label_on_hubert.py\
      --tsvfile $tsv_dir/${name}.tsv \
      --sample-rate ${sample_rate}\
      --label-path ${label_dir}/$name.km\
      --km-path $km_path \
      --ckpt-path $ckpt_path\
      --layer $layer
     
      
  done
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
   echo "create a dummpy dictionary"
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   lab_dir=$tsv_dir/feat_dir/huber_6layer_feat_of_0.1_960h_librispeech_250k_update_from_mfcc_offical_setting/label_dir
   n_cluster=500
   for x in $(seq 0 $((n_cluster - 1 )));do
     echo "$x 1"
   done>>$lab_dir/dict.km.txt
fi

## prepared  finetune stage dictionary
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then
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




### so I will offline get iter1 hubert specify layer feature , then train kmeans model.
if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ];then
  echo "step1: get using 10% train-960.tsv hubert feature"
  root_dir=/workspace2/maduo/
  tsv_dir=$root_dir/dataset/format/librispeech
  feat_dir=$tsv_dir/feat_dir/huber_9layer_feat_of_0.1_960h_librispeech_250k_update_from_mfcc_offical_setting
  ckpt_path=$root_dir/exp/pretrain/pretrain_on_hubert_4gpu_8update_960h_mfcc_250k_update_offical/checkpoint_181_250000.pt
  portion=0.1
  sample_rate=16000
  layer=9
  mkdir -p $feat_dir
  for name in  train-960;do
    python source_md/wav2vec-u2/dump_hubert_feature.py \
      --tsvfile $tsv_dir/${name}.tsv\
      --ckpt-path $ckpt_path\
      --feat-dir $feat_dir\
      --portion ${portion}\
      --sample-rate $sample_rate\
      --layer $layer
  done
fi
if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ];then
    echo "learn  kmeans model on hubert feature"
    tsv_dir=/workspace2/maduo/dataset/format/librispeech
    feat_dir=$tsv_dir/feat_dir/huber_9layer_feat_of_0.1_960h_librispeech_250k_update_from_mfcc_offical_setting
    km_path=$feat_dir/librispeech-960h_0.1_portion_250k_update_hubert_9layer_km_500_clusters.mdl
    python source_md/wav2vec-u2/learn_kmeans_on_hubert.py \
        --n-cluster 500\
        --km-path $km_path\
        --feat-dir $feat_dir

fi

if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ];then
  echo "step13: get k-means pesudo target from k-means model"
  tsv_dir=/workspace2/maduo/dataset/format/librispeech
  ckpt_path=/workspace2/maduo/exp/pretrain/pretrain_on_hubert_4gpu_8update_960h_mfcc_250k_update_offical/checkpoint_181_250000.pt
  feat_dir=$tsv_dir/feat_dir/huber_9layer_feat_of_0.1_960h_librispeech_250k_update_from_mfcc_offical_setting
  km_path=$feat_dir/librispeech-960h_0.1_portion_250k_update_hubert_9layer_km_500_clusters.mdl
  label_dir=$feat_dir/label_dir/

  sample_rate=16000
  layer=9
  mkdir -p $label_dir
  for name in train-960 dev-clean dev-other ;do
    python source_md/wav2vec-u2/dump_pseudo_label_on_hubert.py\
      --tsvfile $tsv_dir/${name}.tsv \
      --sample-rate ${sample_rate}\
      --label-path ${label_dir}/$name.km\
      --km-path $km_path \
      --ckpt-path $ckpt_path\
      --layer $layer


  done
fi
if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ];then
   echo "create a dummpy dictionary"
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   lab_dir=$tsv_dir/feat_dir/huber_9layer_feat_of_0.1_960h_librispeech_250k_update_from_mfcc_offical_setting/label_dir
   n_cluster=500
   for x in $(seq 0 $((n_cluster - 1 )));do
     echo "$x 1"
   done>>$lab_dir/dict.km.txt
fi


### 2023.6.27
### so I will use offical offer hubert base model  and its kmeans model to get target code.
if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ];then
    echo "step15: get k-means pesudo target from offical k-means model and offical hubert base model"
  tsv_dir=/workspace2/maduo/dataset/format/librispeech
  ckpt_path=/workspace2/maduo/model_hub/librispeech/hubert_base_librispeech_offical_no_finetune/hubert_base_ls960.pt
  feat_dir=$tsv_dir/feat_dir/9layer_feat_from_offical_hubert_base_model
  km_path=/workspace2/maduo/model_hub/librispeech/hubert_base_librispeech_offical_no_finetune/hubert_base_ls960_L9_km500.bin
  label_dir=$feat_dir/label_dir/

  sample_rate=16000
  layer=9
  mkdir -p $label_dir
  #for name in train-960 dev-clean dev-other ;do
  #for name in train-clean-100 train-clean-360 train-clean-500;do 
  for name in train-other-500;do
    python source_md/wav2vec-u2/dump_pseudo_label_on_hubert.py\
      --tsvfile $tsv_dir/${name}.tsv \
      --sample-rate ${sample_rate}\
      --label-path ${label_dir}/$name.km\
      --km-path $km_path \
      --ckpt-path $ckpt_path\
      --layer $layer


  done
fi
if [ ${stage} -le 16 ] && [ ${stop_stage} -ge 16 ];then
   echo "create a dummpy dictionary"
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   lab_dir=$tsv_dir/feat_dir/9layer_feat_from_offical_hubert_base_model/label_dir
   n_cluster=500
   for x in $(seq 0 $((n_cluster - 1 )));do
     echo "$x 1"
   done>>$lab_dir/dict.km.txt
fi


## 2023-11-13 get voicelm(only phone) specify layer representation kmeans label
if [ ${stage} -le 20 ] && [ ${stop_stage} -ge 20 ];then
  echo "step1: get using 10% train-960.tsv hubert feature"
  root_dir=/workspace2/maduo/
  tsv_dir=$root_dir/dataset/format/librispeech
  feat_dir=$tsv_dir/feat_dir/voicelm_only_phone_feat_of_0.1_960h_librispeech_400k_update/
  ckpt_path=$root_dir/exp/pretrain/pretrain_on_base_sthubert_4gpu_8update_400k_w2vu2_librispeech_monophncode/checkpoint_289_400000.pt
  portion=0.1
  sample_rate=16000
  layer=7
  mkdir -p $feat_dir
  for name in  train-960;do
    python source_md/wav2vec-u2/dump_hubert_feature.py \
      --tsvfile $tsv_dir/${name}.tsv\
      --ckpt-path $ckpt_path\
      --feat-dir $feat_dir\
      --portion ${portion}\
      --sample-rate $sample_rate\
      --layer $layer
  done
fi

if [ ${stage} -le 21 ] && [ ${stop_stage} -ge 21 ];then
    echo "learn  kmeans model on hubert feature"
    tsv_dir=/workspace2/maduo/dataset/format/librispeech
    feat_dir=$tsv_dir/feat_dir/voicelm_only_phone_feat_of_0.1_960h_librispeech_400k_update/
    km_path=$feat_dir/librispeech-960h_0.1_portion_400k_update_voicelm_only_phone_7layer_km_500_clusters.mdl
    python source_md/wav2vec-u2/learn_kmeans_on_hubert.py \
        --n-cluster 500\
        --km-path $km_path\
        --feat-dir $feat_dir

fi
