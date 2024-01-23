#!/usr/bin/env bash


stage=0

stop_stage=1000

. utils/parse_options.sh
. path_for_fairseq.sh
FAIRSEQ_ROOT=/workspace2/maduo/fairseq

### so I will online get mfcc feature , then train kmeans model.
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
  echo "step2: train k-means model using 10% train-960.tsv mfcc feature"
  tsv_dir=/workspace2/maduo/dataset/format/librispeech
  feat_dir=$tsv_dir/mfcc
  km_path=$tsv_dir/mfcc/librispeech-960h_0.1_portion_mfcc_km_100_clusters.mdl
  n_cluster=100
  nj=35
  sample_rate=16000
  portion=0.1
  for name in  train-960;do
    python source_md/wav2vec-u2/sklearn_kmeans_on_mfcc.py \
      --tsvfile $tsv_dir/${name}.tsv\
      --n-clusters ${n_cluster}\
      --nj $nj \
      --sample-rate ${sample_rate}\
      --portion ${portion}\
      --km-path ${km_path}
  done
fi





if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
  echo "step3: get k-means pesudo target from k-means model"
  tsv_dir=/workspace2/maduo/dataset/format/librispeech
  km_path=$tsv_dir/mfcc/librispeech-960h_0.1_portion_mfcc_km_100_clusters.mdl
  nj=35
  sample_rate=16000  
  lab_dir=$tsv_dir/mfcc/mfcc_lab
  mkdir  -p $lab_dir
  for name in train-960 dev-clean dev-other ;do
    python source_md/wav2vec-u2/dump_pseudo_label_on_mfcc.py\
      --tsvfile $tsv_dir/${name}.tsv \
      --nj $nj \
      --sample-rate ${sample_rate}\
      --label-path ${lab_dir}/$name.km\
      --km-path $km_path 
      
  done
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
   echo "create a dummpy dictionary"
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   lab_dir=$tsv_dir/mfcc/mfcc_lab
   n_cluster=100
   for x in $(seq 0 $((n_cluster - 1 )));do
     echo "$x 1"
   done>>$lab_dir/dict.km.txt
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then
  echo "step3: get k-means pesudo target from k-means model"
  tsv_dir=/workspace2/maduo/dataset/format/librispeech
  km_path=$tsv_dir/mfcc/librispeech-960h_0.1_portion_mfcc_km_100_clusters.mdl
  nj=35
  sample_rate=16000
  lab_dir=$tsv_dir/mfcc/mfcc_lab
  mkdir  -p $lab_dir
  for name in train-clean-100 ;do
    python source_md/wav2vec-u2/dump_pseudo_label_on_mfcc.py\
      --tsvfile $tsv_dir/${name}.tsv \
      --nj $nj \
      --sample-rate ${sample_rate}\
      --label-path ${lab_dir}/$name.km\
      --km-path $km_path

  done
fi


