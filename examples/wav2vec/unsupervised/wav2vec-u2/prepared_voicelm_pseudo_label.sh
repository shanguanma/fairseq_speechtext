#!/usr/bin/env bash


stage=0

stop_stage=1000

. utils/parse_options.sh
. path_for_fairseq_speechtext.sh
#FAIRSEQ_ROOT=/workspace2/maduo/fairseq

### so I will offline get iter1 voicelm specify layer feature , then train kmeans model.
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
  echo "step1: get using 10% train-960.tsv voicelm feature"
  root_dir=/workspace2/maduo/
  tsv_dir=$root_dir/dataset/format/librispeech
  ckpt_path=$root_dir/exp/pretrain/pretrain_on_base_imls-ssl_4gpu_8update_960h_400k_update/checkpoint_298_400000.pt
  portion=0.1
  sample_rate=16000
  for layer in 7 8 9 10;do
   feat_dir=$tsv_dir/feat_dir/voicelm_feat_of_0.1_960h_librispeech_400k_update
   mkdir -p $feat_dir
   for name in  train-960;do
    python source_md/wav2vec-u2/dump_voicelm_feature.py \
      --tsvfile $tsv_dir/${name}.tsv\
      --ckpt-path $ckpt_path\
      --feat-dir $feat_dir\
      --portion ${portion}\
      --sample-rate $sample_rate\
      --layer $layer
  done
 done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
    echo "learn  kmeans model on voicelm feature"
    tsv_dir=/workspace2/maduo/dataset/format/librispeech
    feat_dir=$tsv_dir/feat_dir/voicelm_feat_of_0.1_960h_librispeech_400k_update
    
    for layer in 7 8 9 10;do
     km_path=$feat_dir/librispeech-960h_0.1_portion_400k_update_voicelm_${layer}layer_km_500_clusters.mdl
     python source_md/wav2vec-u2/learn_kmeans_on_voicelm.py \
        --n-cluster 500\
        --km-path $km_path\
        --feats $feat_dir/train-960_10_percent_voicelm_${layer}layer_raw_feature.npy
    done

fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
  echo "step3: get k-means pesudo target from k-means model"
  tsv_dir=/workspace2/maduo/dataset/format/librispeech
  ckpt_path=/workspace2/maduo/exp/pretrain/pretrain_on_base_imls-ssl_4gpu_8update_960h_400k_update/checkpoint_298_400000.pt
  feat_dir=$tsv_dir/feat_dir/voicelm_feat_of_0.1_960h_librispeech_400k_update
  
  sample_rate=16000
  #for layer in 7 8 9 10;do
  for layer in 10;do
   label_dir=$feat_dir/voicelm_${layer}layer_label_dir
   mkdir -p $label_dir
   km_path=$feat_dir/librispeech-960h_0.1_portion_400k_update_voicelm_${layer}layer_km_500_clusters.mdl
  for name in train-960 dev-clean dev-other ;do
    python source_md/wav2vec-u2/dump_pseudo_label_on_voicelm.py\
      --tsvfile $tsv_dir/${name}.tsv \
      --sample-rate ${sample_rate}\
      --label-path ${label_dir}/$name.km\
      --km-path $km_path \
      --ckpt-path $ckpt_path\
      --layer $layer


  done
 done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
   echo "create a dummpy dictionary"
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   n_cluster=500
   for layer in 7 8 9 10;do
    lab_dir=$tsv_dir/feat_dir/vocielm_feat_of_0.1_960h_librispeech_400k_update/voicelm_${layer}layer_label_dir
    for x in $(seq 0 $((n_cluster - 1 )));do
     echo "$x 1"
    done>>$lab_dir/dict.km.txt
  done
fi
