#!/usr/bin/env bash


stage=1

stop_stage=1000
.  utils/parse_options.sh
. path_for_fairseq_speechtext.sh

export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
export OC_CAUSE=1
RVAD_ROOT=/workspace2/maduo/rVADfast
fairseq_dir=/workspace2/maduo/fairseq_speechtext
#if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
#   echo "prepare tsv file for librispeech"
#   raw_wav_dir=/workspace/xianghu/datasets/LibriSpeech ## it is download from https://www.openslr.org/12
#   des_dir=/workspace2/maduo/dataset/format/librispeech
#   for name in dev-clean dev-other test-clean test-other train-clean-100  train-clean-360  train-other-500;do
#     python3  source_md/wav2vec-u2/wav2vec_manifest_md.py\
#           $raw_wav_dir\
#           --dest_file $des_dir/$name.tsv\
#           --ext flac
#   done

#fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
   echo "get vad file"
   ## note: The process take some time.
   input_dir=/workspace2/maduo/dataset/format/librispeech
   des_dir=/workspace2/maduo/dataset/format/librispeech/librispeech_no_silence/
   #datasets="dev-clean dev-other test-clean test-other train-clean-100  train-clean-360  train-other-500"
   #datasets="dev-clean"
   datasets="dev-other test-clean test-other train-clean-100  train-clean-360  train-other-500"
   for name in $datasets;do
     python $fairseq_dir/examples/wav2vec/unsupervised/scripts/vads.py\
             -r $RVAD_ROOT < $input_dir/$name.tsv > $des_dir/${name}.vads
   done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
   echo "apply vad and remove silence"
   input_dir=/workspace2/maduo/dataset/format/librispeech
   des_dir=dataset/format/librispeech/librispeech_no_silence
   #datasets="dev-clean dev-other test-clean test-other train-clean-100  train-clean-360  train-other-500"
   #datasets="dev-clean"
   datasets="dev-other test-clean test-other train-clean-100  train-clean-360  train-other-500"
   for name in $datasets;do
     python $fairseq_dir/examples/wav2vec/unsupervised/scripts/remove_silence.py \
          --tsv $input_dir/${name}.tsv --vads $des_dir/${name}.vads --out $des_dir/$name
     echo "finish $name !!!!!!!!!!!!!!!!!!!!"
   done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
   echo "prepare tsv file for no silence librispeech"
   raw_wav_dir=/workspace2/maduo/dataset/format/librispeech/librispeech_no_silence
   des_dir=/workspace2/maduo/dataset/format/librispeech/librispeech_no_silence
   #datasets="dev-clean"
   datasets="dev-other test-clean test-other train-clean-100  train-clean-360  train-other-500"
   for name in $datasets;do
     python3  source_md/wav2vec-u2/wav2vec_manifest_md.py\
           $raw_wav_dir/$name\
           --dest_file $des_dir/$name.tsv\
           --ext flac
     echo "finish $name !!!!!!!!!!!!!!!!!!!!"
   done

fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
   echo "merger train part into one"
   des_dir=/workspace2/maduo/dataset/format/librispeech/librispeech_no_silence
   python source_md/wav2vec-u2/three_tsv2one.py \
          $des_dir/train-clean-100.tsv\
          $des_dir/train-clean-360.tsv\
          $des_dir/train-other-500.tsv\
          $des_dir/train.tsv

   echo "finish !!!!!!!!!!!!!!!!!!!!"
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then
   echo "using remove silence audio pass into wav2vec_larger model and output specify layer represent as audio feature of wav2vec-u2"
   export PYTHONPATH=/workspace2/maduo/fairseq_speechtext:$PYTHONPATH

   des_dir=/workspace2/maduo/dataset/format/librispeech/librispeech_no_silence
   feat_dir=/workspace2/maduo/dataset/format/librispeech/wav2vec_large_feat_dir_no_silence/ ## it stores wav2vec2 large model 15 layer representation of raw librispeech speech,
                                                        ## it  removes  silence.
   model=/workspace2/maduo/model_hub/librispeech/wav2vec2_Large_LV-60_no_finetune_offical_offer/wav2vec_vox_new.pt
   mkdir -p $feat_dir
   layer=14 #0-based index
   datasets="dev-other test-clean test-other train"
   #datasets="dev-clean"
   for name in  $datasets;do
     python  source_md/wav2vec-u2/wav2vec_extract_features.py\
           $des_dir\
           --split $name\
           --save-dir $feat_dir\
           --checkpoint $model\
           --layer $layer
     echo "finish $name !!!!!!!!!!!!!!!!!!!!"
   done
fi
### prepared no silence data .wrd .ltr files
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ];then
   tsv_dir=/workspace2/maduo/dataset/format/librispeech/librispeech_no_silence
   withsiltsv_dir=/workspace2/maduo/dataset/format/librispeech/
   datasets="dev-other test-clean test-other train"
   #datasets="dev-other test-clean test-other"
   #datasets="dev-clean"
   #datasets="train"  ## for self-train using kaldi
   for name in $datasets;do
     python3 source_md/wav2vec-u2/text/libri_labels_for_no_silence.py\
         $tsv_dir/${name}.tsv\
         $withsiltsv_dir/${name}.tsv\
         --output-dir $tsv_dir\
         --output-name $name
    echo "finish $name !!!!"
   done
fi


### prepared no silence data .phn files
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ];then
   datasets="dev-other test-clean test-other train"
   #datasets="dev-other test-clean test-other"
   #datasets="dev-clean"
   #datasets="train"
   tsv_dir=dataset/format/librispeech/librispeech_no_silence
   for name in $datasets;do
     ## it outputs *.phn file
    cat $tsv_dir/${name}.wrd |  python source_md/wav2vec-u2/text/g2p_wrd_to_phn.py >$tsv_dir/${name}.phn
   done
fi
### so I will online get mfcc feature , then train kmeans model.
### it is same as hubert iter1 setting ,
#### however 1. the mfcc feature is from no silence audio of librispeech now.
####         2. k-means cluster is setting  64.
### this mfcc pseudo_label is used as wav2vec-u2 auxiliary ssl loss
if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ];then
  echo "step12: train k-means model using 10% no silence train-960.tsv mfcc feature"
  tsv_dir=/workspace2/maduo/dataset/format/librispeech/librispeech_no_silence
  #feat_dir=$tsv_dir/mfcc_no_silence
  km_path=$tsv_dir/mfcc_no_silence/no_silence_librispeech-960h_0.1_portion_mfcc_km_100_clusters.mdl
  n_cluster=64
  nj=35
  sample_rate=16000
  portion=0.1
  for name in  train;do
    python source_md/wav2vec-u2/sklearn_kmeans_on_mfcc.py \
      --tsvfile $tsv_dir/${name}.tsv\
      --n-clusters ${n_cluster}\
      --nj $nj \
      --sample-rate ${sample_rate}\
      --portion ${portion}\
      --km-path ${km_path}
  done
fi

if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ];then
  echo "step13: get k-means pesudo target from k-means model"
  tsv_dir=/workspace2/maduo/dataset/format/librispeech/librispeech_no_silence
  km_path=$tsv_dir/mfcc_no_silence/no_silence_librispeech-960h_0.1_portion_mfcc_km_100_clusters.mdl
  nj=35
  sample_rate=16000
  lab_dir=$tsv_dir/mfcc_no_silence/mfcc_lab
  mkdir  -p $lab_dir
  for name in dev-clean dev-other train;do
    python source_md/wav2vec-u2/dump_pseudo_label_on_mfcc.py\
      --tsvfile $tsv_dir/${name}.tsv \
      --nj $nj \
      --sample-rate ${sample_rate}\
      --label-path ${lab_dir}/$name.km\
      --km-path $km_path

  done
fi

if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ];then
   echo "create a dummpy dictionary similar to hubert dictionary"
   dest_dir=/workspace2/maduo/dataset/format/librispeech/librispeech_no_silence
   lab_dir=$dest_dir/mfcc_no_silence/mfcc_lab
   n_cluster=64
   for x in $(seq 0 $((n_cluster - 1 )));do
     echo "$x 1"
   done>>$lab_dir/dict.km.txt
fi

