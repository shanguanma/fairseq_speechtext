#!/usr/bin/env bash

stage=0
stop_stage=100

. utils/parse_options.sh

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
   echo "prepare tsv file for librispeech train set"
   raw_data_dir=/workspace/xianghu/datasets/LibriSpeech/
   dir=/workspace2/maduo/dataset/format/librispeech
   for name in train-clean-100 train-clean-360 train-other-500;do
     python3 source_md/wav2vec-u2/wav2vec_manifest_md.py\
          $raw_data_dir/$name\
          --dest_file $dir/$name.tsv\
          --ext flac 
           
   done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
   echo "prepare tsv file for librispeech dev set and test set"
   raw_data_dir=/workspace/xianghu/datasets/LibriSpeech/
   dir=/workspace2/maduo/dataset/format/librispeech
   for name in dev-clean dev-other test-clean test-other;do
     python3 source_md/wav2vec-u2/wav2vec_manifest_md.py\
          $raw_data_dir/$name\
          --dest_file $dir/$name.tsv\
          --ext flac

   done
fi
