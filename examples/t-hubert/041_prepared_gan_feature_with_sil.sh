#!/usr/bin/env bash


stage=0

stop_stage=1000

. utils/parse_options.sh
#. path_for_fairseq_speechtext.sh
#. path_for_fsq_speechtext.sh
. path_for_fsq_sptt.sh

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
   echo "prepare tsv file for librispeech"
   raw_wav_dir=/workspace/xianghu/datasets/LibriSpeech ## it is download from https://www.openslr.org/12
   des_dir=/workspace2/maduo/dataset/format/librispeech
   for name in dev-clean dev-other test-clean test-other train-clean-100  train-clean-360  train-other-500;do
     python3  source_md/wav2vec-u2/wav2vec_manifest_md.py\
           $raw_wav_dir\
           --dest_file $des_dir/$name.tsv\
           --ext flac
   done

fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
   export PYTHONPATH=/workspace2/maduo/fairseq_speechtext:$PYTHONPATH
   echo "get wav2vec_large feature for librispeech"
   des_dir=/workspace2/maduo/dataset/format/librispeech
   feat_dir=$des_dir/wav2vec_large_feat_dir ## it stores wav2vec2 large model 15 layer representation of raw librispeech speech,
                                            ## it doesn't remove  silence.
   model=/workspace2/maduo/model_hub/librispeech/wav2vec2_Large_LV-60_no_finetune_offical_offer/wav2vec_vox_new.pt
   mkdir -p $feat_dir
   layer=14 #0-based index
   for name in train-clean-100  train-clean-360  train-other-500;do
     python  source_md/wav2vec-u2/wav2vec_extract_features.py\
           $des_dir\
           --split $name\
	   --save-dir $feat_dir\
	   --checkpoint $model\
           --layer $layer
     echo "finish $name !!!!!!!!!!!!!!!!!!!!"
   done
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
   export PYTHONPATH=/workspace2/maduo/fairseq_speechtext:$PYTHONPATH
   echo "get wav2vec_large feature for librispeech"
   des_dir=/workspace2/maduo/dataset/format/librispeech
   feat_dir=$des_dir/wav2vec_large_feat_dir
   model=/workspace2/maduo/model_hub/librispeech/wav2vec2_Large_LV-60_no_finetune_offical_offer/wav2vec_vox_new.pt
   mkdir -p $feat_dir
   layer=14 #0-based index
   for name in dev-clean dev-other test-clean test-other;do
     python  source_md/wav2vec-u2/wav2vec_extract_features.py\
           $des_dir\
           --split $name\
           --save-dir $feat_dir\
           --checkpoint $model\
           --layer $layer
     echo "finish $name !!!!!!!!!!!!!!!!!!!!"
   done
fi



if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
   echo "merger train part into one"
   des_dir=/workspace2/maduo/dataset/format/librispeech/librispeech
   python source_md/wav2vec-u2/three_tsv2one.py \
          $des_dir/train-clean-100.tsv\
          $des_dir/train-clean-360.tsv\
          $des_dir/train-other-500.tsv\
          $des_dir/train-960.tsv
fi
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
   export PYTHONPATH=codebase/fairseq_speechtext:$PYTHONPATH
   echo "get wav2vec_large feature for librispeech"
   #des_dir=/workspace2/maduo/dataset/format/librispeech
   des_dir=dataset/format/librispeech/tsv_dir/
   feat_dir=dataset/format/librispeech/wav2vec_large_feat_dir
   #model=/workspace2/maduo/model_hub/librispeech/wav2vec2_Large_LV-60_no_finetune_offical_offer/wav2vec_vox_new.pt
   model=/home/maduo/model_hub/librispeech/wav2vec2_Large_LV-60_no_finetune_offical_offer/wav2vec_vox_new.pt
   mkdir -p $feat_dir
   layer=14 #0-based index
   for name in train-960 dev-other dev-clean;do
     python  source_md/wav2vec-u2/wav2vec_extract_features.py\
           $des_dir\
           --split $name\
           --save-dir $feat_dir\
           --checkpoint $model\
           --layer $layer
     echo "finish $name !!!!!!!!!!!!!!!!!!!!"
   done
fi





## the above audio don't apply vad ,so it doesn't remove silence

