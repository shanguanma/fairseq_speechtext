#!/bin/bash


stage=0

stop_stage=1000
nj=32
. utils/parse_options.sh
. path_for_fairseq_speechtext.sh
RVAD_ROOT=/mntnfs/lee_data1/maduo/codebase/rVADfast
fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext

export TORCH_DISTRIBUTED_DEBUG=INFO
#if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
#   echo "get vad file"
#   input_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech
#   des_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence
#   datasets="train_m"
#   mkdir -p $des_dir
#   mkdir -p $des_dir/parallel
#   for name in $datasets;do
#     python $fairseq_dir/examples/wav2vec/unsupervised/scripts/vads_for_wavscp.py\
#             -r $RVAD_ROOT < $input_dir/m/$name/wav.scp > $des_dir/${name}.vads
  


#    done
#fi
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
   echo "get remove silence audio file with multi cpus using silero-vad method, it is fastest than rVADfast"
   input_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech
   des_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence
   datasets="train_m"
   #mkdir -p $des_dir/parallel
   for name in $datasets;do
   #export  OMP_NUM_THREADS=1
    ## debug:
    #  torchrun --nproc_per_node=5 --master_port=12345 codebase/fairseq_speechtext/examples/wav2vec/unsupervised/scripts/vads_for_wavscp_parallel.py -r codebase/rVADfast --output_dir tests --wav_scp tests/wav5.scp 
  # it is very slow, Even when using parallel
  #torchrun --nproc_per_node=$nj --master_port=12345 \
  #     $fairseq_dir/examples/wav2vec/unsupervised/scripts/vads_for_wavscp_parallel.py \
  #     -r $RVAD_ROOT --output_dir $des_dir/parallel --wav_scp $input_dir/m/$name/wav.scp
  
  #torchrun --nproc_per_node=$nj --master_port=12345 \
  #    $fairseq_dir/examples/wav2vec/unsupervised/scripts/silero-vad.py \
  #    --wavscp $input_dir/m/$name/wav.scp  --out $des_dir/$name/base_on_silero-vad_onnx_torchrun_parallel
   
  ## this script can process segments file in kaldi format
  #  (fsq_speechtext) [maduo@pbcmlg02 maduo]$  torchrun --nproc_per_node=5 --master_port=12345  codebase/fairseq_speechtext/examples/wav2vec/unsupervised/scripts/silero-vad_process_segement.py --wav_file tests/test_kaldi_format/wav.scp --onnx true --segments tests/test_kaldi_format/segments   --text_file tests/test_kaldi_format/text --out tests/wenetspeech_wo_silence_silero_vad_again
    torchrun --nproc_per_node=$nj --master_port=12345 \
        $fairseq_dir/examples/wav2vec/unsupervised/scripts/silero-vad_process_segement.py\
        --wav_file $input_dir/m/$name/wav.scp --onnx true \
        --segments  $input_dir/m/$name/segments \
        --text_file $input_dir/m/$name/text\
        --out $des_dir/$name/base_on_silero-vad_onnx_torchrun_parallel
  done
fi

#if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
#   echo "apply vad and remove silence"
#   input_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech
#   des_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence
#   datasets="train_m"
#   for name in $datasets;do
#     python $fairseq_dir/examples/wav2vec/unsupervised/scripts/remove_silence_for_wavscp.py \
#          --wavscp $input_dir/m/$name/wav.scp --vads $des_dir/${name}.vads --out $des_dir/$name
#     echo "finish $name !!!!!!!!!!!!!!!!!!!!"
#   done
#fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
   echo "prepare tsv file for no silence wenetspeech"
   raw_wav_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/
   des_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence
   #datasets="dev-clean"
   #datasets="dev-other test-clean test-other train-clean-100  train-clean-360  train-other-500"
   datasets="train_m"
   for name in $datasets;do
     python3  source-md/wav2vec-u2/wav2vec_manifest_md.py\
           $raw_wav_dir/$name\
           --dest_file $des_dir/$name/$name.tsv\
           --ext opus
     echo "finish $name !!!!!!!!!!!!!!!!!!!!"
   done

fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then
   echo "get remove silence audio file with multi cpus using silero-vad method, it is fastest than rVADfast"
   input_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech
   des_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence
   #datasets="dev  test_meeting  test_net"
   #datasets="dev test_net test_meeting"
   datasets="test_net"
   for name in $datasets;do
     torchrun --nproc_per_node=$nj --master_port=12347 \
        $fairseq_dir/examples/wav2vec/unsupervised/scripts/silero-vad_process_segement.py\
        --wav_file $input_dir/m/$name/wav.scp --onnx true \
        --segments  $input_dir/m/$name/segments \
        --text_file $input_dir/m/$name/text\
        --out $des_dir/$name/base_on_silero-vad_onnx_torchrun_parallel
    echo "finish ${datasets}!!!!"
   done
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ];then
   echo "prepare tsv file for no silence wenetspeech"
   raw_wav_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/
   des_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence
   #datasets="dev-clean"
   #datasets="dev-other test-clean test-other train-clean-100  train-clean-360  train-other-500"
   #datasets="dev  test_meeting  test_net"
   datasets="test_net"
   for name in $datasets;do
     python3  source-md/wav2vec-u2/wav2vec_manifest_md.py\
           $raw_wav_dir/$name\
           --dest_file $des_dir/$name/$name.tsv\
           --ext opus
     echo "finish $name !!!!!!!!!!!!!!!!!!!!"
   done

fi






### so I will online get mfcc feature , then train kmeans model.
if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ];then
  echo "step10: train k-means model using 10% train_m.tsv mfcc feature"
  tsv_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir/
  feat_dir=$tsv_dir/mfcc_no_silence
  km_path=$feat_dir/wenetspeech-medium-1000h_0.1_portion_mfcc_km_100_clusters.mdl
  n_cluster=100
  nj=35
  sample_rate=16000
  portion=0.1
  mkdir -p $feat_dir
  for name in  train_m;do
    python codebase/fairseq_speechtext/examples/wav2vec/unsupervised/wav2vec-u2/sklearn_kmeans_on_mfcc.py \
      --tsvfile $tsv_dir/${name}.tsv\
      --n-clusters ${n_cluster}\
      --nj $nj \
      --sample-rate ${sample_rate}\
      --portion ${portion}\
      --km-path ${km_path}
  done
fi





if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ];then
  echo "step11: get k-means pesudo target from k-means model"
  tsv_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir/
  km_path=$tsv_dir/mfcc_no_silence/wenetspeech-medium-1000h_0.1_portion_mfcc_km_100_clusters.mdl
  nj=35
  sample_rate=16000
  lab_dir=$tsv_dir/mfcc_no_silence/mfcc_lab
  mkdir  -p $lab_dir
  for name in train_m dev test_meeting test_net ;do
    python codebase/fairseq_speechtext/examples/wav2vec/unsupervised/wav2vec-u2/dump_pseudo_label_on_mfcc.py\
      --tsvfile $tsv_dir/${name}.tsv \
      --nj $nj \
      --sample-rate ${sample_rate}\
      --label-path ${lab_dir}/$name.km\
      --km-path $km_path

  done

  fi

if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ];then
   echo "create a dummpy dictionary"
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir/
   lab_dir=$tsv_dir/mfcc_no_silence/mfcc_lab
   n_cluster=100
   for x in $(seq 0 $((n_cluster - 1 )));do
     echo "$x 1"
   done>>$lab_dir/dict.km.txt
fi


## in order to compute  testset PER in GAN decoding, so I commput transcript phoneme sequence speech
if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ];then
   echo "prepared word2phn"
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence
   scp_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech/m
   dest_dir=$tsv_dir/tsv_dir
   lexicon_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq/wenetspeech_lang/  ## its raw name is wenetspeech_dict
   #datasets="test_net test_meeting"
   datasets="train_m"
   mkdir -p $dest_dir
   for name in $datasets;do
      cat $tsv_dir/$name/${name}.tsv | sort -k1 > $dest_dir/${name}.tsv
      python codebase/fairseq_speechtext/examples/wav2vec/unsupervised/Chinese/remove_miss_files.py\
         $dest_dir/${name}.tsv\
         $scp_dir/$name/text\
         $dest_dir/${name}.text

   done
   for name in $datasets;do
      # splits the Chinese words into character and keep the English words
      python codebase/fairseq_speechtext/examples/wav2vec/unsupervised/Chinese/character_tokenizer.py \
          $dest_dir/${name}.text $dest_dir/$name.text_split
      
      python codebase/fairseq_speechtext/examples/wav2vec/unsupervised/Chinese/character2phones.py\
           $lexicon_dir/uniq_lexicon.txt\
           $dest_dir/$name.text_split  \
           $dest_dir/$name.pre_phn

     ## remove uttid 
     awk '{for(i=2;i<=NF;i=i+1) printf " "$i;print ""}' $dest_dir/$name.text_split > $dest_dir/$name.wrd
     head $dest_dir/$name.wrd
     awk '{for(i=2;i<=NF;i=i+1) printf " "$i;print ""}' $dest_dir/$name.pre_phn >$dest_dir/$name.phn
     head $dest_dir/$name.phn
   done
fi






if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ];then
   echo "dump 14th layer representation of hubert large model"
   data_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir/
   dest_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/hubert_large_feat_dir_no_silence/
   checkpoint_dir=/mntcephfs/lab_data/maduo/model_hub/Chinese/
   datasets="dev"
   for name in $datasets;do
       python codebase/fairseq_speechtext/examples/wav2vec/unsupervised/Chinese/hubert_extract_features_for_tsv.py\
         $data_dir\
         --split $name\
         --save-dir $dest_dir\
         --checkpoint $checkpoint_dir/chinese-hubert-large-fairseq-ckpt.pt


   done
fi

if [ ${stage} -le 16 ] && [ ${stop_stage} -ge 16 ];then
   echo "dump 14th layer representation of hubert large model"
   data_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir/
   dest_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/hubert_large_feat_dir_no_silence/
   checkpoint_dir=/mntcephfs/lab_data/maduo/model_hub/Chinese/
   datasets="test_net test_meeting"
   for name in $datasets;do
       python codebase/fairseq_speechtext/examples/wav2vec/unsupervised/Chinese/hubert_extract_features_for_tsv.py\
         $data_dir\
         --split $name\
         --save-dir $dest_dir\
         --checkpoint $checkpoint_dir/chinese-hubert-large-fairseq-ckpt.pt


   done
fi

if [ ${stage} -le 17 ] && [ ${stop_stage} -ge 17 ];then
   echo "dump 14th layer representation of hubert large model"
   data_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir/
   dest_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/hubert_large_feat_dir_no_silence/
   checkpoint_dir=/mntcephfs/lab_data/maduo/model_hub/Chinese/
   datasets="train_m"
   for name in $datasets;do
       python codebase/fairseq_speechtext/examples/wav2vec/unsupervised/Chinese/hubert_extract_features_for_tsv.py\
         $data_dir\
         --split $name\
         --save-dir $dest_dir\
         --checkpoint $checkpoint_dir/chinese-hubert-large-fairseq-ckpt.pt


   done
fi


if [ ${stage} -le 20 ] && [ ${stop_stage} -ge 20 ];then
   echo "format with silence audio, in order to unify code api"
   data_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech/m/
   datasets="train_m"
   nj1=14
   for name in $datasets;do
   torchrun --nproc_per_node=$nj1 --master_port=12349 \
       codebase/fairseq_speechtext/examples/wav2vec/unsupervised/Chinese/format_wav_scp.py \
       --wav_file $data_dir/$name/wav.scp \
       --out $data_dir/no_segements/$name\
       --segments $data_dir/$name/segments\
       --text_file $data_dir/$name/text
   done
fi

if [ ${stage} -le 21 ] && [ ${stop_stage} -ge 21 ];then
   echo "prepare tsv file for with silence wenetspeech"
   raw_wav_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech/m/no_segements
   des_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech/m/no_segements_tsv_dir
   datasets="test_net"
   for name in $datasets;do
     python3  source-md/wav2vec-u2/wav2vec_manifest_md.py\
           $raw_wav_dir/$name\
           --dest_file $des_dir/$name.tsv\
           --ext opus
     echo "finish $name !!!!!!!!!!!!!!!!!!!!"
   done
fi



