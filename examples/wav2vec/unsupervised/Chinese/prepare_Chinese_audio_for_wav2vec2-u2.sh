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
   #datasets="train_m"
   datasets="dev test_net test_meeting"
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


  echo  "copy *.phn into dest_dir"
  datasets="dev test_net test_meeting"
  tsv_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence
  in_dir=$tsv_dir/tsv_dir
  dest_dir=$tsv_dir/hubert_large_feat_dir_no_silence
  for name in $datasets;do
      cp -r $in_dir/$name.phn  $dest_dir/$name.phn
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
   #datasets="train_m"
   datasets="test_net test_meeting dev"
   nj1=40
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
   #datasets="test_net"
   datasets="train_m test_net test_meeting dev"
   mkdir -p $des_dir
   for name in $datasets;do
     python3  source-md/wav2vec-u2/wav2vec_manifest_md.py\
           $raw_wav_dir/$name\
           --dest_file $des_dir/$name.tsv\
           --ext opus
     echo "finish $name !!!!!!!!!!!!!!!!!!!!"
   done
fi

if [ ${stage} -le 22 ] && [ ${stop_stage} -ge 22 ];then
  echo "rename train_m into train"

fi

## in order to compute  testset PER in GAN decoding, so I commput transcript phoneme sequence speech without tone
if [ ${stage} -le 30 ] && [ ${stop_stage} -ge 30 ];then
   echo "prepared word2phn"
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence
   scp_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech/m
   dest_dir=$tsv_dir/tsv_dir
   lexicon_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone/wenetspeech_dict/  ## its raw name is wenetspeech_dict
   datasets="dev test_net test_meeting"
   #datasets="train_m"
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
           $lexicon_dir/uniq_lexicon_remove_tone.txt\
           $dest_dir/$name.text_split  \
           $dest_dir/$name.pre_phn

     ## remove uttid
     awk '{for(i=2;i<=NF;i=i+1) printf " "$i;print ""}' $dest_dir/$name.text_split > $dest_dir/$name.wrd
     head $dest_dir/$name.wrd
     awk '{for(i=2;i<=NF;i=i+1) printf " "$i;print ""}' $dest_dir/$name.pre_phn >$dest_dir/$name.phn
     head $dest_dir/$name.phn
   done
fi

if [ ${stage} -le 31 ] && [ ${stop_stage} -ge 31 ];then
  echo  "copy *.phn into dest_dir"
  datasets="dev test_net test_meeting"
  tsv_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence
  in_dir=$tsv_dir/tsv_dir
  dest_dir=$tsv_dir/hubert_large_feat_dir_no_silence
  for name in $datasets;do
      cp -r $in_dir/$name.phn  $dest_dir/$name.phn
  done
fi



## 2024-2-19 we will use wav2vec-large model instead of hubert-large model as frontend feature extractor for wav2vec-u2 model(GAN model)
if [ ${stage} -le 40 ] && [ ${stop_stage} -ge 40 ];then
    echo "dump 14th layer representation of hubert large model" # the title is error.
   data_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir/
   dest_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/wav2vec2_large_feat_dir_no_silence/
   checkpoint_dir=/mntcephfs/lab_data/maduo/model_hub/Chinese/
   datasets="train_m dev"
   mkdir -p $dest_dir
   for name in $datasets;do
       python codebase/fairseq_speechtext/examples/wav2vec/unsupervised/Chinese/wav2vec_extract_features.py\
         $data_dir\
         --split $name\
         --save-dir $dest_dir\
         --checkpoint $checkpoint_dir/chinese-wav2vec2-large-fairseq-ckpt.pt


   done
fi





# 2024-2-26 we will use hubert-large model to extract feature and cluster 500 class to as one target for voicelm(T-Hubert) model
### so I will offline get iter1 hubert specify layer feature , then train kmeans model.
if [ ${stage} -le 50 ] && [ ${stop_stage} -ge 50 ];then
  echo "step50: get using 10% train-960.tsv hubert feature"
  root_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech/m/
  tsv_dir=$root_dir/no_segements_tsv_dir/
  feat_dir=$tsv_dir/feat_dir/hubert_14layer_feat_of_0.1_1000h_wenetspeech_m_from_tencent_game_pretrain_hubert_large
  ckpt_path=/mntcephfs/lab_data/maduo/model_hub/Chinese/chinese-hubert-large-fairseq-ckpt.pt
  portion=0.1
  sample_rate=16000
  layer=14
  mkdir -p $feat_dir
  for name in train_m;do
    python  codebase/fairseq_speechtext/examples/wav2vec/unsupervised/wav2vec-u2/dump_hubert_feature.py \
      --tsvfile $tsv_dir/${name}.tsv\
      --ckpt-path $ckpt_path\
      --feat-dir $feat_dir\
      --portion ${portion}\
      --sample-rate $sample_rate\
      --layer $layer
  done
fi

if [ ${stage} -le 51 ] && [ ${stop_stage} -ge 51 ];then
    echo "learn  kmeans model on hubert feature"
    tsv_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech/m/no_segements_tsv_dir/
    feat_dir=$tsv_dir/feat_dir/hubert_14layer_feat_of_0.1_1000h_wenetspeech_m_from_tencent_game_pretrain_hubert_large
    km_path=$feat_dir/wenetspeech_m-1000h_0.1_portion_hubert_large_14layer_km_500_clusters.mdl
    python codebase/fairseq_speechtext/examples/wav2vec/unsupervised/wav2vec-u2/learn_kmeans_on_hubert.py \
        --n-cluster 500\
        --km-path $km_path\
        --feat-dir $feat_dir

fi

if [ ${stage} -le 52 ] && [ ${stop_stage} -ge 52 ];then
  echo "step: get k-means pesudo target from k-means model"
  tsv_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech/m/no_segements_tsv_dir/
  ckpt_path=/mntcephfs/lab_data/maduo/model_hub/Chinese/chinese-hubert-large-fairseq-ckpt.pt
  feat_dir=$tsv_dir/feat_dir/hubert_14layer_feat_of_0.1_1000h_wenetspeech_m_from_tencent_game_pretrain_hubert_large
  km_path=$feat_dir/wenetspeech_m-1000h_0.1_portion_hubert_large_14layer_km_500_clusters.mdl
  label_dir=$feat_dir/label_dir/

  sample_rate=16000
  layer=14
  mkdir -p $label_dir
  #for name in train-960 dev-clean dev-other ;do
  #for name in train_m dev test_meeting test_net;do
  for name in dev test_meeting test_net train_m;do
    python codebase/fairseq_speechtext/examples/wav2vec/unsupervised/wav2vec-u2/dump_pseudo_label_on_hubert.py\
      --tsvfile $tsv_dir/${name}.tsv \
      --sample-rate ${sample_rate}\
      --label-path ${label_dir}/$name.km\
      --km-path $km_path \
      --ckpt-path $ckpt_path\
      --layer $layer


  done
fi


if [ ${stage} -le 53 ] && [ ${stop_stage} -ge 53 ];then
   echo "create a dummpy dictionary"
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech/m/no_segements_tsv_dir/
   lab_dir=$tsv_dir/feat_dir/hubert_14layer_feat_of_0.1_1000h_wenetspeech_m_from_tencent_game_pretrain_hubert_large/label_dir
   n_cluster=500
   for x in $(seq 0 $((n_cluster - 1 )));do
     echo "$x 1"
   done>>$lab_dir/dict.km.txt
fi

## 2024-2-27 we get Chinese hubert-large  model 15layer feature from wenetspeech raw speech (with silence), 
# then pass it into generator of GAN(wav2vec-u2 Chinese version) to get phone-like label as the other target of voicelm(T-Hubert).

if [ ${stage} -le 54 ] && [ ${stop_stage} -ge 54 ];then
   echo "dump 14th layer representation of hubert large model"
   data_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech/m/no_segements_tsv_dir/
   dest_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech/m/hubert_large_feat_dir_with_silence/
   checkpoint_dir=/mntcephfs/lab_data/maduo/model_hub/Chinese/
   datasets="test_meeting test_net"
   mkdir -p $dest_dir
   for name in $datasets;do
       python codebase/fairseq_speechtext/examples/wav2vec/unsupervised/Chinese/hubert_extract_features_for_tsv.py\
         $data_dir\
         --split $name\
         --save-dir $dest_dir\
         --checkpoint $checkpoint_dir/chinese-hubert-large-fairseq-ckpt.pt


   done
fi

if [ ${stage} -le 55 ] && [ ${stop_stage} -ge 55 ];then
   echo "dump 14th layer representation of hubert large model"
   data_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech/m/no_segements_tsv_dir/
   dest_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech/m/hubert_large_feat_dir_with_silence/
   checkpoint_dir=/mntcephfs/lab_data/maduo/model_hub/Chinese/
   datasets="dev train_m"
   mkdir -p $dest_dir
   for name in $datasets;do
       python codebase/fairseq_speechtext/examples/wav2vec/unsupervised/Chinese/hubert_extract_features_for_tsv.py\
         $data_dir\
         --split $name\
         --save-dir $dest_dir\
         --checkpoint $checkpoint_dir/chinese-hubert-large-fairseq-ckpt.pt


   done
fi


## in order to compute  testset PER in GAN decoding raw wav(without remove silence), so I commput transcript phoneme sequence speech without tone
if [ ${stage} -le 56 ] && [ ${stop_stage} -ge 56 ];then
   echo "prepared word2phn"
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech/m/no_segements_tsv_dir
   scp_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech/m/
   #dest_dir=$tsv_dir/tsv_dir
   lexicon_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone/wenetspeech_dict/  ## its raw name is wenetspeech_dict
   datasets="dev test_net test_meeting"
   #datasets="train_m"
   #datasets="train_m"
   #mkdir -p $dest_dir
   #for name in $datasets;do
   #   cat $tsv_dir/$name/${name}.tsv | sort -k1 > $dest_dir/${name}.tsv
   #   python codebase/fairseq_speechtext/examples/wav2vec/unsupervised/Chinese/remove_miss_files.py\
   #      $dest_dir/${name}.tsv\
   #      $scp_dir/$name/text\
   #      $dest_dir/${name}.text

   #done
   for name in $datasets;do
      # splits the Chinese words into character and keep the English words
      cat $tsv_dir/${name}.tsv | sort -k1 > $tsv_dir/${name}.tsv1
      cat $scp_dir/${name}/text >  $tsv_dir/${name}.text
      head  $tsv_dir/${name}.tsv1
      head $tsv_dir/${name}.text
      python codebase/fairseq_speechtext/examples/wav2vec/unsupervised/Chinese/character_tokenizer.py \
          $tsv_dir/${name}.text $tsv_dir/$name.text_split

      python codebase/fairseq_speechtext/examples/wav2vec/unsupervised/Chinese/character2phones.py\
           $lexicon_dir/uniq_lexicon_remove_tone.txt\
           $tsv_dir/$name.text_split  \
           $tsv_dir/$name.pre_phn

     ## remove uttid
     awk '{for(i=2;i<=NF;i=i+1) printf " "$i;print ""}' $tsv_dir/$name.text_split > $tsv_dir/$name.wrd
     head $tsv_dir/$name.wrd
     awk '{for(i=2;i<=NF;i=i+1) printf " "$i;print ""}' $tsv_dir/$name.pre_phn >$tsv_dir/$name.phn
     head $tsv_dir/$name.phn
   done
fi

## 2024-2-28 I use high frequecey phone lexicon to get devset and testset transcript
## in order to compute  testset PER in GAN decoding, so I commput transcript phoneme sequence speech without tone
if [ ${stage} -le 57 ] && [ ${stop_stage} -ge 57 ];then
   echo "prepared word2phn"
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence
   scp_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech/m
   dest_dir=$tsv_dir/tsv_dir_remove_lower_freq_phone
   lexicon_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone/wenetspeech_dict/  ## its raw name is wenetspeech_dict
   datasets="dev test_net test_meeting"
   #datasets="train_m"
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

      #python codebase/fairseq_speechtext/examples/wav2vec/unsupervised/Chinese/character2phones.py\
       python codebase/fairseq_speechtext/examples/wav2vec/unsupervised/Chinese/character2phones_new.py\
           $lexicon_dir/uniq_lexicon_remove_tone_filtered.lst\
           $dest_dir/$name.text_split  \
           $dest_dir/$name.pre_phn\
           $dest_dir/$name.remove_utt
     
     # remove utt from tsv base on ${name}.remove_utt
     python /mntnfs/lee_data1/maduo/codebase/fairseq_speechtext/examples/wav2vec/unsupervised/Chinese/remove_utt_from_tsv.py \
              $dest_dir/${name}.remove_utt \
              $dest_dir/${name}.tsv\
              $dest_dir/${name}.text_split\
              $dest_dir/${name}.tsv_new\
              $dest_dir/${name}.text_split_new
     mv $dest_dir/${name}.tsv_new $dest_dir/${name}.tsv
     mv $dest_dir/${name}.text_split_new $dest_dir/${name}.text_split
     ## remove uttid
     awk '{for(i=2;i<=NF;i=i+1) printf " "$i;print ""}' $dest_dir/$name.text_split > $dest_dir/$name.wrd
     head $dest_dir/$name.wrd
     awk '{for(i=2;i<=NF;i=i+1) printf " "$i;print ""}' $dest_dir/$name.pre_phn >$dest_dir/$name.phn
     head $dest_dir/$name.phn
   done
fi


# Due to the removal of low-frequency phonemes, the dictionary is reduced. Therefore, when transcribing into phoneme sequences, some words are directly removed from the transcribed text (mostly English words) when they are not in the dictionary. Therefore, the audio files must also be removed accordingly, because the development The set and test set need to calculate PER
if [ ${stage} -le 59 ] && [ ${stop_stage} -ge 59 ];then
   echo "dump 14th layer representation of hubert large model"
   data_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir_remove_lower_freq_phone
   dest_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/hubert_large_feat_dir_no_silence/
   checkpoint_dir=/mntcephfs/lab_data/maduo/model_hub/Chinese/
   datasets="dev test_net test_meeting"
   for name in $datasets;do
       # the script will generate feature file and copy *.tsv *.wrd *.phn into $dest_dir
       python codebase/fairseq_speechtext/examples/wav2vec/unsupervised/Chinese/hubert_extract_features_for_tsv.py\
         $data_dir\
         --split $name\
         --save-dir $dest_dir\
         --checkpoint $checkpoint_dir/chinese-hubert-large-fairseq-ckpt.pt


   done
fi

# Due to the removal of low-frequency phonemes, the dictionary is reduced. Therefore, when transcribing into phoneme sequences, some words are directly removed from the transcribed text (mostly English words) when they are not in the dictionary. Therefore, the audio files must also be removed accordingly, because the development The set and test set need to calculate PER
if [ ${stage} -le 60 ] && [ ${stop_stage} -ge 60 ];then
  echo "step11: get k-means pesudo target from k-means model"
  root_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir
  km_path=$root_dir/mfcc_no_silence/wenetspeech-medium-1000h_0.1_portion_mfcc_km_100_clusters.mdl
  nj=35
  sample_rate=16000
  lab_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir_remove_lower_freq_phone/
  tsv_dir=$lab_dir
  mkdir  -p $lab_dir
  for name in dev test_meeting test_net ;do
    python codebase/fairseq_speechtext/examples/wav2vec/unsupervised/wav2vec-u2/dump_pseudo_label_on_mfcc.py\
      --tsvfile $tsv_dir/${name}.tsv \
      --nj $nj \
      --sample-rate ${sample_rate}\
      --label-path ${lab_dir}/$name.km\
      --km-path $km_path

  done

  fi

# (TODO) modify
# 2024-2-28 I remove lower frequecey phone from lexicon
## in order to compute  testset PER in GAN decoding raw wav(without remove silence), so I commput transcript phoneme sequence speech without tone
if [ ${stage} -le 100 ] && [ ${stop_stage} -ge 100 ];then
   echo "prepared word2phn"
   input_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech/m/no_segements_tsv_dir
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech/m/no_segements_tsv_dir_remove_lower_freq_phone
   scp_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech/m/
   #dest_dir=$tsv_dir/tsv_dir
   lexicon_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone/wenetspeech_dict/  ## its raw name is wenetspeech_dict
   datasets="dev test_net test_meeting"
   #datasets="train_m"
   #datasets="train_m"
   #mkdir -p $dest_dir
   #for name in $datasets;do
   #   cat $tsv_dir/$name/${name}.tsv | sort -k1 > $dest_dir/${name}.tsv
   #   python codebase/fairseq_speechtext/examples/wav2vec/unsupervised/Chinese/remove_miss_files.py\
   #      $dest_dir/${name}.tsv\
   #      $scp_dir/$name/text\
   #      $dest_dir/${name}.text

   #done
   mkdir -p $tsv_dir
   for name in $datasets;do
       cp -r $input_dir/${name}.tsv $tsv_dir/
      # splits the Chinese words into character and keep the English words
      cat $tsv_dir/${name}.tsv | sort -k1 > $tsv_dir/${name}.tsv1
      cat $scp_dir/${name}/text >  $tsv_dir/${name}.text
      head  $tsv_dir/${name}.tsv1
      head $tsv_dir/${name}.text
      python codebase/fairseq_speechtext/examples/wav2vec/unsupervised/Chinese/character_tokenizer.py \
          $tsv_dir/${name}.text $tsv_dir/$name.text_split

      python codebase/fairseq_speechtext/examples/wav2vec/unsupervised/Chinese/character2phones.py\
           $lexicon_dir/uniq_lexicon_remove_tone_filtered.lst\
           $tsv_dir/$name.text_split  \
           $tsv_dir/$name.pre_phn

     ## remove uttid
     awk '{for(i=2;i<=NF;i=i+1) printf " "$i;print ""}' $tsv_dir/$name.text_split > $tsv_dir/$name.wrd
     head $tsv_dir/$name.wrd
     awk '{for(i=2;i<=NF;i=i+1) printf " "$i;print ""}' $tsv_dir/$name.pre_phn >$tsv_dir/$name.phn
     head $tsv_dir/$name.phn
   done
fi
