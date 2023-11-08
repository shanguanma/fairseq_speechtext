#!/usr/bin/env bash

stage=0

stop_stage=100
. utils/parse_options.sh
. path_for_fsq_speechtext.sh
if [ ${stage} -le 0 ]&& [ ${stop_stage} -ge 0 ];then
   echo "dump voicelm2 feature of audio into local for training k-means model"
   root_data="dataset/format/librispeech/40M_librispeech_lm_monophncode_librispeech_frame_monophncode_using_wav2vec-u2_model"
   text_dict="dataset/format/librispeech/40M_librispeech_lm_monophncode_librispeech_frame_monophncode_using_wav2vec-u2_model/dict.textphncode.txt"
   label_paths="dataset/format/librispeech/40M_librispeech_lm_monophncode_librispeech_frame_monophncode_using_wav2vec-u2_model/train-960.speechphncode"  ## audio label file path
   manifest_path="dataset/format/librispeech/40M_librispeech_lm_monophncode_librispeech_frame_monophncode_using_wav2vec-u2_model/train-960.tsv"  ## audio file path
   #label_paths=tests/train-960_10.speechphncode
   #manifest_path=tests/train-960_10.tsv   

   manifest_text_path="dataset/format/librispeech/40M_librispeech_lm_monophncode_librispeech_frame_monophncode_using_wav2vec-u2_model/train-960.textphncode"  ## text file path
   #layer=7
   layer=12
   
   sample_rate=16000

   ckpt_path="exp/pretrain/pretrain_on_base_voicelm2_4gpu_8update_960h_400k_update_flash_attention_lr4e_4_40M_unpaired_text_1to40/checkpoint_best.pt"
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   feat_dir=$tsv_dir/feat_dir/voicelm2_feat_of_0.1_960h_librispeech
   #feat_dir=test/feat_dir
   CUDA_VISIBLE_DEVICES=2\
     python source_md/wav2vec-u2/dump_voicelm2_feature_for_kmeans.py\
                --text_dict $text_dict\
                --label_paths $label_paths\
                --audio_tsv $manifest_path\
                --text_path $manifest_text_path\
                --layer $layer\
                --sample_rate $sample_rate\
                --ckpt_path $ckpt_path\
                --feat_dir $feat_dir

fi

if [ ${stage} -le 1 ]&& [ ${stop_stage} -ge 1 ];then
   echo "training k-means model"
  
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   feat_dir=$tsv_dir/feat_dir/voicelm2_feat_of_0.1_960h_librispeech
   
    #for layer in 7 8 9 10;do
    for c in 100 500;do
    #for c in 500;do
     #km_path=$feat_dir/train-960_10_percent_voicelm2_7layer_km_${c}_clusters.mdl
     km_path=$feat_dir/train-960_10_percent_voicelm2_12layer_km_${c}_clusters.mdl
     feats=$feat_dir/train-960_10_percent_voicelm2_12layer_raw_feature.npy

     python source_md/wav2vec-u2/learn_kmeans.py \
        --n-cluster $c\
        --km-path $km_path\
        --feats $feats
    done  
fi


if [ ${stage} -le 2 ]&& [ ${stop_stage} -ge 2 ];then
   echo "dump voicelm2 feature of audio into local for training k-means model"
    root_data="dataset/format/librispeech/40M_librispeech_lm_monophncode_librispeech_frame_monophncode_using_wav2vec-u2_model"
    text_dict=$root_data/dict.textphncode.txt
    label_paths=$root_data
    manifest_path=$root_data
    manifest_text_path=$root_data
    #text_dict="dataset/format/librispeech/40M_librispeech_lm_monophncode_librispeech_frame_monophncode_using_wav2vec-u2_model/dict.textphncode.txt"
    #label_paths="dataset/format/librispeech/40M_librispeech_lm_monophncode_librispeech_frame_monophncode_using_wav2vec-u2_model/train-960.speechphncode"  ## audio label file path
    #manifest_path="dataset/format/librispeech/40M_librispeech_lm_monophncode_librispeech_frame_monophncode_using_wav2vec-u2_model/train-960.tsv"  ## audio file path
    #manifest_text_path="dataset/format/librispeech/40M_librispeech_lm_monophncode_librispeech_frame_monophncode_using_wav2vec-u2_model/train-960.textphncode"  ## text file path
    #layer=7
    layer=12
    sample_rate=16000
    ckpt_path="exp/pretrain/pretrain_on_base_voicelm2_4gpu_8update_960h_400k_update_flash_attention_lr4e_4_40M_unpaired_text_1to40/checkpoint_best.pt"   
    #km_model_path=train-960_10_percent_voicelm2_7layer_km_${c}_clusters.mdl
    #label_dir=dataset/format/librispeech/feat_dir/voicelm2_feat_of_0.1_960h_librispeech/voicelm2_7layer_${c}_clusters_label_dir
    #mkdir -p $label_dir
    #for name in dev-other dev-clean train-960;do
    #for name in dev-other;do
    for name in dev-clean train-960;do
       for c in 100 500;do
         label_dir=dataset/format/librispeech/feat_dir/voicelm2_feat_of_0.1_960h_librispeech/voicelm2_7layer_${c}_clusters_label_dir
         km_model_path=dataset/format/librispeech/feat_dir/voicelm2_feat_of_0.1_960h_librispeech/train-960_10_percent_voicelm2_7layer_km_${c}_clusters.mdl
         mkdir -p $label_dir
        CUDA_VISIBLE_DEVICES=2 \
          python  source_md/wav2vec-u2/dump_voicelm2_pseudo_label.py\
                --text_dict $text_dict\
                --label_paths $label_paths/$name.speechphncode\
                --audio_tsv $manifest_path/$name.tsv\
                --text_path $manifest_text_path/$name.textphncode\
                --layer $layer\
                --sample_rate $sample_rate\
                --ckpt_path $ckpt_path\
                --label_path $label_dir/$name.km\
                --km_model_path $km_model_path
      done
   done
                               
                                     

fi
 
## for computer_phmi
if [ ${stage} -le 3 ]&& [ ${stop_stage} -ge 3 ];then
   echo "dump voicelm2 feature of audio into local for training k-means model"
    root_data="dataset/format/librispeech/40M_librispeech_lm_monophncode_librispeech_frame_monophncode_using_wav2vec-u2_model"
    text_dict=$root_data/dict.textphncode.txt
    label_paths=$root_data
    manifest_path=$root_data
    manifest_text_path=$root_data
    #text_dict="dataset/format/librispeech/40M_librispeech_lm_monophncode_librispeech_frame_monophncode_using_wav2vec-u2_model/dict.textphncode.txt"
    #label_paths="dataset/format/librispeech/40M_librispeech_lm_monophncode_librispeech_frame_monophncode_using_wav2vec-u2_model/train-960.speechphncode"  ## audio label file path
    #manifest_path="dataset/format/librispeech/40M_librispeech_lm_monophncode_librispeech_frame_monophncode_using_wav2vec-u2_model/train-960.tsv"  ## audio file path
    #manifest_text_path="dataset/format/librispeech/40M_librispeech_lm_monophncode_librispeech_frame_monophncode_using_wav2vec-u2_model/train-960.textphncode"  ## text file path
    #layer=7
    layer=12
    sample_rate=16000
    ckpt_path="exp/pretrain/pretrain_on_base_voicelm2_4gpu_8update_960h_400k_update_flash_attention_lr4e_4_40M_unpaired_text_1to40/checkpoint_best.pt"
    #km_model_path=train-960_10_percent_voicelm2_7layer_km_${c}_clusters.mdl
    #label_dir=dataset/format/librispeech/feat_dir/voicelm2_feat_of_0.1_960h_librispeech/voicelm2_7layer_${c}_clusters_label_dir
    #mkdir -p $label_dir
    #for name in dev-other dev-clean train-960;do
    for name in dev-other;do
    #for name in dev-clean train-960;do
       for c in 100 500;do
         label_dir=dataset/format/librispeech/feat_dir/voicelm2_feat_of_0.1_960h_librispeech/voicelm2_7layer_${c}_clusters_label_dir_for_phmi
         km_model_path=dataset/format/librispeech/feat_dir/voicelm2_feat_of_0.1_960h_librispeech/train-960_10_percent_voicelm2_7layer_km_${c}_clusters.mdl
         mkdir -p $label_dir
        CUDA_VISIBLE_DEVICES=0 \
          python  source_md/wav2vec-u2/dump_voicelm2_pseudo_label_for_phmi.py\
                --text_dict $text_dict\
                --label_paths $label_paths/$name.speechphncode\
                --audio_tsv $manifest_path/$name.tsv\
                --text_path $manifest_text_path/$name.textphncode\
                --layer $layer\
                --sample_rate $sample_rate\
                --ckpt_path $ckpt_path\
                --label_path $label_dir/$name.km\
                --km_model_path $km_model_path
      done
   done



fi
 
