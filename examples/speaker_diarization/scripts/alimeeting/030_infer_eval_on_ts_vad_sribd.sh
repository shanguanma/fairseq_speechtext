#!/usr/bin/env bash


stage=0
stop_stage=1000
. utils/parse_options.sh
. path_for_fairseq_speechtext.sh # sribd
#. path_for_fsq_speechtext.sh # sribd, python=3.11
#. path_for_fsq_sptt.sh      # hltsz
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
export OC_CAUSE=1 # for full backtrace

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
   echo "infer alimeeting eval dataset via ts_vad model trained on alimeeting train_ali_far dataset with musan and rirs noise"
   echo "get overlap and nonoverlap DER"
   regions="all overlap nonoverlap"
   #regions="all"
   collar="0.0 0.25"
   subset="Eval Test"
   for name in $regions;do
     #echo "in the $name regions"
     for c in $collar;do
      #echo "in the $c mode,"
      for sub in $subset;do
        echo "in the $c mode, compute $name regions for the $sub dataset"
        exp_name=baseline_camppluse_zh_en_common_advanced_with_musan_rirs
        rs_len=4
        segment_shift=1
        gen_subset=$sub
        speech_encoder_type="cam++"
        speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
        speaker_embed_dim=192
        root_path=/mntcephfs/lab_data/maduo
        data_path=/mntcephfs/lab_data/maduo/datasets/alimeeting/

        exp_dir=${root_path}/exp/speaker_diarization/ts_vad
        ts_vad_path=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
        spk_path=${root_path}/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding
        results_path=${exp_dir}/${exp_name}/inf_detail/${gen_subset}_${name}_$c

        rttm_dir=$root_path/model_hub/ts_vad
        #sctk_tool_path=$ts_vad_path/SCTK-2.4.12
        s=$sub
        rttm_name=alimeeting_${s,,} ## offer groud truth label, ${s,,} means that Eval->eval or Test->test
        python3 ${ts_vad_path}/ts_vad/generate_with_spyder.py ${data_path} \
          --user-dir ${ts_vad_path}/ts_vad \
          --results-path ${results_path} \
          --path ${exp_dir}/${exp_name}/checkpoints/checkpoint_best.pt \
          --task ts_vad_task \
          --spk-path ${spk_path} \
          --rs-len ${rs_len} \
          --segment-shift ${segment_shift} \
          --gen-subset ${gen_subset} \
          --batch-size 64 \
          --sample-rate 16000 \
          --inference \
          --speech-encoder-type ${speech_encoder_type}\
	      --rttm_dir ${rttm_dir}\
          --rttm_name $rttm_name\
          --speaker_embedding_name_dir $speaker_embedding_name_dir\
          --speaker_embed_dim $speaker_embed_dim\
          --regions ${name}\
          --collar $c
    done
  done
 done
# Eval set collar=0.0 ,DER%,
# thresthold=0.5
# all    overlap   nonoverlap
# 12.74  21.08      7.65

# Eval set collar=0.25 ,DER%,
# thresthold=0.5
# all    overlap   nonoverlap
# 4.80   12.74      3.03

# Test set collar=0.0 ,DER%,
# thresthold=0.5
# all    overlap   nonoverlap
# 13.22   22.37     7.93

# Test set collar=0.25 ,DER%,
# thresthold=0.5
# all    overlap   nonoverlap
# 5.48   15.98      2.42

fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
   echo "infer alimeeting eval dataset via ts_vad model trained on alimeeting train_ali_far dataset with musan and rirs noise"
   echo "get overlap and nonoverlap DER"
   regions="all overlap nonoverlap"
   #regions="all"
   #collar="0.0 0.25"
   collar="0.25"
   subset="Eval Test"
   for name in $regions;do
     #echo "in the $name regions"
     for c in $collar;do
      #echo "in the $c mode,"
      for sub in $subset;do
        echo "in the $c mode, compute $name regions for the $sub dataset"
        exp_name=baseline_wavlm_large_finetune_with_musan_rirs
        rs_len=4
        segment_shift=1
        gen_subset=$sub
        speech_encoder_type="wavlm_large_ft"
        speaker_embedding_name_dir="wavlm_large_finetune_feature_dir"
        speaker_embed_dim=256
        root_path=/mntcephfs/lab_data/maduo
        data_path=/mntcephfs/lab_data/maduo/datasets/alimeeting/

        exp_dir=${root_path}/exp/speaker_diarization/ts_vad
        ts_vad_path=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
        spk_path=${root_path}/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding
        results_path=${exp_dir}/${exp_name}/inf_detail/${gen_subset}_${name}_$c

        rttm_dir=$root_path/model_hub/ts_vad
        #sctk_tool_path=$ts_vad_path/SCTK-2.4.12
        s=$sub
        rttm_name=alimeeting_${s,,} ## offer groud truth label, ${s,,} means that Eval->eval or Test->test
        python3 ${ts_vad_path}/ts_vad/generate_with_spyder.py ${data_path} \
          --user-dir ${ts_vad_path}/ts_vad \
          --results-path ${results_path} \
          --path ${exp_dir}/${exp_name}/checkpoints/checkpoint_best.pt \
          --task ts_vad_task \
          --spk-path ${spk_path} \
          --rs-len ${rs_len} \
          --segment-shift ${segment_shift} \
          --gen-subset ${gen_subset} \
          --batch-size 64 \
          --sample-rate 16000 \
          --inference \
          --speech-encoder-type ${speech_encoder_type}\
          --rttm_dir ${rttm_dir}\
          --rttm_name $rttm_name\
          --speaker_embedding_name_dir $speaker_embedding_name_dir\
          --speaker_embed_dim $speaker_embed_dim\
          --regions ${name}\
          --collar $c
    done
  done
 done
 fi

 if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
   echo "infer alimeeting eval dataset via ts_vad model trained on alimeeting train_ali_far dataset with musan and rirs noise"
   echo "get overlap and nonoverlap DER"
   regions="all overlap nonoverlap"
   #regions="all"
   #collar="0.0 0.25"
   collar="0.25"
   subset="Eval Test"
   for name in $regions;do
     #echo "in the $name regions"
     for c in $collar;do
      #echo "in the $c mode,"
      for sub in $subset;do
        echo "in the $c mode, compute $name regions for the $sub dataset"
        exp_name=baseline_wavlm_large_finetune_update_extract_true_with_musan_rirs
        rs_len=4
        segment_shift=1
        gen_subset=$sub
        speech_encoder_type="wavlm_large_ft"
        speaker_embedding_name_dir="wavlm_large_finetune_feature_dir"
        speaker_embed_dim=256
        root_path=/mntcephfs/lab_data/maduo
        data_path=/mntcephfs/lab_data/maduo/datasets/alimeeting/

        exp_dir=${root_path}/exp/speaker_diarization/ts_vad
        ts_vad_path=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
        spk_path=${root_path}/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding
        results_path=${exp_dir}/${exp_name}/inf_detail/${gen_subset}_${name}_$c

        rttm_dir=$root_path/model_hub/ts_vad
        #sctk_tool_path=$ts_vad_path/SCTK-2.4.12
        s=$sub
        rttm_name=alimeeting_${s,,} ## offer groud truth label, ${s,,} means that Eval->eval or Test->test
        python3 ${ts_vad_path}/ts_vad/generate_with_spyder.py ${data_path} \
          --user-dir ${ts_vad_path}/ts_vad \
          --results-path ${results_path} \
          --path ${exp_dir}/${exp_name}/checkpoints/checkpoint_best.pt \
          --task ts_vad_task \
          --spk-path ${spk_path} \
          --rs-len ${rs_len} \
          --segment-shift ${segment_shift} \
          --gen-subset ${gen_subset} \
          --batch-size 64 \
          --sample-rate 16000 \
          --inference \
          --speech-encoder-type ${speech_encoder_type}\
          --rttm_dir ${rttm_dir}\
          --rttm_name $rttm_name\
          --speaker_embedding_name_dir $speaker_embedding_name_dir\
          --speaker_embed_dim $speaker_embed_dim\
          --regions ${name}\
          --collar $c
    done
  done
 done
 fi

 if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
   echo "infer alimeeting eval dataset via ts_vad model trained on alimeeting train_ali_far dataset with musan and rirs noise"

        exp_name=baseline_camppluse_zh_en_common_advanced_with_musan_rirs
        rs_len=4
        segment_shift=1
        gen_subset=Eval
        speech_encoder_type="cam++"
        speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
        speaker_embed_dim=192
        #root_path=/home/maduo
        #data_path=/data/alimeeting/
        root_path=/mntcephfs/lab_data/maduo
        data_path=/mntcephfs/lab_data/maduo/datasets/alimeeting/

        exp_dir=${root_path}/exp/speaker_diarization/ts_vad
        ts_vad_path=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
        spk_path=${root_path}/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding
        results_path=${exp_dir}/${exp_name}/inf_with_md_eval/${gen_subset}

        rttm_dir=$root_path/model_hub/ts_vad
        sctk_tool_path=$ts_vad_path/SCTK-2.4.12
        rttm_name=alimeeting_eval ## offer groud truth label
        python3 ${ts_vad_path}/ts_vad/generate.py ${data_path} \
          --user-dir ${ts_vad_path}/ts_vad \
          --results-path ${results_path} \
          --path ${exp_dir}/${exp_name}/checkpoints/checkpoint_best.pt \
          --task ts_vad_task \
          --spk-path ${spk_path} \
          --rs-len ${rs_len} \
          --segment-shift ${segment_shift} \
          --gen-subset ${gen_subset} \
          --batch-size 64 \
          --sample-rate 16000 \
          --inference \
          --speech-encoder-type ${speech_encoder_type}\
          --rttm_dir ${rttm_dir}\
          --sctk_tool_path ${sctk_tool_path}\
          --rttm_name $rttm_name\
      --speaker_embedding_name_dir $speaker_embedding_name_dir
#Model DER:  0.13411843776013668
#Model ACC:  0.9565870768832015
#100%|██████████| 25/25 [00:25<00:00,  1.02s/it]
#Eval for threshold 0.20: DER 7.60%, MS 1.01%, FA 6.25%, SC 0.33%
#
#Eval for threshold 0.30: DER 5.68%, MS 1.53%, FA 3.77%, SC 0.39%
#
#Eval for threshold 0.35: DER 5.18%, MS 1.79%, FA 3.01%, SC 0.38%
#
#Eval for threshold 0.40: DER 4.89%, MS 2.09%, FA 2.43%, SC 0.37%
#
#Eval for threshold 0.45: DER 4.75%, MS 2.44%, FA 1.95%, SC 0.36%
#
#Eval for threshold 0.50: DER 4.75%, MS 2.88%, FA 1.52%, SC 0.35%
#
#Eval for threshold 0.55: DER 4.88%, MS 3.34%, FA 1.21%, SC 0.34%
#
#Eval for threshold 0.60: DER 5.13%, MS 3.85%, FA 0.96%, SC 0.31%
#
#Eval for threshold 0.70: DER 6.09%, MS 5.27%, FA 0.65%, SC 0.18%
#
#Eval for threshold 0.80: DER 7.89%, MS 7.31%, FA 0.45%, SC 0.12%

# 2024-8-2 note:
# conclude: cpmpared with training on RXT3090, traing on V100 32GB, its DER increase about 0.1, from 4.64(RTX3090,24GB) ->4.75(V100 32GB)
fi

 if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then
   echo "infer alimeeting eval dataset via ts_vad model trained on alimeeting train_ali_far dataset with musan and rirs noise"

        exp_name=baseline_camppluse_zh_en_common_advanced_with_musan_rirs_two_gpus
        rs_len=4
        segment_shift=1
        gen_subset=Eval
        speech_encoder_type="cam++"
        speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
        speaker_embed_dim=192
        #root_path=/home/maduo
        #data_path=/data/alimeeting/
        root_path=/mntcephfs/lab_data/maduo
        data_path=/mntcephfs/lab_data/maduo/datasets/alimeeting/

        exp_dir=${root_path}/exp/speaker_diarization/ts_vad
        ts_vad_path=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
        spk_path=${root_path}/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding
        results_path=${exp_dir}/${exp_name}/inf_with_md_eval/${gen_subset}

        rttm_dir=$root_path/model_hub/ts_vad
        sctk_tool_path=$ts_vad_path/SCTK-2.4.12
        rttm_name=alimeeting_eval ## offer groud truth label
        python3 ${ts_vad_path}/ts_vad/generate.py ${data_path} \
          --user-dir ${ts_vad_path}/ts_vad \
          --results-path ${results_path} \
          --path ${exp_dir}/${exp_name}/checkpoints/checkpoint_best.pt \
          --task ts_vad_task \
          --spk-path ${spk_path} \
          --rs-len ${rs_len} \
          --segment-shift ${segment_shift} \
          --gen-subset ${gen_subset} \
          --batch-size 64 \
          --sample-rate 16000 \
          --inference \
          --speech-encoder-type ${speech_encoder_type}\
          --rttm_dir ${rttm_dir}\
          --sctk_tool_path ${sctk_tool_path}\
          --rttm_name $rttm_name\
          --speaker_embedding_name_dir $speaker_embedding_name_dir
# Model DER:  0.13184296665756676
#Model ACC:  0.9570168162867878
#frame_len: 0.04
#100%|██████████| 25/25 [00:25<00:00,  1.02s/it]
#Eval for threshold 0.20: DER 6.70%, MS 1.08%, FA 5.25%, SC 0.37%
#
#Eval for threshold 0.30: DER 5.20%, MS 1.58%, FA 3.25%, SC 0.37%
#
#Eval for threshold 0.35: DER 4.85%, MS 1.87%, FA 2.63%, SC 0.35%
#
#Eval for threshold 0.40: DER 4.62%, MS 2.11%, FA 2.15%, SC 0.37%
#
#Eval for threshold 0.45: DER 4.47%, MS 2.34%, FA 1.76%, SC 0.37%
#
#Eval for threshold 0.50: DER 4.49%, MS 2.63%, FA 1.48%, SC 0.38%
#
#Eval for threshold 0.55: DER 4.63%, MS 3.00%, FA 1.24%, SC 0.38%
#
#Eval for threshold 0.60: DER 4.82%, MS 3.43%, FA 1.03%, SC 0.37%
#
#Eval for threshold 0.70: DER 5.69%, MS 4.65%, FA 0.73%, SC 0.31%
#
#Eval for threshold 0.80: DER 7.46%, MS 6.71%, FA 0.52%, SC 0.22%

    fi
