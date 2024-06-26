#!/usr/bin/env bash


stage=0
stop_stage=1000
. utils/parse_options.sh
#. path_for_fairseq_speechtext.sh
. path_for_fsq_speechtext.sh
#. path_for_fsq_sptt.sh
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
export OC_CAUSE=1 # for full backtrace

if [ ${stage} -le 35 ] && [ ${stop_stage} -ge 35 ];then
   echo "infer alimeeting eval dataset via ts_vad model trained on alimeeting train_ali_far dataset with musan and rirs noise"
   echo "get overlap and nonoverlap DER"
   regions="all overlap nonoverlap"
   #regions="all"
   collar="0.0 0.25"
   subset="Eval Test"
   for name in $regions;do
     for c in $collar;do
      for sub in $subset;do
        exp_name=baseline_camppluse_zh_en_common_advanced_with_musan_rirs
        rs_len=4
        segment_shift=1
        gen_subset=$sub
        speech_encoder_type="cam++"
        speaker_embedding_name_dir="cam++_en_zh_feature_dir"
        speaker_embed_dim=192
        root_path=/home/maduo
        data_path=/data/alimeeting/

        exp_dir=${root_path}/exp/speaker_diarization/ts_vad
        ts_vad_path=${root_path}/codebase/fairseq_speechtext/examples/speaker_diarization
        spk_path=${root_path}/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding
        results_path=${exp_dir}/${exp_name}/inf_detail/${gen_subset}_${name}_$c

        rttm_dir=$root_path/model_hub/ts_vad
        #sctk_tool_path=$ts_vad_path/SCTK-2.4.12
        s=$sub
        rttm_name=alimeeting_${s,,} ## offer groud truth label, ${s,,} means that Eval->eval or Test->test
        python3 ${ts_vad_path}/ts_vad/generate_detail.py ${data_path} \
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
