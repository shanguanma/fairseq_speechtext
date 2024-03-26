#!/usr/bin/env bash

stage=0
stop_stage=1000
nj=32
. utils/parse_options.sh
#. path_for_fairseq_speechtext.sh
#. path_for_fsq_speechtext.sh
. path_for_fsq_sptt.sh
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then

  for split in train dev test;do
    #split=train
    libricss_path=/data/SimLibriCSS/SimLibriCSS-${split}
    python source_md/extract_target_speech_simulate_libricss.py \
       --rttm_path ${libricss_path}/rttm \
       --orig_audio_path ${libricss_path}/wav \
       --target_audio_path ${libricss_path}/target_audio 
  done
fi



if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then

 # this stage need to one gpu.
 speaker_pretrain_model=/home/maduo/model_hub/speaker_pretrain_model/en/opensource/pretrain.model
 echo "Get target embeddings"
 for name in train dev test;do
   spk_path=/home/maduo/model_hub/ts_vad/spk_embed/libricss_sim/SpeakerEmbedding/$name
   mkdir -p $spk_path
   target_audio_path=/data/SimLibriCSS/SimLibriCSS-${name}/target_audio
   python3 source_md/extract_target_embedding_simulate_libricss.py \
    --target_audio_path $target_audio_path \
    --target_embedding_path $spk_path \
    --source $speaker_pretrain_model
 done
fi
