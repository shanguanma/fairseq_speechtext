#!/usr/bin/env bash


stage=0
stop_stage=1000
. utils/parse_options.sh
. path_for_fsq_sptt.sh
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
        config_name=base_libricss_sim
        exp_name=baseline_libricss_sim
        other_args=
        root_path=/home/maduo
        data_path=/data/SimLibriCSS/ ## input format data for speaker diarization


        exp_dir=${root_path}/exp/speaker_diarization/ts_vad
        ts_vad_path=${root_path}/codebase/fairseq_speechtext/examples/speaker_diarization/ts_vad
        speech_encoder_path=${root_path}/model_hub/speaker_pretrain_model/en/opensource/pretrain.model #this model is from https://github.com/TaoRuijie/ECAPA-TDNN/blob/main/exps/pretrain.model
        spk_path=${root_path}/model_hub/ts_vad/spk_embed/libricss_sim/SpeakerEmbedding
        #rir_path=/data/rirs_noises/RIRS_NOISES
        mkdir -p ${exp_dir}/${exp_name}

        fairseq-hydra-train \
          --config-dir ${ts_vad_path}/config \
          --config-name ${config_name} \
          hydra.run.dir=${exp_dir}/${exp_name} \
          hydra.job.name=${exp_name} \
          hydra.sweep.dir=${exp_dir}/${exp_name} \
          task.data=${data_path} \
          common.user_dir=${ts_vad_path} \
          +task.spk_path=${spk_path} \
          +model.speech_encoder_path=${speech_encoder_path} \
	  checkpoint.save_interval_updates=10000 \
          ${other_args}
fi
