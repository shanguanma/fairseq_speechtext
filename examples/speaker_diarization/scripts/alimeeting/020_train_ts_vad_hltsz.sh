#!/usr/bin/env bash


stage=0
stop_stage=1000
. utils/parse_options.sh
#. path_for_fairseq_speechtext.sh
#. path_for_fsq_speechtext.sh
. path_for_fsq_sptt.sh
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
   echo "train ts_vad using ecap-tdnn speaker embedding"
	config_name=base
	exp_name=baseline
	other_args=
	root_path=/home/maduo
	data_path=/data/alimeeting/


	exp_dir=${root_path}/exp/speaker_diarization/ts_vad
	ts_vad_path=${root_path}/codebase/fairseq_speechtext/examples/speaker_diarization/ts_vad
	speech_encoder_path=${root_path}/model_hub/ts_vad/ecapa-tdnn.model # Speaker encoder path
	spk_path=${root_path}/model_hub/ts_vad/spk_embed/SpeakerEmbedding
        rir_path=/data/rirs_noises/RIRS_NOISES
	mkdir -p ${exp_dir}/${exp_name}

	fairseq-hydra-train \
	  --config-dir ${ts_vad_path}/config \
	  --config-name ${config_name} \
	  hydra.run.dir=${exp_dir}/${exp_name} \
	  hydra.job.name=${exp_name} \
	  hydra.sweep.dir=${exp_dir}/${exp_name} \
	  task.data=${data_path} \
	  task.rir_path=${rir_path}\
	  common.user_dir=${ts_vad_path} \
	  +task.spk_path=${spk_path} \
	  +model.speech_encoder_path=${speech_encoder_path} \
	  ${other_args}

# one RTX3090: consume about 5 hours on training stage.
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
   echo "train ts_vad  with musan_rirs using ecap-tdnn speaker embedding"
        config_name=base
        exp_name=baseline_with_musan_rirs
        other_args=
        root_path=/home/maduo
        data_path=/data/alimeeting/


        exp_dir=${root_path}/exp/speaker_diarization/ts_vad
        ts_vad_path=${root_path}/codebase/fairseq_speechtext/examples/speaker_diarization/ts_vad
        speech_encoder_path=${root_path}/model_hub/ts_vad/ecapa-tdnn.model # Speaker encoder path
        spk_path=${root_path}/model_hub/ts_vad/spk_embed/SpeakerEmbedding
        rir_path=/data/rirs_noises/RIRS_NOISES
	musan_path=/data/musan
        mkdir -p ${exp_dir}/${exp_name}

        fairseq-hydra-train \
          --config-dir ${ts_vad_path}/config \
          --config-name ${config_name} \
          hydra.run.dir=${exp_dir}/${exp_name} \
          hydra.job.name=${exp_name} \
          hydra.sweep.dir=${exp_dir}/${exp_name} \
          task.data=${data_path} \
          task.rir_path=${rir_path}\
	  +task.musan_path=${musan_path}\
          common.user_dir=${ts_vad_path} \
          +task.spk_path=${spk_path} \
          +model.speech_encoder_path=${speech_encoder_path} \
          ${other_args}

# one RTX3090: consume about 5 hours on training stage.
fi

