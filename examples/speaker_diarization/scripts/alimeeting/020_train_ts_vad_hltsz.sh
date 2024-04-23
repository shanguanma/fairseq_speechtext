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
	speech_encoder_path=${root_path}/model_hub/ts_vad/ecapa-tdnn.model # Speaker encoder path, this model is finetune on alimeeting dataset
	spk_path=${root_path}/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding
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
        spk_path=${root_path}/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding
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



if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
   echo "train ts_vad  with musan_rirs using speech_campplus_sv_zh_en_16k-common_advanced of 3D-speaker speaker embedding"
        config_name=base
        exp_name=baseline_camppluse_zh_en_common_advanced_with_musan_rirs
        other_args=
        root_path=/home/maduo
        data_path=/data/alimeeting/


        exp_dir=${root_path}/exp/speaker_diarization/ts_vad
        ts_vad_path=${root_path}/codebase/fairseq_speechtext/examples/speaker_diarization/ts_vad
        #speech_encoder_path=${root_path}/model_hub/ts_vad/ecapa-tdnn.model # Speaker encoder path
	speech_encoder_path=${root_path}/.cache/modelscope/hub/iic/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt # Speaker encoder path
	                   ## this is download from ts_vad/generate_speaker_embedding_from_modelscope.py 
			   ## you can download this model manual follow commands
			   ##  git lfs install
                           ##  git clone https://www.modelscope.cn/iic/speech_campplus_sv_zh_en_16k-common_advanced.git

        #spk_path=${root_path}/model_hub/ts_vad/spk_embed/SpeakerEmbedding
        spk_path=${root_path}/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding
	speech_encoder_type="cam++"
	speaker_embedding_name_dir="cam++_en_zh_feature_dir"
        rir_path=/data/rirs_noises/RIRS_NOISES
        musan_path=/data/musan
        mkdir -p ${exp_dir}/${exp_name}
        world_size=1
        update_freq=1

        fairseq-hydra-train \
          --config-dir ${ts_vad_path}/config \
          --config-name ${config_name} \
          hydra.run.dir=${exp_dir}/${exp_name} \
          hydra.job.name=${exp_name} \
          hydra.sweep.dir=${exp_dir}/${exp_name} \
          task.data=${data_path} \
          task.rir_path=${rir_path}\
          +task.musan_path=${musan_path}\
	  +task.speaker_embedding_name_dir=$speaker_embedding_name_dir\
	  distributed_training.distributed_world_size=${world_size}\
          distributed_training.distributed_port=-1\
          distributed_training.ddp_backend=legacy_ddp\
          optimization.update_freq=[${update_freq}]\
	  task.speech_encoder_type=$speech_encoder_type\
          common.user_dir=${ts_vad_path} \
          +task.spk_path=${spk_path} \
          +model.speech_encoder_path=${speech_encoder_path} \
          ${other_args}

# one RTX3090: consume about 5 hours on training stage.
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
   echo "train ts_vad  with rirs using speech_campplus_sv_zh_en_16k-common_advanced of 3D-speaker speaker embedding"
        config_name=base
        exp_name=baseline_camppluse_zh_en_common_advanced_with_rirs
        other_args=
        root_path=/home/maduo
        data_path=/data/alimeeting/


        exp_dir=${root_path}/exp/speaker_diarization/ts_vad
        ts_vad_path=${root_path}/codebase/fairseq_speechtext/examples/speaker_diarization/ts_vad
        #speech_encoder_path=${root_path}/model_hub/ts_vad/ecapa-tdnn.model # Speaker encoder path
        speech_encoder_path=${root_path}/.cache/modelscope/hub/iic/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt # Speaker encoder path
                           ## this is download from ts_vad/generate_speaker_embedding_from_modelscope.py
                           ## you can download this model manual follow commands
                           ##  git lfs install
                           ##  git clone https://www.modelscope.cn/iic/speech_campplus_sv_zh_en_16k-common_advanced.git

        #spk_path=${root_path}/model_hub/ts_vad/spk_embed/SpeakerEmbedding
        spk_path=${root_path}/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding
        speech_encoder_type="cam++"
        speaker_embedding_name_dir="cam++_en_zh_feature_dir"
        rir_path=/data/rirs_noises/RIRS_NOISES
        #musan_path=/data/musan
        mkdir -p ${exp_dir}/${exp_name}
        world_size=1
        update_freq=1

        fairseq-hydra-train \
          --config-dir ${ts_vad_path}/config \
          --config-name ${config_name} \
          hydra.run.dir=${exp_dir}/${exp_name} \
          hydra.job.name=${exp_name} \
          hydra.sweep.dir=${exp_dir}/${exp_name} \
          task.data=${data_path} \
          task.rir_path=${rir_path}\
          +task.speaker_embedding_name_dir=$speaker_embedding_name_dir\
          distributed_training.distributed_world_size=${world_size}\
          distributed_training.distributed_port=-1\
          distributed_training.ddp_backend=legacy_ddp\
          optimization.update_freq=[${update_freq}]\
          task.speech_encoder_type=$speech_encoder_type\
	  common.user_dir=${ts_vad_path} \
          +task.spk_path=${spk_path} \
          +model.speech_encoder_path=${speech_encoder_path} \
          ${other_args}

# one RTX3090: consume about 5 hours on training stage.
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
   echo "train ts_vad  with musan_rirs using ecapa_tdnn_based_DINO_ft_on_CNCeleb1-2_devset-LM of wespeaker DINO ft speaker embedding"
        config_name=base
        exp_name=baseline_ecapa_tdnn_based_DINO_ft_on_CNCeleb1-2_devset-LM_with_musan_rirs
        other_args=
        root_path=/home/maduo
        data_path=/data/alimeeting/


        exp_dir=${root_path}/exp/speaker_diarization/ts_vad
        ts_vad_path=${root_path}/codebase/fairseq_speechtext/examples/speaker_diarization/ts_vad
        #speech_encoder_path=${root_path}/model_hub/ts_vad/ecapa-tdnn.model # Speaker encoder path
        #speech_encoder_path=${root_path}/.cache/modelscope/hub/iic/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt # Speaker encoder path
	speech_encoder_path=/home/maduo/model_hub/speaker_pretrain_model/zh/wespeaker/ecapa_tdnn_based_DINO_ft_on_CNCeleb1-2_devset-LM/avg_model.pt
                           ## this is download from ts_vad/generate_speaker_embedding_from_modelscope.py
                           ## you can download this model manual follow commands
                           ##  git lfs install
                           ##  git clone https://www.modelscope.cn/iic/speech_campplus_sv_zh_en_16k-common_advanced.git

        #spk_path=${root_path}/model_hub/ts_vad/spk_embed/SpeakerEmbedding
        spk_path=${root_path}/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding
        speech_encoder_type="ecapa_wespeaker"
        speaker_embedding_name_dir="ecapa_tdnn_based_DINO_ft_on_CNCeleb1-2_devset-LM_feature_dir"
        rir_path=/data/rirs_noises/RIRS_NOISES
        musan_path=/data/musan
        mkdir -p ${exp_dir}/${exp_name}
        world_size=1
        update_freq=1

        fairseq-hydra-train \
          --config-dir ${ts_vad_path}/config \
          --config-name ${config_name} \
          hydra.run.dir=${exp_dir}/${exp_name} \
          hydra.job.name=${exp_name} \
          hydra.sweep.dir=${exp_dir}/${exp_name} \
          task.data=${data_path} \
          task.rir_path=${rir_path}\
          +task.musan_path=${musan_path}\
          +task.speaker_embedding_name_dir=$speaker_embedding_name_dir\
          distributed_training.distributed_world_size=${world_size}\
          distributed_training.distributed_port=-1\
          distributed_training.ddp_backend=legacy_ddp\
          optimization.update_freq=[${update_freq}]\
          task.speech_encoder_type=$speech_encoder_type\
          common.user_dir=${ts_vad_path} \
	  +task.spk_path=${spk_path} \
          +model.speech_encoder_path=${speech_encoder_path} \
          ${other_args}

# one RTX3090: consume about 5 hours on training stage.
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then
   echo "train ts_vad  with musan_rirs using resnet34-lm  of wespeaker speaker embedding"
        config_name=base
        exp_name=baseline_ResNet34-TSTP-emb256-fbank80-num_frms200-aug0.6-spTrue-saFalse-ArcMargin-SGD-epoch150_exp2_on_CNCeleb1-2-LM_with_musan_rirs
        other_args=
        root_path=/home/maduo
        data_path=/data/alimeeting/


        exp_dir=${root_path}/exp/speaker_diarization/ts_vad
        ts_vad_path=${root_path}/codebase/fairseq_speechtext/examples/speaker_diarization/ts_vad
        #speech_encoder_path=${root_path}/model_hub/ts_vad/ecapa-tdnn.model # Speaker encoder path
        speech_encoder_path=${root_path}/model_hub/speaker_pretrain_model/zh/wespeaker/ResNet34-TSTP-emb256-fbank80-num_frms200-aug0.6-spTrue-saFalse-ArcMargin-SGD-epoch150_exp2_on_CNCeleb1-2-LM/final_model.pt # Speaker encoder path
                           ## this is download from ts_vad/generate_speaker_embedding_from_modelscope.py
                           ## you can download this model manual follow commands
                           ##  git lfs install
                           ##  git clone https://www.modelscope.cn/iic/speech_campplus_sv_zh_en_16k-common_advanced.git

        #spk_path=${root_path}/model_hub/ts_vad/spk_embed/SpeakerEmbedding
        spk_path=${root_path}/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding
        speech_encoder_type="resnet34_wespeaker"
        speaker_embedding_name_dir="ResNet34-TSTP-emb256-fbank80-num_frms200-aug0.6-spTrue-saFalse-ArcMargin-SGD-epoch150_exp2_on_CNCeleb1-2-LM_feature_dir"
        rir_path=/data/rirs_noises/RIRS_NOISES
        musan_path=/data/musan
        mkdir -p ${exp_dir}/${exp_name}
        world_size=1
        update_freq=1

        fairseq-hydra-train \
          --config-dir ${ts_vad_path}/config \
          --config-name ${config_name} \
          hydra.run.dir=${exp_dir}/${exp_name} \
          hydra.job.name=${exp_name} \
          hydra.sweep.dir=${exp_dir}/${exp_name} \
          task.data=${data_path} \
          task.rir_path=${rir_path}\
          +task.musan_path=${musan_path}\
          +task.speaker_embedding_name_dir=$speaker_embedding_name_dir\
          distributed_training.distributed_world_size=${world_size}\
          distributed_training.distributed_port=-1\
          distributed_training.ddp_backend=legacy_ddp\
          optimization.update_freq=[${update_freq}]\
          task.speech_encoder_type=$speech_encoder_type\
	  common.user_dir=${ts_vad_path} \
          +task.spk_path=${spk_path} \
          +model.speech_encoder_path=${speech_encoder_path} \
	  +model.speaker_embed_dim=256\
          ${other_args}

# one RTX3090: consume about 5 hours on training stage.
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ];then
   echo "train ts_vad  with musan_rirs using resnet34-lm  of wespeaker speaker embedding"
        config_name=base
        exp_name=baseline_ResNet34-TSTP-emb256-fbank80-num_frms200-aug0.6-spTrue-saFalse-ArcMargin-SGD-epoch150_exp2_on_CNCeleb1-2-LM_with_musan_rirs_exp2
        other_args=
        root_path=/home/maduo
        data_path=/data/alimeeting/


        exp_dir=${root_path}/exp/speaker_diarization/ts_vad
        ts_vad_path=${root_path}/codebase/fairseq_speechtext/examples/speaker_diarization/ts_vad
        #speech_encoder_path=${root_path}/model_hub/ts_vad/ecapa-tdnn.model # Speaker encoder path
        speech_encoder_path=${root_path}/model_hub/speaker_pretrain_model/zh/wespeaker/ResNet34-TSTP-emb256-fbank80-num_frms200-aug0.6-spTrue-saFalse-ArcMargin-SGD-epoch150_exp2_on_CNCeleb1-2-LM/final_model.pt # Speaker encoder path
                           ## this is download from ts_vad/generate_speaker_embedding_from_modelscope.py
                           ## you can download this model manual follow commands
                           ##  git lfs install
                           ##  git clone https://www.modelscope.cn/iic/speech_campplus_sv_zh_en_16k-common_advanced.git

        #spk_path=${root_path}/model_hub/ts_vad/spk_embed/SpeakerEmbedding
        spk_path=${root_path}/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding
        speech_encoder_type="resnet34_wespeaker"
        speaker_embedding_name_dir="ResNet34-TSTP-emb256-fbank80-num_frms200-aug0.6-spTrue-saFalse-ArcMargin-SGD-epoch150_exp2_on_CNCeleb1-2-LM_feature_dir"
        rir_path=/data/rirs_noises/RIRS_NOISES
        musan_path=/data/musan
        mkdir -p ${exp_dir}/${exp_name}
        world_size=1
        update_freq=1

        fairseq-hydra-train \
          --config-dir ${ts_vad_path}/config \
          --config-name ${config_name} \
          hydra.run.dir=${exp_dir}/${exp_name} \
          hydra.job.name=${exp_name} \
          hydra.sweep.dir=${exp_dir}/${exp_name} \
          task.data=${data_path} \
          task.rir_path=${rir_path}\
          +task.musan_path=${musan_path}\
          +task.speaker_embedding_name_dir=$speaker_embedding_name_dir\
          distributed_training.distributed_world_size=${world_size}\
          distributed_training.distributed_port=-1\
          distributed_training.ddp_backend=legacy_ddp\
          optimization.update_freq=[${update_freq}]\
          task.speech_encoder_type=$speech_encoder_type\
          common.user_dir=${ts_vad_path} \
          +task.spk_path=${spk_path} \
	  +model.speech_encoder_path=${speech_encoder_path} \
          +model.speaker_embed_dim=256\
          ${other_args}

# one RTX3090: consume about 5 hours on training stage.
fi



if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ];then
   echo "train ts_vad  with musan_rirs using ecapa_tdnn_wespeaker_on_CNCeleb1-2-LM  speaker embedding"
        config_name=base
        exp_name=baseline_ecapa_tdnn_wespeaker_on_CNCeleb1-2-LM_with_musan_rirs
        other_args=
        root_path=/home/maduo
        data_path=/data/alimeeting/


        exp_dir=${root_path}/exp/speaker_diarization/ts_vad
        ts_vad_path=${root_path}/codebase/fairseq_speechtext/examples/speaker_diarization/ts_vad
        #speech_encoder_path=${root_path}/model_hub/ts_vad/ecapa-tdnn.model # Speaker encoder path
        #speech_encoder_path=${root_path}/.cache/modelscope/hub/iic/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt # Speaker encoder path

	# (TODO) maduo copy model from sribd 
        speech_encoder_path=/home/maduo/model_hub/speaker_pretrain_model/zh/wespeaker/ecapa_tdnn_wespeaker_cnceleb1-2-LM/final_model.pt

        #spk_path=${root_path}/model_hub/ts_vad/spk_embed/SpeakerEmbedding
        spk_path=${root_path}/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding
        speech_encoder_type="ecapa_wespeaker"
        speaker_embedding_name_dir="ecapa_tdnn_wespeaker_cnceleb1-2-LM_feature_dir"
	speaker_embed_dim=192
        rir_path=/data/rirs_noises/RIRS_NOISES
        musan_path=/data/musan
        mkdir -p ${exp_dir}/${exp_name}
        world_size=1
        update_freq=1

        fairseq-hydra-train \
          --config-dir ${ts_vad_path}/config \
          --config-name ${config_name} \
          hydra.run.dir=${exp_dir}/${exp_name} \
          hydra.job.name=${exp_name} \
          hydra.sweep.dir=${exp_dir}/${exp_name} \
          task.data=${data_path} \
          task.rir_path=${rir_path}\
          +task.musan_path=${musan_path}\
          +task.speaker_embedding_name_dir=$speaker_embedding_name_dir\
          distributed_training.distributed_world_size=${world_size}\
          distributed_training.distributed_port=-1\
          distributed_training.ddp_backend=legacy_ddp\
          optimization.update_freq=[${update_freq}]\
          task.speech_encoder_type=$speech_encoder_type\
          common.user_dir=${ts_vad_path} \
          +task.spk_path=${spk_path} \
          +model.speech_encoder_path=${speech_encoder_path} \
	  +task.speaker_embed_dim=$speaker_embed_dim\
	  ${other_args}

# one RTX3090: consume about 5 hours on training stage.
fi


if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ];then
   echo "train ts_vad  with musan_rirs using ecapa_tdnn_wespeaker_on_CNCeleb1-2-LM  speaker embedding"
        config_name=base
        exp_name=baseline_ecapa_tdnn_wespeaker_on_CNCeleb1-2-LM_with_musan_rirs_debug
        other_args=
        root_path=/home/maduo
        data_path=/data/alimeeting/


        exp_dir=${root_path}/exp/speaker_diarization/ts_vad
        ts_vad_path=${root_path}/codebase/fairseq_speechtext/examples/speaker_diarization/ts_vad
        #speech_encoder_path=${root_path}/model_hub/ts_vad/ecapa-tdnn.model # Speaker encoder path
        #speech_encoder_path=${root_path}/.cache/modelscope/hub/iic/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt # Speaker encoder path

        # (TODO) maduo copy model from sribd
        speech_encoder_path=/home/maduo/model_hub/speaker_pretrain_model/zh/wespeaker/ecapa_tdnn_wespeaker_cnceleb1-2-LM/final_model.pt

        #spk_path=${root_path}/model_hub/ts_vad/spk_embed/SpeakerEmbedding
        spk_path=${root_path}/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding
        speech_encoder_type="ecapa_wespeaker"
        speaker_embedding_name_dir="ecapa_tdnn_wespeaker_cnceleb1-2-LM_feature_dir_debug"
        speaker_embed_dim=192
        rir_path=/data/rirs_noises/RIRS_NOISES
        musan_path=/data/musan
        mkdir -p ${exp_dir}/${exp_name}
        world_size=1
        update_freq=1

        fairseq-hydra-train \
          --config-dir ${ts_vad_path}/config \
          --config-name ${config_name} \
          hydra.run.dir=${exp_dir}/${exp_name} \
          hydra.job.name=${exp_name} \
          hydra.sweep.dir=${exp_dir}/${exp_name} \
          task.data=${data_path} \
          task.rir_path=${rir_path}\
          +task.musan_path=${musan_path}\
          +task.speaker_embedding_name_dir=$speaker_embedding_name_dir\
          distributed_training.distributed_world_size=${world_size}\
          distributed_training.distributed_port=-1\
          distributed_training.ddp_backend=legacy_ddp\
          optimization.update_freq=[${update_freq}]\
          task.speech_encoder_type=$speech_encoder_type\
          common.user_dir=${ts_vad_path} \
          +task.spk_path=${spk_path} \
          +model.speech_encoder_path=${speech_encoder_path} \
          +task.speaker_embed_dim=$speaker_embed_dim\
          ${other_args}

# one RTX3090: consume about 5 hours on training stage.
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ];then
   echo "train ts_vad  with musan_rirs using resnet34-lm  of wespeaker speaker embedding"
        config_name=base
        exp_name=baseline_ResNet34-TSTP-emb256-fbank80-num_frms200-aug0.6-spTrue-saFalse-ArcMargin-SGD-epoch150_exp2_on_CNCeleb1-2-LM_with_musan_rirs_exp2_debug
        other_args=
        root_path=/home/maduo
        data_path=/data/alimeeting/


        exp_dir=${root_path}/exp/speaker_diarization/ts_vad
        ts_vad_path=${root_path}/codebase/fairseq_speechtext/examples/speaker_diarization/ts_vad
        #speech_encoder_path=${root_path}/model_hub/ts_vad/ecapa-tdnn.model # Speaker encoder path
        speech_encoder_path=${root_path}/model_hub/speaker_pretrain_model/zh/wespeaker/ResNet34-TSTP-emb256-fbank80-num_frms200-aug0.6-spTrue-saFalse-ArcMargin-SGD-epoch150_exp2_on_CNCeleb1-2-LM/final_model.pt # Speaker encoder path

        #spk_path=${root_path}/model_hub/ts_vad/spk_embed/SpeakerEmbedding
        spk_path=${root_path}/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding
        speech_encoder_type="resnet34_wespeaker"
        speaker_embedding_name_dir="ResNet34-TSTP-emb256-fbank80-num_frms200-aug0.6-spTrue-saFalse-ArcMargin-SGD-epoch150_exp2_on_CNCeleb1-2-LM_feature_dir_debug"
        speaker_embed_dim=256
        rir_path=/data/rirs_noises/RIRS_NOISES
        musan_path=/data/musan
        mkdir -p ${exp_dir}/${exp_name}
        world_size=1
        update_freq=1

        fairseq-hydra-train \
          --config-dir ${ts_vad_path}/config \
          --config-name ${config_name} \
          hydra.run.dir=${exp_dir}/${exp_name} \
          hydra.job.name=${exp_name} \
          hydra.sweep.dir=${exp_dir}/${exp_name} \
          task.data=${data_path} \
          task.rir_path=${rir_path}\
          +task.musan_path=${musan_path}\
          +task.speaker_embedding_name_dir=$speaker_embedding_name_dir\
          distributed_training.distributed_world_size=${world_size}\
          distributed_training.distributed_port=-1\
          distributed_training.ddp_backend=legacy_ddp\
          optimization.update_freq=[${update_freq}]\
          task.speech_encoder_type=$speech_encoder_type\
          common.user_dir=${ts_vad_path} \
          +task.spk_path=${spk_path} \
	  +model.speech_encoder_path=${speech_encoder_path} \
          +task.speaker_embed_dim=$speaker_embed_dim\
	  +model.speaker_embed_dim=$speaker_embed_dim\
          ${other_args}

# one RTX3090: consume about 5 hours on training stage.
fi
## in order to check the  correct the func load_alimeeting_ts_embed() in ts_vad/data/ts_vad_dataset.py
if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ];then
   echo "train ts_vad  with musan_rirs using resnet34-lm  of wespeaker speaker embedding"
        config_name=base
        exp_name=baseline_ResNet34-TSTP-emb256-fbank80-num_frms200-aug0.6-spTrue-saFalse-ArcMargin-SGD-epoch150_exp2_on_CNCeleb1-2-LM_with_musan_rirs_exp2_debug2
        other_args=
        root_path=/home/maduo
        data_path=/data/alimeeting/


        exp_dir=${root_path}/exp/speaker_diarization/ts_vad
        ts_vad_path=${root_path}/codebase/fairseq_speechtext/examples/speaker_diarization/ts_vad
        #speech_encoder_path=${root_path}/model_hub/ts_vad/ecapa-tdnn.model # Speaker encoder path
        speech_encoder_path=${root_path}/model_hub/speaker_pretrain_model/zh/wespeaker/ResNet34-TSTP-emb256-fbank80-num_frms200-aug0.6-spTrue-saFalse-ArcMargin-SGD-epoch150_exp2_on_CNCeleb1-2-LM/final_model.pt # Speaker encoder path

        #spk_path=${root_path}/model_hub/ts_vad/spk_embed/SpeakerEmbedding
        spk_path=${root_path}/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding
        speech_encoder_type="resnet34_wespeaker"
        speaker_embedding_name_dir="ResNet34-TSTP-emb256-fbank80-num_frms200-aug0.6-spTrue-saFalse-ArcMargin-SGD-epoch150_exp2_on_CNCeleb1-2-LM_feature_dir_debug"
        speaker_embed_dim=256
        rir_path=/data/rirs_noises/RIRS_NOISES
        musan_path=/data/musan
        mkdir -p ${exp_dir}/${exp_name}
        world_size=1
        update_freq=1
        fairseq_dir=/home/maduo/codebase/fairseq_speechtext
        #fairseq-hydra-train \
	python $fairseq_dir/fairseq_cli/hydra_train.py\
          --config-dir ${ts_vad_path}/config \
          --config-name ${config_name} \
          hydra.run.dir=${exp_dir}/${exp_name} \
          hydra.job.name=${exp_name} \
          hydra.sweep.dir=${exp_dir}/${exp_name} \
          task.data=${data_path} \
          task.rir_path=${rir_path}\
          +task.musan_path=${musan_path}\
          +task.speaker_embedding_name_dir=$speaker_embedding_name_dir\
          distributed_training.distributed_world_size=${world_size}\
          distributed_training.distributed_port=-1\
          distributed_training.ddp_backend=legacy_ddp\
          optimization.update_freq=[${update_freq}]\
          task.speech_encoder_type=$speech_encoder_type\
          common.user_dir=${ts_vad_path} \
          +task.spk_path=${spk_path} \
          +model.speech_encoder_path=${speech_encoder_path} \
          +task.speaker_embed_dim=$speaker_embed_dim\
          +model.speaker_embed_dim=$speaker_embed_dim\
          ${other_args}
fi

if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ];then
   echo "train ts_vad  with musan_rirs using ecapa_tdnn_based_DINO_ft  of wespeaker speaker embedding"
        config_name=base
        exp_name=baseline_ecapa_tdnn_based_DINO_ft_on_CNCeleb1-2_devset-LM_with_musan_rirs_debug
        other_args=
        root_path=/home/maduo
        data_path=/data/alimeeting/


        exp_dir=${root_path}/exp/speaker_diarization/ts_vad
        ts_vad_path=${root_path}/codebase/fairseq_speechtext/examples/speaker_diarization/ts_vad
        #speech_encoder_path=${root_path}/model_hub/ts_vad/ecapa-tdnn.model # Speaker encoder path

	speech_encoder_path=/home/maduo/model_hub/speaker_pretrain_model/zh/wespeaker/ecapa_tdnn_based_DINO_ft_on_CNCeleb1-2_devset-LM/avg_model.pt
        #spk_path=${root_path}/model_hub/ts_vad/spk_embed/SpeakerEmbedding
        spk_path=${root_path}/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding
        speech_encoder_type="ecapa_wespeaker"
        speaker_embedding_name_dir="ecapa_tdnn_based_DINO_ft_on_CNCeleb1-2_devset-LM_feature_dir_debug"
        
	speaker_embed_dim=192
        rir_path=/data/rirs_noises/RIRS_NOISES
        musan_path=/data/musan
        mkdir -p ${exp_dir}/${exp_name}
        world_size=1
        update_freq=1
        fairseq_dir=/home/maduo/codebase/fairseq_speechtext
        #fairseq-hydra-train \
        python $fairseq_dir/fairseq_cli/hydra_train.py\
          --config-dir ${ts_vad_path}/config \
          --config-name ${config_name} \
          hydra.run.dir=${exp_dir}/${exp_name} \
          hydra.job.name=${exp_name} \
          hydra.sweep.dir=${exp_dir}/${exp_name} \
          task.data=${data_path} \
          task.rir_path=${rir_path}\
          +task.musan_path=${musan_path}\
          +task.speaker_embedding_name_dir=$speaker_embedding_name_dir\
          distributed_training.distributed_world_size=${world_size}\
          distributed_training.distributed_port=-1\
          distributed_training.ddp_backend=legacy_ddp\
          optimization.update_freq=[${update_freq}]\
          task.speech_encoder_type=$speech_encoder_type\
          common.user_dir=${ts_vad_path} \
          +task.spk_path=${spk_path} \
          +model.speech_encoder_path=${speech_encoder_path} \
          +task.speaker_embed_dim=$speaker_embed_dim\
	  +model.speaker_embed_dim=$speaker_embed_dim\
          ${other_args}
fi

if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ];then
	echo "train ts_vad  with musan_rirs using cam++ model of iic/speech_campplus_sv_zh-cn_16k-common(200k speakers) speaker embedding"
        config_name=base
        exp_name=baseline_cam++_speech_campplus_sv_zh-cn_16k-common_200k_speakers_with_musan_rirs
        other_args=
        root_path=/home/maduo
        data_path=/data/alimeeting/

        exp_dir=${root_path}/exp/speaker_diarization/ts_vad
        ts_vad_path=${root_path}/codebase/fairseq_speechtext/examples/speaker_diarization/ts_vad
        #speech_encoder_path=${root_path}/model_hub/ts_vad/ecapa-tdnn.model # Speaker encoder path 
	speech_encoder_path=${root_path}/.cache/modelscope/hub/iic/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin # Speaker encoder path
                           ## this is download from ts_vad/generate_chunk_speaker_embedding_from_modelscope_for_diarization.py
                           ## you can download this model manual follow commands
                           ##  git lfs install
                           ##  git clone https://www.modelscope.cn/iic/speech_campplus_sv_zh-cn_16k-common.git

        #spk_path=${root_path}/model_hub/ts_vad/spk_embed/SpeakerEmbedding
        spk_path=${root_path}/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding
        speech_encoder_type="cam++"
        speaker_embedding_name_dir="cam++_zh_cn_16k_common_feature_dir"
        speaker_embed_dim=192
        rir_path=/data/rirs_noises/RIRS_NOISES
        musan_path=/data/musan
        mkdir -p ${exp_dir}/${exp_name}
        world_size=1
        update_freq=1
        fairseq_dir=/home/maduo/codebase/fairseq_speechtext
        #fairseq-hydra-train \
        python $fairseq_dir/fairseq_cli/hydra_train.py\
          --config-dir ${ts_vad_path}/config \
          --config-name ${config_name} \
          hydra.run.dir=${exp_dir}/${exp_name} \
          hydra.job.name=${exp_name} \
          hydra.sweep.dir=${exp_dir}/${exp_name} \
          task.data=${data_path} \
          task.rir_path=${rir_path}\
          +task.musan_path=${musan_path}\
          +task.speaker_embedding_name_dir=$speaker_embedding_name_dir\
          distributed_training.distributed_world_size=${world_size}\
          distributed_training.distributed_port=-1\
          distributed_training.ddp_backend=legacy_ddp\
          optimization.update_freq=[${update_freq}]\
          task.speech_encoder_type=$speech_encoder_type\
          common.user_dir=${ts_vad_path} \
	  +task.spk_path=${spk_path} \
          +model.speech_encoder_path=${speech_encoder_path} \
          +task.speaker_embed_dim=$speaker_embed_dim\
          +model.speaker_embed_dim=$speaker_embed_dim\
          ${other_args}
fi
