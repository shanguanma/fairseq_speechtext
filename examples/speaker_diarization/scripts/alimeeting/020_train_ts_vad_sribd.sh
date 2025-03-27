#!/usr/bin/env bash


stage=0
stop_stage=1000
. utils/parse_options.sh
. path_for_fairseq_speechtext.sh
#. path_for_fsq_speechtext_py39.sh
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
   echo "train ts_vad  with musan_rirs using speech_campplus_sv_zh_en_16k-common_advanced of 3D-speaker speaker embedding"
    config_name=base
    exp_name=baseline_camppluse_zh_en_common_advanced_with_musan_rirs
    other_args=
    root_path=/mntcephfs/lab_data/maduo
    data_path=/mntcephfs/lab_data/maduo/datasets/alimeeting/

    exp_dir=${root_path}/exp/speaker_diarization/ts_vad
    ts_vad_path=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/ts_vad
    # Speaker encoder path
    speech_encoder_path=${root_path}/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt
    ## this is download from ts_vad/generate_speaker_embedding_from_modelscope.py
    ## you can download this model manual follow commands
    ##  git lfs install
    ##  git clone https://www.modelscope.cn/iic/speech_campplus_sv_zh_en_16k-common_advanced.git

    #spk_path=${root_path}/model_hub/ts_vad/spk_embed/SpeakerEmbedding
    spk_path=${root_path}/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding
    speech_encoder_type="cam++"
    speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
    speaker_embed_dim=192

    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    musan_path=/mntcephfs/lee_dataset/asr/musan
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

fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
   echo "train ts_vad  with musan_rirs using wavlm_large_ft of unispeech speaker embedding"
    config_name=base
    exp_name=baseline_wavlm_large_finetune_with_musan_rirs
    other_args=
    root_path=/mntcephfs/lab_data/maduo
    data_path=/mntcephfs/lab_data/maduo/datasets/alimeeting/

    exp_dir=${root_path}/exp/speaker_diarization/ts_vad
    ts_vad_path=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/ts_vad
    # Speaker encoder path
    speech_encoder_path=${root_path}/model_hub/speaker_pretrain_model/wavlm_ft/wavlm_large_finetune.pth
    ## this is download from ts_vad/generate_speaker_embedding_from_modelscope.py
    ## you can download this model manual follow commands
    ##  git lfs install
    ##  git clone https://www.modelscope.cn/iic/speech_campplus_sv_zh_en_16k-common_advanced.git

    #spk_path=${root_path}/model_hub/ts_vad/spk_embed/SpeakerEmbedding
    spk_path=${root_path}/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding
    speech_encoder_type="wavlm_large_ft"
    speaker_embedding_name_dir="wavlm_large_finetune_feature_dir"
    speaker_embed_dim=256
    update_extract=False # Expect not to update the SSL network part if False
    wavlm_pretrain_model=${root_path}/model_hub/speaker_pretrain_model/wavlm/WavLM-Large.pt

    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    musan_path=/mntcephfs/lee_dataset/asr/musan
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
      +model.update_extract=$update_extract\
      +model.wavlm_pretrain_model=$wavlm_pretrain_model\
      ${other_args}

fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
   echo "train ts_vad  with musan_rirs using wavlm_large_ft of unispeech speaker embedding"
    config_name=base
    exp_name=baseline_wavlm_large_finetune_update_extract_true_with_musan_rirs
    other_args=
    root_path=/mntcephfs/lab_data/maduo
    data_path=/mntcephfs/lab_data/maduo/datasets/alimeeting/

    exp_dir=${root_path}/exp/speaker_diarization/ts_vad
    ts_vad_path=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/ts_vad
    # Speaker encoder path
    speech_encoder_path=${root_path}/model_hub/speaker_pretrain_model/wavlm_ft/wavlm_large_finetune.pth
    ## this is download from ts_vad/generate_speaker_embedding_from_modelscope.py
    ## you can download this model manual follow commands
    ##  git lfs install
    ##  git clone https://www.modelscope.cn/iic/speech_campplus_sv_zh_en_16k-common_advanced.git

    #spk_path=${root_path}/model_hub/ts_vad/spk_embed/SpeakerEmbedding
    spk_path=${root_path}/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding
    speech_encoder_type="wavlm_large_ft"
    speaker_embedding_name_dir="wavlm_large_finetune_feature_dir"
    speaker_embed_dim=256
    update_extract=True # Expect not to update the SSL network part if False
    wavlm_pretrain_model=${root_path}/model_hub/speaker_pretrain_model/wavlm/WavLM-Large.pt

    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    musan_path=/mntcephfs/lee_dataset/asr/musan
    mkdir -p ${exp_dir}/${exp_name}
    world_size=1
    update_freq=1
    lr=2e-6
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
      optimization.lr=[$lr]\
      task.speech_encoder_type=$speech_encoder_type\
      common.user_dir=${ts_vad_path} \
      +task.spk_path=${spk_path} \
      +model.speech_encoder_path=${speech_encoder_path} \
      +task.speaker_embed_dim=$speaker_embed_dim\
      +model.speaker_embed_dim=$speaker_embed_dim\
      +model.update_extract=$update_extract\
      +model.wavlm_pretrain_model=$wavlm_pretrain_model\
      ${other_args}

fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
   echo "train ts_vad  with musan_rirs using speech_campplus_sv_zh_en_16k-common_advanced of 3D-speaker speaker embedding using two V100 gpus "
    config_name=base
    exp_name=baseline_camppluse_zh_en_common_advanced_with_musan_rirs_two_gpus
    other_args=
    root_path=/mntcephfs/lab_data/maduo
    data_path=/mntcephfs/lab_data/maduo/datasets/alimeeting/

    exp_dir=${root_path}/exp/speaker_diarization/ts_vad
    ts_vad_path=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/ts_vad
    # Speaker encoder path
    speech_encoder_path=${root_path}/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt
    ## this is download from ts_vad/generate_speaker_embedding_from_modelscope.py
    ## you can download this model manual follow commands
    ##  git lfs install
    ##  git clone https://www.modelscope.cn/iic/speech_campplus_sv_zh_en_16k-common_advanced.git

    #spk_path=${root_path}/model_hub/ts_vad/spk_embed/SpeakerEmbedding
    spk_path=${root_path}/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding
    speech_encoder_type="cam++"
    speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
    speaker_embed_dim=192

    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    musan_path=/mntcephfs/lee_dataset/asr/musan
    mkdir -p ${exp_dir}/${exp_name}
    world_size=2
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

fi
