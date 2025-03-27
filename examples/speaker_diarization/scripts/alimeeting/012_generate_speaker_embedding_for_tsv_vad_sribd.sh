#!/usr/bin/env bash


stage=0
stop_stage=1000
. utils/parse_options.sh
. path_for_fairseq_speechtext.sh
#. path_for_fsq_speechtext.sh
#. path_for_fsq_sptt.sh
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
   echo "prepare train target audio list"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   input_dir=/mntcephfs/lab_data/maduo/datasets/alimeeting/Train_Ali_far/target_audio/
   file=$input_dir/wavs.txt
    python source_md/prepare_alimeeting_target_audio_list.py \
	    $input_dir $file

fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
   echo "generate train speaker embedding"
   feature_name=cam++_en_zh_advanced_feature_dir
   model_id=iic/speech_campplus_sv_zh_en_16k-common_advanced
   # 提取embedding
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   input_dir=/mntcephfs/lab_data/maduo/datasets/alimeeting/Train_Ali_far/target_audio/
   file=$input_dir/wavs.txt
   wav_path=$file
   dest_dir=/mntcephfs/lab_data/maduo/model_hub/
   save_dir=$dest_dir/ts_vad/spk_embed/alimeeting/SpeakerEmbedding/Train/$feature_name
   python $fairseq_dir/examples/speaker_diarization/ts_vad/generate_chunk_speaker_embedding_from_modelscope_for_diarization.py\
	   --model_id $model_id --wavs $wav_path\
	   --save_dir $save_dir
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
   echo "prepare eval(dev) target audio list"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   input_dir=/mntcephfs/lab_data/maduo/datasets/alimeeting/Eval_Ali/Eval_Ali_far/target_audio/
   file=$input_dir/wavs.txt
    python source_md/prepare_alimeeting_target_audio_list.py \
            $input_dir $file

fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
   echo "generate eval(dev) speaker embedding"
   feature_name=cam++_en_zh_advanced_feature_dir
   model_id=iic/speech_campplus_sv_zh_en_16k-common_advanced
   # 提取embedding
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   input_dir=/mntcephfs/lab_data/maduo/datasets/alimeeting/Eval_Ali/Eval_Ali_far/target_audio/
   file=$input_dir/wavs.txt
   wav_path=$file
   dest_dir=/mntcephfs/lab_data/maduo/model_hub/
   save_dir=$dest_dir/ts_vad/spk_embed/alimeeting/SpeakerEmbedding/Eval/$feature_name
   python $fairseq_dir/examples/speaker_diarization/ts_vad/generate_chunk_speaker_embedding_from_modelscope_for_diarization.py\
           --model_id $model_id --wavs $wav_path\
           --save_dir $save_dir
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
   echo "prepare test set target audio list"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   input_dir=/mntcephfs/lab_data/maduo/datasets/alimeeting/Test_Ali/Test_Ali_far/target_audio/
   file=$input_dir/wavs.txt
    python source_md/prepare_alimeeting_target_audio_list.py \
            $input_dir $file

fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then
   echo "generate test set speaker embedding"
   feature_name=cam++_en_zh_advanced_feature_dir
   model_id=iic/speech_campplus_sv_zh_en_16k-common_advanced
   # 提取embedding
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   input_dir=/mntcephfs/lab_data/maduo/datasets/alimeeting/Test_Ali/Test_Ali_far/target_audio/
   file=$input_dir/wavs.txt
   wav_path=$file
   dest_dir=/mntcephfs/lab_data/maduo/model_hub/
   save_dir=$dest_dir/ts_vad/spk_embed/alimeeting/SpeakerEmbedding/Test/$feature_name
   python $fairseq_dir/examples/speaker_diarization/ts_vad/generate_chunk_speaker_embedding_from_modelscope_for_diarization.py\
           --model_id $model_id --wavs $wav_path\
           --save_dir $save_dir
fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ];then
   echo "generate train speaker embedding"
   feature_name=wavlm_large_finetune_feature_dir
   pretrained_speaker_model=/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm_ft/wavlm_large_finetune.pth
   pretrained_model=/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Large.pt
   # 提取embedding
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   input_dir=/mntcephfs/lab_data/maduo/datasets/alimeeting/Train_Ali_far/target_audio/
   file=$input_dir/wavs.txt
   wav_path=$file
   dest_dir=/mntcephfs/lab_data/maduo/model_hub/
   save_dir=$dest_dir/ts_vad/spk_embed/alimeeting/SpeakerEmbedding/Train/$feature_name
   python $fairseq_dir/examples/speaker_diarization/ts_vad/generate_chunk_speaker_embedding_from_SSL_for_diarization.py\
           --pretrained_speaker_model $pretrained_speaker_model\
           --pretrained_model ${pretrained_model} \
           --wavs $wav_path\
           --save_dir $save_dir\
           --model_name wavlm_large_ft
   echo "Train set finish extract!"
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ];then
   echo "generate dev and test set speaker embedding"
   feature_name=wavlm_large_finetune_feature_dir
   pretrained_speaker_model=/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm_ft/wavlm_large_finetune.pth
   pretrained_model=/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Large.pt

   # 提取embedding
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   #for name in Test Eval;do
   for name in Eval;do
    input_dir=/mntcephfs/lab_data/maduo/datasets/alimeeting/${name}_Ali/${name}_Ali_far/target_audio/
    file=$input_dir/wavs.txt
    wav_path=$file
    dest_dir=/mntcephfs/lab_data/maduo/model_hub/
    save_dir=$dest_dir/ts_vad/spk_embed/alimeeting/SpeakerEmbedding/$name/$feature_name
    python $fairseq_dir/examples/speaker_diarization/ts_vad/generate_chunk_speaker_embedding_from_SSL_for_diarization.py\
           --pretrained_speaker_model $pretrained_speaker_model\
           --pretrained_model ${pretrained_model} \
           --wavs $wav_path\
           --save_dir $save_dir\
           --model_name wavlm_large_ft
    echo "$name set finish extract!"
   done
fi
