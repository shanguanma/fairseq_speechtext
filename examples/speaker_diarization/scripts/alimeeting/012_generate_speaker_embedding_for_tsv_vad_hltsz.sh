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
   echo "prepare train target audio list"
   fairseq_dir=/home/maduo/codebase/fairseq_speechtext
   input_dir=/data/alimeeting/Train_Ali/Train_Ali_far/target_audio/
   file=$input_dir/wavs.txt
    python source_md/prepare_alimeeting_target_audio_list.py \
	    $input_dir $file
      
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
   echo "generate train speaker embedding"
   model_id=iic/speech_campplus_sv_zh_en_16k-common_advanced
   # 提取embedding
   fairseq_dir=/home/maduo/codebase/fairseq_speechtext
   input_dir=/data/alimeeting/Train_Ali/Train_Ali_far/target_audio/
   file=$input_dir/wavs.txt
   wav_path=$file
   save_dir=/home/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding/Train/cam++_en_zh_feature_dir
   python $fairseq_dir/examples/speaker_diarization/ts_vad/generate_chunk_speaker_embedding_from_modelscope_for_diarization.py\
	   --model_id $model_id --wavs $wav_path\
	   --save_dir $save_dir
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
   echo "prepare eval(dev) target audio list"
   fairseq_dir=/home/maduo/codebase/fairseq_speechtext
   input_dir=/data/alimeeting/Eval_Ali/Eval_Ali_far/target_audio/
   file=$input_dir/wavs.txt
    python source_md/prepare_alimeeting_target_audio_list.py \
            $input_dir $file

fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
   echo "generate eval(dev) speaker embedding"
   model_id=iic/speech_campplus_sv_zh_en_16k-common_advanced
   # 提取embedding
   fairseq_dir=/home/maduo/codebase/fairseq_speechtext
   input_dir=/data/alimeeting/Eval_Ali/Eval_Ali_far/target_audio/
   file=$input_dir/wavs.txt
   wav_path=$file
   save_dir=/home/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding/Eval/cam++_en_zh_feature_dir
   python $fairseq_dir/examples/speaker_diarization/ts_vad/generate_chunk_speaker_embedding_from_modelscope_for_diarization.py\
           --model_id $model_id --wavs $wav_path\
           --save_dir $save_dir
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
   echo "prepare test set target audio list"
   fairseq_dir=/home/maduo/codebase/fairseq_speechtext
   input_dir=/data/alimeeting/Test_Ali/Test_Ali_far/target_audio/
   file=$input_dir/wavs.txt
    python source_md/prepare_alimeeting_target_audio_list.py \
            $input_dir $file

fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then
   echo "generate test set speaker embedding"
   model_id=iic/speech_campplus_sv_zh_en_16k-common_advanced
   # 提取embedding
   fairseq_dir=/home/maduo/codebase/fairseq_speechtext
   input_dir=/data/alimeeting/Test_Ali/Test_Ali_far/target_audio/
   file=$input_dir/wavs.txt
   wav_path=$file
   save_dir=/home/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding/Test/cam++_en_zh_feature_dir
   python $fairseq_dir/examples/speaker_diarization/ts_vad/generate_chunk_speaker_embedding_from_modelscope_for_diarization.py\
           --model_id $model_id --wavs $wav_path\
           --save_dir $save_dir
fi





if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ];then
   echo "generate test set speaker embedding"
   #model_id=iic/speech_campplus_sv_zh_en_16k-common_advanced
   pretrained_model=/home/maduo/model_hub/speaker_pretrain_model/zh/wespeaker/ecapa_tdnn_based_DINO_ft_on_CNCeleb1-2_devset-LM/avg_model.pt
   # 提取embedding
   fairseq_dir=/home/maduo/codebase/fairseq_speechtext
   #for name in Test Eval Train;do
   #for name in Eval;do
   for name in Test Train;do
    input_dir=/data/alimeeting/${name}_Ali_far/target_audio/
    file=$input_dir/wavs.txt
    wav_path=$file
    save_dir=/home/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding/${name}/ecapa_tdnn_based_DINO_ft_on_CNCeleb1-2_devset-LM_feature_dir
    python $fairseq_dir/examples/speaker_diarization/ts_vad/generate_chunk_speaker_embedding_from_wespeaker_for_diarization.py\
           --pretrained_model ${pretrained_model} \
	   --wavs $wav_path\
           --save_dir $save_dir
   done
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ];then
   echo "generate alimeeting dev test data set speaker embedding"
   #model_id=iic/speech_campplus_sv_zh_en_16k-common_advanced
   pretrained_model=/home/maduo/model_hub/speaker_pretrain_model/zh/wespeaker/ResNet34-TSTP-emb256-fbank80-num_frms200-aug0.6-spTrue-saFalse-ArcMargin-SGD-epoch150_exp2_on_CNCeleb1-2-LM/final_model.pt
   # 提取embedding
   fairseq_dir=/home/maduo/codebase/fairseq_speechtext
   #for name in Test Eval Train;do
   #for name in Eval;do
   for name in Test Eval;do
    input_dir=/data/alimeeting/${name}_Ali/${name}_Ali_far/target_audio/
    
    file=$input_dir/wavs.txt
    wav_path=$file
    #save_dir=/home/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding/${name}/ResNet34-TSTP-emb256-fbank80-num_frms200-aug0.6-spTrue-saFalse-ArcMargin-SGD-epoch150_exp2_on_CNCeleb1-2-LM_feature_dir
    save_dir=/home/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding/${name}/ResNet34-TSTP-emb256-fbank80-num_frms200-aug0.6-spTrue-saFalse-ArcMargin-SGD-epoch150_exp2_on_CNCeleb1-2-LM_feature_dir_debug
    python $fairseq_dir/examples/speaker_diarization/ts_vad/generate_chunk_speaker_embedding_from_wespeaker_for_diarization.py\
           --pretrained_model ${pretrained_model} \
           --wavs $wav_path\
           --save_dir $save_dir\
	   --model_name ResNet34  
   done
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ];then
   echo "prepare test set target audio list"
   fairseq_dir=/home/maduo/codebase/fairseq_speechtext
   input_dir=/data/alimeeting/Train_Ali_far/target_audio/
   file=$input_dir/wavs.txt
    python source_md/prepare_alimeeting_target_audio_list.py \
            $input_dir $file
	
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ];then
   echo "generate alimeeting train data set speaker embedding"
   #model_id=iic/speech_campplus_sv_zh_en_16k-common_advanced
   pretrained_model=/home/maduo/model_hub/speaker_pretrain_model/zh/wespeaker/ResNet34-TSTP-emb256-fbank80-num_frms200-aug0.6-spTrue-saFalse-ArcMargin-SGD-epoch150_exp2_on_CNCeleb1-2-LM/final_model.pt
   # 提取embedding
   fairseq_dir=/home/maduo/codebase/fairseq_speechtext
   #for name in Test Eval Train;do
   #for name in Eval;do
   for name in Train;do
    input_dir=/data/alimeeting/Train_Ali_far/target_audio/
   
    file=$input_dir/wavs.txt
    wav_path=$file
    #save_dir=/home/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding/${name}/ResNet34-TSTP-emb256-fbank80-num_frms200-aug0.6-spTrue-saFalse-ArcMargin-SGD-epoch150_exp2_on_CNCeleb1-2-LM_feature_dir
    save_dir=/home/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding/${name}/ResNet34-TSTP-emb256-fbank80-num_frms200-aug0.6-spTrue-saFalse-ArcMargin-SGD-epoch150_exp2_on_CNCeleb1-2-LM_feature_dir_debug

    python $fairseq_dir/examples/speaker_diarization/ts_vad/generate_chunk_speaker_embedding_from_wespeaker_for_diarization.py\
           --pretrained_model ${pretrained_model} \
           --wavs $wav_path\
           --save_dir $save_dir\
           --model_name ResNet34
   done
fi





if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ];then
   echo "generate alimeeting dev test data set speaker embedding"
   #model_id=iic/speech_campplus_sv_zh_en_16k-common_advanced
   #(TODO) copy model from sribd
   #pretrained_model=/home/maduo/model_hub/speaker_pretrain_model/zh/wespeaker/ResNet34-TSTP-emb256-fbank80-num_frms200-aug0.6-spTrue-saFalse-ArcMargin-SGD-epoch150_exp2_on_CNCeleb1-2-LM/final_model.pt
    pretrained_model=/home/maduo/model_hub/speaker_pretrain_model/zh/wespeaker/ecapa_tdnn_wespeaker_cnceleb1-2-LM/final_model.pt
   # 提取embedding
   fairseq_dir=/home/maduo/codebase/fairseq_speechtext
   #for name in Test Eval Train;do
   #for name in Eval;do
   for name in Test;do
   #for name in Test Eval;do
    input_dir=/data/alimeeting/${name}_Ali/${name}_Ali_far/target_audio/

    file=$input_dir/wavs.txt
    wav_path=$file
   save_dir=/home/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding/${name}/ecapa_tdnn_wespeaker_cnceleb1-2-LM_feature_dir_debug
    python $fairseq_dir/examples/speaker_diarization/ts_vad/generate_chunk_speaker_embedding_from_wespeaker_for_diarization.py\
           --pretrained_model ${pretrained_model} \
           --wavs $wav_path\
           --save_dir $save_dir\
           --model_name ECAPA_TDNN_GLOB_c1024
   done
fi


if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ];then
   echo "generate alimeeting train data set speaker embedding"
   #model_id=iic/speech_campplus_sv_zh_en_16k-common_advanced
   pretrained_model=/home/maduo/model_hub/speaker_pretrain_model/zh/wespeaker/ecapa_tdnn_wespeaker_cnceleb1-2-LM/final_model.pt
   # 提取embedding
   fairseq_dir=/home/maduo/codebase/fairseq_speechtext
   #for name in Test Eval Train;do
   #for name in Eval;do
   for name in Train;do
    input_dir=/data/alimeeting/Train_Ali_far/target_audio/

    file=$input_dir/wavs.txt
    wav_path=$file
    save_dir=/home/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding/${name}/ecapa_tdnn_wespeaker_cnceleb1-2-LM_feature_dir_debug

    python $fairseq_dir/examples/speaker_diarization/ts_vad/generate_chunk_speaker_embedding_from_wespeaker_for_diarization.py\
           --pretrained_model ${pretrained_model} \
           --wavs $wav_path\
           --save_dir $save_dir\
           --model_name ECAPA_TDNN_GLOB_c1024
   done
fi


if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ];then
   echo "generate train set speaker embedding"
   #model_id=iic/speech_campplus_sv_zh_en_16k-common_advanced
   pretrained_model=/home/maduo/model_hub/speaker_pretrain_model/zh/wespeaker/ecapa_tdnn_based_DINO_ft_on_CNCeleb1-2_devset-LM/avg_model.pt
   # 提取embedding
   fairseq_dir=/home/maduo/codebase/fairseq_speechtext
   #for name in Test Eval Train;do
   #for name in Eval;do
   for name in Train;do
    input_dir=/data/alimeeting/${name}_Ali_far/target_audio/
    file=$input_dir/wavs.txt
    wav_path=$file
    save_dir=/home/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding/${name}/ecapa_tdnn_based_DINO_ft_on_CNCeleb1-2_devset-LM_feature_dir_debug
    python $fairseq_dir/examples/speaker_diarization/ts_vad/generate_chunk_speaker_embedding_from_wespeaker_for_diarization.py\
           --pretrained_model ${pretrained_model} \
           --wavs $wav_path\
           --save_dir $save_dir\
	   --model_name  ECAPA_TDNN_GLOB_c1024
   done
fi

if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ];then
   echo "generate dev and test set speaker embedding"
   #model_id=iic/speech_campplus_sv_zh_en_16k-common_advanced
   pretrained_model=/home/maduo/model_hub/speaker_pretrain_model/zh/wespeaker/ecapa_tdnn_based_DINO_ft_on_CNCeleb1-2_devset-LM/avg_model.pt
   # 提取embedding
   fairseq_dir=/home/maduo/codebase/fairseq_speechtext
   for name in Test Eval;do
   #for name in Eval;do
    input_dir=/data/alimeeting/${name}_Ali/${name}_Ali_far/target_audio/
    file=$input_dir/wavs.txt
    wav_path=$file
    save_dir=/home/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding/${name}/ecapa_tdnn_based_DINO_ft_on_CNCeleb1-2_devset-LM_feature_dir_debug
    python $fairseq_dir/examples/speaker_diarization/ts_vad/generate_chunk_speaker_embedding_from_wespeaker_for_diarization.py\
           --pretrained_model ${pretrained_model} \
           --wavs $wav_path\
           --save_dir $save_dir\
           --model_name  ECAPA_TDNN_GLOB_c1024
   done
fi



if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ];then
   echo "generate train speaker embedding using Chinese common 200k speaker"
   model_id=iic/speech_campplus_sv_zh-cn_16k-common
   # 提取embedding
   fairseq_dir=/home/maduo/codebase/fairseq_speechtext
   for name in Train;do
    input_dir=/data/alimeeting/${name}_Ali_far/target_audio/
    file=$input_dir/wavs.txt
    wav_path=$file
    save_dir=/home/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding/$name/cam++_zh_cn_16k_common_feature_dir
    python $fairseq_dir/examples/speaker_diarization/ts_vad/generate_chunk_speaker_embedding_from_modelscope_for_diarization.py\
           --model_id $model_id --wavs $wav_path\
           --save_dir $save_dir
   done
fi

if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ];then
   echo "generate dev test speaker embedding using Chinese common 200k speaker"
   model_id=iic/speech_campplus_sv_zh-cn_16k-common
   # 提取embedding
   fairseq_dir=/home/maduo/codebase/fairseq_speechtext
   for name in Test Eval;do
    input_dir=/data/alimeeting/${name}_Ali/${name}_Ali_far/target_audio/
    file=$input_dir/wavs.txt
    wav_path=$file
    save_dir=/home/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding/$name/cam++_zh_cn_16k_common_feature_dir
    python $fairseq_dir/examples/speaker_diarization/ts_vad/generate_chunk_speaker_embedding_from_modelscope_for_diarization.py\
           --model_id $model_id --wavs $wav_path\
           --save_dir $save_dir
   done
fi
