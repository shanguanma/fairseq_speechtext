#!/usr/bin/env bash


stage=0
stop_stage=1000

. utils/parse_options.sh
. path_for_nn_vad.sh
#vad_type="oracle"


if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ];then
   echo "prepare kaldi format for  magicdata-RAMC data"
   vad_code=/home/maduo/codebase/voice-activity-detection
   python $vad_code/prepare_magicdata_180h.py
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
   echo "prepare vad  magicdata-RAMC format data for train vad model "
   data_path=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/data/magicdata-RAMC/
   vad_code=/home/maduo/codebase/voice-activity-detection
   #for name in train dev test;do
   for name in train  test;do
      python $vad_code/prepared_vad_data_for_magicdata-RAMC.py \
              --data_path $data_path \
              --type $name
   done
fi
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
   echo "train a vad model using magicdata-RAMC dataset "
   vad_code=/home/maduo/codebase/voice-activity-detection
   
   CUDA_VISIBLE_DEVICES=0 python $vad_code/main.py train $vad_code/tests/configs/vad/train_config_magicdata-RAMC.yaml

fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
   echo "generate vad json file using pretrain transformer vad model "
   #echo "transformer vad model is from https://github.com/voithru/voice-activity-detection/blob/main/tests/checkpoints/vad/sample.checkpoint"
   echo "I used my pretrain vad model to get testset vad segement of magicdata-RAMC"
   vad_code=/home/maduo/codebase/voice-activity-detection
   vad_model=/home/maduo/codebase/voice-activity-detection/results/tests/vad/magicdata-RAMC-sample/v001/checkpoints
   data=data/magicdata-RAMC/test/
   output_dir=$data/predict_vad
   mkdir -p $output_dir
   for audio_path in `awk '{print $1}' $data/wav.scp`;do
      audio_name=$(basename $audio_path | sed s:.wav$::)
      python  $vad_code/main.py predict \
	        --threshold 0.91 \
	        --output-path $output_dir/${audio_name}.json \
                $audio_path\
	        $vad_model
   done
 fi

