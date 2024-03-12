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
   echo "infer alimeeting eval dataset via ts_vad model trained on alimeeting train_ali_far dataset."
	
	exp_name=baseline
	rs_len=4
	segment_shift=1
	gen_subset=Eval
	speech_encoder_type=ecapa

	code_path=/mnt/bn/junyi-nas2/codebase/joint-optimization
	data_path=/mnt/bd/alimeeting3/alimeeting_eval
        
	root_path=/home/maduo
        data_path=/home/maduo/dataset/alimeeting/
		
	exp_dir=${root_path}/exp/speaker_diarization/ts_vad
        ts_vad_path=${root_path}/codebase/fairseq_speechtext/examples/speaker_diarization/ts_vad
	spk_path=${root_path}/model_hub/ts_vad/spk_embed/SpeakerEmbedding
	results_path=${exp_dir}/${exp_name}/inf

	python3 ${ts_vad_path}/generate.py ${data_path} \
	  --user-dir ${ts_vad_path} \
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
	  --speech-encoder-type ${speech_encoder_type} 
#result :
# Model DER:  0.15990771624530234
#Model ACC:  0.9468786490975941
#100%|██████████| 25/25 [00:28<00:00,  1.15s/it]
#Eval for threshold 0.20: DER 12.63%, MS 0.87%, FA 11.24%, SC 0.52%

#Eval for threshold 0.30: DER 9.19%, MS 1.46%, FA 7.08%, SC 0.64%

#Eval for threshold 0.35: DER 8.22%, MS 1.87%, FA 5.66%, SC 0.70%

#Eval for threshold 0.40: DER 7.41%, MS 2.22%, FA 4.42%, SC 0.77%

#Eval for threshold 0.45: DER 6.86%, MS 2.65%, FA 3.40%, SC 0.80%

#Eval for threshold 0.50: DER 6.60%, MS 3.17%, FA 2.63%, SC 0.79% # as release result.

#Eval for threshold 0.55: DER 6.63%, MS 3.85%, FA 2.04%, SC 0.75%

#Eval for threshold 0.60: DER 6.76%, MS 4.53%, FA 1.52%, SC 0.70%

#Eval for threshold 0.70: DER 7.77%, MS 6.42%, FA 0.82%, SC 0.53%

#Eval for threshold 0.80: DER 10.07%, MS 9.34%, FA 0.45%, SC 0.29%


fi
