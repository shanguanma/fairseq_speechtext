#!/usr/bin/env bash


stage=0
stop_stage=1000
. utils/parse_options.sh
#. path_for_fairseq_speechtext.sh
#. path_for_fsq_speechtext.sh
. path_for_fsq_sptt.sh
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
export OC_CAUSE=1 # for full backtrace


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
        ts_vad_path=${root_path}/codebase/fairseq_speechtext/examples/speaker_diarization
	spk_path=${root_path}/model_hub/ts_vad/spk_embed/SpeakerEmbedding
	results_path=${exp_dir}/${exp_name}/inf

	rttm_dir=$root_path/model_hub/ts_vad
	sctk_tool_path=$ts_vad_path/SCTK-2.4.12

	python3 ${ts_vad_path}/ts_vad/generate.py ${data_path} \
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
	  --sctk_tool_path ${sctk_tool_path} 
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

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
   echo "infer alimeeting eval dataset via ts_vad model trained on alimeeting train_ali_far dataset with musan and rirs noise"

        exp_name=baseline_with_musan_rirs
        rs_len=4
        segment_shift=1
        gen_subset=Eval
        speech_encoder_type=ecapa

        code_path=/mnt/bn/junyi-nas2/codebase/joint-optimization
        data_path=/mnt/bd/alimeeting3/alimeeting_eval

        root_path=/home/maduo
        data_path=/home/maduo/dataset/alimeeting/

        exp_dir=${root_path}/exp/speaker_diarization/ts_vad
        ts_vad_path=${root_path}/codebase/fairseq_speechtext/examples/speaker_diarization
        spk_path=${root_path}/model_hub/ts_vad/spk_embed/SpeakerEmbedding
        results_path=${exp_dir}/${exp_name}/inf

        rttm_dir=$root_path/model_hub/ts_vad
        sctk_tool_path=$ts_vad_path/SCTK-2.4.12

        python3 ${ts_vad_path}/ts_vad/generate.py ${data_path} \
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
          --sctk_tool_path ${sctk_tool_path}

# result:
# Model DER:  0.15053331683643045
# Model ACC:  0.9496916344496984
#100%|██████████| 25/25 [00:29<00:00,  1.19s/it]
#Eval for threshold 0.20: DER 9.43%, MS 1.15%, FA 7.74%, SC 0.54%

#Eval for threshold 0.30: DER 7.25%, MS 1.79%, FA 4.81%, SC 0.66%

#Eval for threshold 0.35: DER 6.66%, MS 2.14%, FA 3.80%, SC 0.72%

#Eval for threshold 0.40: DER 6.28%, MS 2.48%, FA 3.02%, SC 0.78%

#Eval for threshold 0.45: DER 6.08%, MS 2.89%, FA 2.38%, SC 0.81%

#Eval for threshold 0.50: DER 6.03%, MS 3.39%, FA 1.83%, SC 0.82% as release result

#Eval for threshold 0.55: DER 6.16%, MS 3.98%, FA 1.40%, SC 0.78%

#Eval for threshold 0.60: DER 6.40%, MS 4.62%, FA 1.06%, SC 0.71%

#Eval for threshold 0.70: DER 7.45%, MS 6.28%, FA 0.70%, SC 0.47%

#Eval for threshold 0.80: DER 9.48%, MS 8.71%, FA 0.47%, SC 0.30%
fi
