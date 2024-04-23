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
 
	root_path=/home/maduo
        data_path=/data/alimeeting/
		
	exp_dir=${root_path}/exp/speaker_diarization/ts_vad
        ts_vad_path=${root_path}/codebase/fairseq_speechtext/examples/speaker_diarization
	spk_path=${root_path}/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding
	results_path=${exp_dir}/${exp_name}/inf

	rttm_dir=$root_path/model_hub/ts_vad
	sctk_tool_path=$ts_vad_path/SCTK-2.4.12
        rttm_name=alimeeting_eval
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
	  --sctk_tool_path ${sctk_tool_path} \
	  --rttm_name $rttm_name
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

        root_path=/home/maduo
        data_path=/data/alimeeting/

        exp_dir=${root_path}/exp/speaker_diarization/ts_vad
        ts_vad_path=${root_path}/codebase/fairseq_speechtext/examples/speaker_diarization
        spk_path=${root_path}/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding
        results_path=${exp_dir}/${exp_name}/inf

        rttm_dir=$root_path/model_hub/ts_vad
        sctk_tool_path=$ts_vad_path/SCTK-2.4.12
        rttm_name=alimeeting_eval
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
          --sctk_tool_path ${sctk_tool_path}\
	  --rttm_name $rttm_name

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

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
   echo "infer alimeeting eval dataset via ts_vad model trained on alimeeting train_ali_far dataset with musan and rirs noise"
       
        exp_name=baseline_camppluse_zh_en_common_advanced_with_musan_rirs
        rs_len=4
        segment_shift=1
        gen_subset=Eval
        speech_encoder_type="cam++"
        speaker_embedding_name_dir="cam++_en_zh_feature_dir"
        root_path=/home/maduo
        data_path=/data/alimeeting/

        exp_dir=${root_path}/exp/speaker_diarization/ts_vad
        ts_vad_path=${root_path}/codebase/fairseq_speechtext/examples/speaker_diarization
        spk_path=${root_path}/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding
        results_path=${exp_dir}/${exp_name}/inf/${gen_subset}

        rttm_dir=$root_path/model_hub/ts_vad
        sctk_tool_path=$ts_vad_path/SCTK-2.4.12
        rttm_name=alimeeting_eval ## offer groud truth label
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
          --sctk_tool_path ${sctk_tool_path}\
          --rttm_name $rttm_name\
	  --speaker_embedding_name_dir $speaker_embedding_name_dir

#Model DER:  0.1335449615491137
#Model ACC:  0.9569092086727427
#100%|██████████| 25/25 [00:26<00:00,  1.08s/it]
#Eval for threshold 0.20: DER 7.52%, MS 1.00%, FA 6.24%, SC 0.28%
#
#Eval for threshold 0.30: DER 5.59%, MS 1.51%, FA 3.77%, SC 0.31%
#
#Eval for threshold 0.35: DER 5.03%, MS 1.76%, FA 2.97%, SC 0.31%
#
#Eval for threshold 0.40: DER 4.79%, MS 2.09%, FA 2.38%, SC 0.32%
#
#Eval for threshold 0.45: DER 4.64%, MS 2.42%, FA 1.89%, SC 0.33%
#
#Eval for threshold 0.50: DER 4.65%, MS 2.78%, FA 1.54%, SC 0.33%
#
#Eval for threshold 0.55: DER 4.80%, MS 3.23%, FA 1.25%, SC 0.31%
#
#Eval for threshold 0.60: DER 5.02%, MS 3.72%, FA 1.02%, SC 0.28%
#
#Eval for threshold 0.70: DER 6.07%, MS 5.17%, FA 0.71%, SC 0.20%
#
#Eval for threshold 0.80: DER 7.73%, MS 7.14%, FA 0.46%, SC 0.12%
#

fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
   echo "infer alimeeting test dataset via ts_vad model trained on alimeeting train_ali_far dataset with musan and rirs noise"

        exp_name=baseline_camppluse_zh_en_common_advanced_with_musan_rirs
        rs_len=4
        segment_shift=1
        gen_subset=Test
        speech_encoder_type="cam++"
        speaker_embedding_name_dir="cam++_en_zh_feature_dir"
        root_path=/home/maduo
        data_path=/data/alimeeting/

        exp_dir=${root_path}/exp/speaker_diarization/ts_vad
        ts_vad_path=${root_path}/codebase/fairseq_speechtext/examples/speaker_diarization
        spk_path=${root_path}/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding
        results_path=${exp_dir}/${exp_name}/inf/${gen_subset}

        rttm_dir=$root_path/model_hub/ts_vad
        sctk_tool_path=$ts_vad_path/SCTK-2.4.12
        rttm_name=alimeeting_test ## offer groud truth label
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
          --sctk_tool_path ${sctk_tool_path}\
          --rttm_name $rttm_name\
          --speaker_embedding_name_dir $speaker_embedding_name_dir

#Model DER:  0.1347407789658194
#Model ACC:  0.9536766740797219
#100%|██████████| 60/60 [01:12<00:00,  1.20s/it]
#Eval for threshold 0.20: DER 7.83%, MS 1.16%, FA 6.22%, SC 0.46%
#
#Eval for threshold 0.30: DER 5.85%, MS 1.78%, FA 3.49%, SC 0.59%
#
#Eval for threshold 0.35: DER 5.43%, MS 2.13%, FA 2.69%, SC 0.61%
#
#Eval for threshold 0.40: DER 5.17%, MS 2.52%, FA 2.02%, SC 0.63%
#
#Eval for threshold 0.45: DER 5.09%, MS 2.96%, FA 1.50%, SC 0.64%
#
#Eval for threshold 0.50: DER 5.23%, MS 3.45%, FA 1.16%, SC 0.62%
#
#Eval for threshold 0.55: DER 5.47%, MS 3.99%, FA 0.89%, SC 0.59%
#
#Eval for threshold 0.60: DER 5.86%, MS 4.66%, FA 0.66%, SC 0.54%
#
#Eval for threshold 0.70: DER 7.11%, MS 6.32%, FA 0.37%, SC 0.42%
#
#Eval for threshold 0.80: DER 9.27%, MS 8.79%, FA 0.21%, SC 0.27%
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
   echo "infer alimeeting eval dataset via ts_vad model trained on alimeeting train_ali_far dataset with rirs noise"

        exp_name=baseline_camppluse_zh_en_common_advanced_with_rirs
        rs_len=4
        segment_shift=1
        gen_subset=Eval
        speech_encoder_type="cam++"
        speaker_embedding_name_dir="cam++_en_zh_feature_dir"
        root_path=/home/maduo
        data_path=/data/alimeeting/

        exp_dir=${root_path}/exp/speaker_diarization/ts_vad
        ts_vad_path=${root_path}/codebase/fairseq_speechtext/examples/speaker_diarization
        spk_path=${root_path}/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding
        results_path=${exp_dir}/${exp_name}/inf/${gen_subset}

        rttm_dir=$root_path/model_hub/ts_vad
        sctk_tool_path=$ts_vad_path/SCTK-2.4.12
        rttm_name=alimeeting_eval ## offer groud truth label
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
          --sctk_tool_path ${sctk_tool_path}\
          --rttm_name $rttm_name\
          --speaker_embedding_name_dir $speaker_embedding_name_dir

#Model DER:  0.1443200708969706
#Model ACC:  0.9522853090899802
#100%|██████████| 25/25 [00:29<00:00,  1.18s/it]
#Eval for threshold 0.20: DER 9.23%, MS 0.99%, FA 7.84%, SC 0.40%
#
#Eval for threshold 0.30: DER 6.73%, MS 1.63%, FA 4.57%, SC 0.53%
#
#Eval for threshold 0.35: DER 6.09%, MS 1.94%, FA 3.57%, SC 0.58%
#
#Eval for threshold 0.40: DER 5.79%, MS 2.32%, FA 2.86%, SC 0.61%
#
#Eval for threshold 0.45: DER 5.66%, MS 2.75%, FA 2.26%, SC 0.65%
#
#Eval for threshold 0.50: DER 5.67%, MS 3.26%, FA 1.77%, SC 0.64%
#
#Eval for threshold 0.55: DER 5.79%, MS 3.79%, FA 1.38%, SC 0.62%
#
#Eval for threshold 0.60: DER 6.01%, MS 4.39%, FA 1.03%, SC 0.59%
#
#Eval for threshold 0.70: DER 7.09%, MS 6.05%, FA 0.63%, SC 0.41%
#
#Eval for threshold 0.80: DER 9.17%, MS 8.53%, FA 0.41%, SC 0.22%

fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then
   echo "infer alimeeting test dataset via ts_vad model trained on alimeeting train_ali_far dataset with rirs noise"

        exp_name=baseline_camppluse_zh_en_common_advanced_with_rirs
        rs_len=4
        segment_shift=1
        gen_subset=Test
        speech_encoder_type="cam++"
        speaker_embedding_name_dir="cam++_en_zh_feature_dir"
        root_path=/home/maduo
        data_path=/data/alimeeting/

        exp_dir=${root_path}/exp/speaker_diarization/ts_vad
        ts_vad_path=${root_path}/codebase/fairseq_speechtext/examples/speaker_diarization
        spk_path=${root_path}/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding
        results_path=${exp_dir}/${exp_name}/inf/${gen_subset}

        rttm_dir=$root_path/model_hub/ts_vad
        sctk_tool_path=$ts_vad_path/SCTK-2.4.12
        rttm_name=alimeeting_Test ## offer groud truth label
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
          --sctk_tool_path ${sctk_tool_path}\
          --rttm_name $rttm_name\
          --speaker_embedding_name_dir $speaker_embedding_name_dir

#	Model DER:  0.14041944620757557
#Model ACC:  0.9512467831828949
#100%|██████████| 60/60 [01:12<00:00,  1.22s/it]
#Eval for threshold 0.20: DER 8.97%, MS 1.19%, FA 7.30%, SC 0.47%
#
#Eval for threshold 0.30: DER 6.66%, MS 1.82%, FA 4.21%, SC 0.64%
#
#Eval for threshold 0.35: DER 6.12%, MS 2.18%, FA 3.23%, SC 0.71%
#
#Eval for threshold 0.40: DER 5.79%, MS 2.58%, FA 2.45%, SC 0.75%
#
#Eval for threshold 0.45: DER 5.63%, MS 3.04%, FA 1.80%, SC 0.80%
#
#Eval for threshold 0.50: DER 5.73%, MS 3.59%, FA 1.32%, SC 0.81%
#
#Eval for threshold 0.55: DER 5.96%, MS 4.25%, FA 0.92%, SC 0.79%
#
#Eval for threshold 0.60: DER 6.41%, MS 5.03%, FA 0.66%, SC 0.73%
#
#Eval for threshold 0.70: DER 7.84%, MS 7.00%, FA 0.32%, SC 0.53%
#
#Eval for threshold 0.80: DER 10.26%, MS 9.77%, FA 0.16%, SC 0.33%


fi
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ];then
   echo "infer alimeeting eval dataset via ts_vad model trained on alimeeting train_ali_far dataset with musan and rirs noise"

        exp_name=baseline_ecapa_tdnn_based_DINO_ft_on_CNCeleb1-2_devset-LM_with_musan_rirs
        rs_len=4
        segment_shift=1
        gen_subset=Eval ## choice from "Eval" and "Test"
	speech_encoder_type="ecapa_wespeaker"
        speaker_embedding_name_dir="ecapa_tdnn_based_DINO_ft_on_CNCeleb1-2_devset-LM_feature_dir"
        root_path=/home/maduo
        data_path=/data/alimeeting/

        exp_dir=${root_path}/exp/speaker_diarization/ts_vad
        ts_vad_path=${root_path}/codebase/fairseq_speechtext/examples/speaker_diarization
        spk_path=${root_path}/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding
        results_path=${exp_dir}/${exp_name}/inf/${gen_subset}

        rttm_dir=$root_path/model_hub/ts_vad
        sctk_tool_path=$ts_vad_path/SCTK-2.4.12
        rttm_name=alimeeting_eval ## offer groud truth label
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
          --sctk_tool_path ${sctk_tool_path}\
          --rttm_name $rttm_name\
          --speaker_embedding_name_dir $speaker_embedding_name_dir
 # 2024-04-08 11:14:12 | INFO | ts_vad.data.ts_vad_dataset | loaded sentence=15141, shortest sent=5120.0, longest sent=64000.0, rs_len=4, segment_shift=1, rir=False, musan=False, noise_ratio=0.5
#2024-04-08 11:14:12 | INFO | fairseq.tasks.fairseq_task | can_reuse_epoch_itr = True
#2024-04-08 11:14:12 | INFO | fairseq.tasks.fairseq_task | reuse_dataloader = True
#2024-04-08 11:14:12 | INFO | fairseq.tasks.fairseq_task | rebuild_batches = False
#2024-04-08 11:14:12 | INFO | fairseq.tasks.fairseq_task | creating new batches for epoch 1
#Model DER:  0.5056185605733574
#Model ACC:  0.8222160498806
#100%|██████████| 25/25 [00:29<00:00,  1.17s/it]
#Eval for threshold 0.20: DER 97.22%, MS 0.62%, FA 96.17%, SC 0.44%
#
#Eval for threshold 0.30: DER 74.70%, MS 0.98%, FA 72.36%, SC 1.36%
#
#Eval for threshold 0.35: DER 65.03%, MS 1.21%, FA 61.75%, SC 2.07%
#
#Eval for threshold 0.40: DER 55.54%, MS 1.51%, FA 50.94%, SC 3.10%
#
#Eval for threshold 0.45: DER 46.04%, MS 1.94%, FA 39.82%, SC 4.28%
#
#Eval for threshold 0.50: DER 37.63%, MS 2.81%, FA 29.21%, SC 5.60%
#
#Eval for threshold 0.55: DER 30.97%, MS 4.36%, FA 19.98%, SC 6.63%
#
#Eval for threshold 0.60: DER 26.62%, MS 7.28%, FA 12.53%, SC 6.81%
#
#Eval for threshold 0.70: DER 27.09%, MS 18.65%, FA 3.75%, SC 4.69%
#
#Eval for threshold 0.80: DER 39.26%, MS 36.74%, FA 1.31%, SC 1.21%
#
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ];then
   echo "infer alimeeting eval dataset via ts_vad model trained on alimeeting train_ali_far dataset with musan and rirs noise"

        exp_name=baseline_ResNet34-TSTP-emb256-fbank80-num_frms200-aug0.6-spTrue-saFalse-ArcMargin-SGD-epoch150_exp2_on_CNCeleb1-2-LM_with_musan_rirs
	rs_len=4
        segment_shift=1
        gen_subset=Eval ## choice from "Eval" and "Test"
        #speech_encoder_type="ecapa_wespeaker"
        #speaker_embedding_name_dir="ecapa_tdnn_based_DINO_ft_on_CNCeleb1-2_devset-LM_feature_dir"
        speech_encoder_type="resnet34_wespeaker"
        speaker_embedding_name_dir="ResNet34-TSTP-emb256-fbank80-num_frms200-aug0.6-spTrue-saFalse-ArcMargin-SGD-epoch150_exp2_on_CNCeleb1-2-LM_feature_dir"
	root_path=/home/maduo
        data_path=/data/alimeeting/

        exp_dir=${root_path}/exp/speaker_diarization/ts_vad
        ts_vad_path=${root_path}/codebase/fairseq_speechtext/examples/speaker_diarization
        spk_path=${root_path}/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding
        results_path=${exp_dir}/${exp_name}/inf/${gen_subset}

        rttm_dir=$root_path/model_hub/ts_vad
        sctk_tool_path=$ts_vad_path/SCTK-2.4.12
        rttm_name=alimeeting_eval ## offer groud truth label
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
          --sctk_tool_path ${sctk_tool_path}\
          --rttm_name $rttm_name\
          --speaker_embedding_name_dir $speaker_embedding_name_dir

# Model DER:  0.4986899776724591
# Model ACC:  0.8312962633185514
#100%|██████████| 25/25 [00:28<00:00,  1.13s/it]
#Eval for threshold 0.20: DER 84.95%, MS 0.38%, FA 83.52%, SC 1.06%
#
#Eval for threshold 0.30: DER 73.46%, MS 1.36%, FA 69.60%, SC 2.50%
#
#Eval for threshold 0.35: DER 57.45%, MS 2.89%, FA 49.16%, SC 5.40%
#
#Eval for threshold 0.40: DER 48.17%, MS 12.24%, FA 29.25%, SC 6.69%
#
#Eval for threshold 0.45: DER 47.90%, MS 24.14%, FA 19.75%, SC 4.02%
#
#Eval for threshold 0.50: DER 44.57%, MS 28.95%, FA 10.01%, SC 5.61%
#
#Eval for threshold 0.55: DER 45.24%, MS 33.33%, FA 5.98%, SC 5.93%
#
#Eval for threshold 0.60: DER 48.56%, MS 42.10%, FA 3.79%, SC 2.67%
#
#Eval for threshold 0.70: DER 52.44%, MS 49.68%, FA 1.06%, SC 1.70%
#
#Eval for threshold 0.80: DER 56.62%, MS 55.55%, FA 0.16%, SC 0.91%

fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ];then
   echo "infer alimeeting eval dataset via ts_vad model trained on alimeeting train_ali_far dataset with musan and rirs noise"

        exp_name=baseline_ecapa_tdnn_wespeaker_on_CNCeleb1-2-LM_with_musan_rirs_debug
        rs_len=4
        segment_shift=1
        gen_subset=Eval ## choice from "Eval" and "Test"
        #speech_encoder_type="ecapa_wespeaker"
        #speaker_embedding_name_dir="ecapa_tdnn_based_DINO_ft_on_CNCeleb1-2_devset-LM_feature_dir"
        speech_encoder_type="ecapa_wespeaker"
        speaker_embedding_name_dir="ecapa_tdnn_wespeaker_cnceleb1-2-LM_feature_dir_debug"
        root_path=/home/maduo
        data_path=/data/alimeeting/

        exp_dir=${root_path}/exp/speaker_diarization/ts_vad
        ts_vad_path=${root_path}/codebase/fairseq_speechtext/examples/speaker_diarization
        spk_path=${root_path}/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding
        results_path=${exp_dir}/${exp_name}/inf/${gen_subset}

        rttm_dir=$root_path/model_hub/ts_vad
        sctk_tool_path=$ts_vad_path/SCTK-2.4.12
        rttm_name=alimeeting_eval ## offer groud truth label
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
          --sctk_tool_path ${sctk_tool_path}\
          --rttm_name $rttm_name\
          --speaker_embedding_name_dir $speaker_embedding_name_dir
#Model DER:  0.1607081853903199
#Model ACC:  0.9463613199452096
#100%|██████████| 25/25 [00:26<00:00,  1.08s/it]
#Eval for threshold 0.20: DER 12.32%, MS 1.03%, FA 10.87%, SC 0.43%
#
#Eval for threshold 0.30: DER 8.58%, MS 1.68%, FA 6.29%, SC 0.61%
#
#Eval for threshold 0.35: DER 7.75%, MS 2.06%, FA 5.03%, SC 0.66%
#
#Eval for threshold 0.40: DER 7.16%, MS 2.52%, FA 3.93%, SC 0.72%
#
#Eval for threshold 0.45: DER 6.84%, MS 3.03%, FA 3.01%, SC 0.80%
#
#Eval for threshold 0.50: DER 6.70%, MS 3.68%, FA 2.17%, SC 0.85%
#
#Eval for threshold 0.55: DER 6.88%, MS 4.42%, FA 1.64%, SC 0.82%
#
#Eval for threshold 0.60: DER 7.39%, MS 5.45%, FA 1.23%, SC 0.72%
#
#Eval for threshold 0.70: DER 8.94%, MS 7.71%, FA 0.71%, SC 0.52%
#
#Eval for threshold 0.80: DER 11.75%, MS 11.03%, FA 0.43%, SC 0.29%
	
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ];then
   echo "infer alimeeting eval dataset via ts_vad model trained on alimeeting train_ali_far dataset with musan and rirs noise"

        exp_name=baseline_ecapa_tdnn_wespeaker_on_CNCeleb1-2-LM_with_musan_rirs_debug
        rs_len=4
        segment_shift=1
        gen_subset=Test ## choice from "Eval" and "Test"
        #speech_encoder_type="ecapa_wespeaker"
        #speaker_embedding_name_dir="ecapa_tdnn_based_DINO_ft_on_CNCeleb1-2_devset-LM_feature_dir"
        speech_encoder_type="ecapa_wespeaker"
        speaker_embedding_name_dir="ecapa_tdnn_wespeaker_cnceleb1-2-LM_feature_dir_debug"
        root_path=/home/maduo
        data_path=/data/alimeeting/

        exp_dir=${root_path}/exp/speaker_diarization/ts_vad
        ts_vad_path=${root_path}/codebase/fairseq_speechtext/examples/speaker_diarization
        spk_path=${root_path}/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding
        results_path=${exp_dir}/${exp_name}/inf/${gen_subset}

        rttm_dir=$root_path/model_hub/ts_vad
        sctk_tool_path=$ts_vad_path/SCTK-2.4.12
        rttm_name=alimeeting_test ## offer groud truth label
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
          --sctk_tool_path ${sctk_tool_path}\
          --rttm_name $rttm_name\
          --speaker_embedding_name_dir $speaker_embedding_name_dir

## Model DER:  0.16744439716349727
#Model ACC:  0.9391902470207185
#100%|██████████| 60/60 [01:09<00:00,  1.16s/it]
#Eval for threshold 0.20: DER 13.90%, MS 1.25%, FA 11.71%, SC 0.94%
#
#Eval for threshold 0.30: DER 9.99%, MS 2.00%, FA 6.63%, SC 1.35%
#
#Eval for threshold 0.35: DER 8.98%, MS 2.44%, FA 4.98%, SC 1.55%
#
#Eval for threshold 0.40: DER 8.32%, MS 2.95%, FA 3.63%, SC 1.75%
#
#Eval for threshold 0.45: DER 7.98%, MS 3.54%, FA 2.57%, SC 1.87%
#
#Eval for threshold 0.50: DER 7.88%, MS 4.19%, FA 1.73%, SC 1.96%
#
#Eval for threshold 0.55: DER 8.18%, MS 5.12%, FA 1.16%, SC 1.90%
#
#Eval for threshold 0.60: DER 8.80%, MS 6.30%, FA 0.79%, SC 1.72%
#
#Eval for threshold 0.70: DER 10.83%, MS 9.27%, FA 0.36%, SC 1.21%
#
#Eval for threshold 0.80: DER 14.19%, MS 13.34%, FA 0.16%, SC 0.69%
# 
fi


if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ];then
   echo "infer alimeeting eval dataset via ts_vad model trained on alimeeting train_ali_far dataset with musan and rirs noise"

        exp_name=baseline_ResNet34-TSTP-emb256-fbank80-num_frms200-aug0.6-spTrue-saFalse-ArcMargin-SGD-epoch150_exp2_on_CNCeleb1-2-LM_with_musan_rirs_exp2_debug
        rs_len=4
        segment_shift=1
        gen_subset=Eval ## choice from "Eval" and "Test"
        #speech_encoder_type="ecapa_wespeaker"
        #speaker_embedding_name_dir="ecapa_tdnn_based_DINO_ft_on_CNCeleb1-2_devset-LM_feature_dir"
        speech_encoder_type="resnet34_wespeaker"
        speaker_embedding_name_dir="ResNet34-TSTP-emb256-fbank80-num_frms200-aug0.6-spTrue-saFalse-ArcMargin-SGD-epoch150_exp2_on_CNCeleb1-2-LM_feature_dir_debug"
        speaker_embed_dim=256
	root_path=/home/maduo
        data_path=/data/alimeeting/

        exp_dir=${root_path}/exp/speaker_diarization/ts_vad
        ts_vad_path=${root_path}/codebase/fairseq_speechtext/examples/speaker_diarization
        spk_path=${root_path}/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding
        results_path=${exp_dir}/${exp_name}/inf/${gen_subset}

        rttm_dir=$root_path/model_hub/ts_vad
        sctk_tool_path=$ts_vad_path/SCTK-2.4.12
        rttm_name=alimeeting_eval ## offer groud truth label
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
          --sctk_tool_path ${sctk_tool_path}\
          --rttm_name $rttm_name\
          --speaker_embedding_name_dir $speaker_embedding_name_dir\
	  --speaker_embed_dim $speaker_embed_dim
#Model DER:  0.13770786100096796
#Model ACC:  0.9550995658502464
#100%|██████████| 25/25 [00:28<00:00,  1.14s/it]
## Eval for threshold 0.20: DER 9.38%, MS 0.84%, FA 8.25%, SC 0.30%
#
#Eval for threshold 0.30: DER 6.68%, MS 1.37%, FA 4.98%, SC 0.33%
#
#Eval for threshold 0.35: DER 5.93%, MS 1.67%, FA 3.90%, SC 0.36%
#
#Eval for threshold 0.40: DER 5.45%, MS 1.95%, FA 3.09%, SC 0.40%
#
#Eval for threshold 0.45: DER 5.18%, MS 2.31%, FA 2.45%, SC 0.42%
#
#Eval for threshold 0.50: DER 5.04%, MS 2.68%, FA 1.93%, SC 0.43%
#
#Eval for threshold 0.55: DER 5.05%, MS 3.18%, FA 1.46%, SC 0.41%
#
#Eval for threshold 0.60: DER 5.31%, MS 3.80%, FA 1.14%, SC 0.37%
#
#Eval for threshold 0.70: DER 6.33%, MS 5.35%, FA 0.73%, SC 0.24%
#
#Eval for threshold 0.80: DER 8.51%, MS 7.85%, FA 0.52%, SC 0.14%

fi

if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ];then
   echo "infer alimeeting eval dataset via ts_vad model trained on alimeeting train_ali_far dataset with musan and rirs noise"

        exp_name=baseline_ResNet34-TSTP-emb256-fbank80-num_frms200-aug0.6-spTrue-saFalse-ArcMargin-SGD-epoch150_exp2_on_CNCeleb1-2-LM_with_musan_rirs_exp2_debug
        rs_len=4
        segment_shift=1
        gen_subset=Test ## choice from "Eval" and "Test"
        #speech_encoder_type="ecapa_wespeaker"
        #speaker_embedding_name_dir="ecapa_tdnn_based_DINO_ft_on_CNCeleb1-2_devset-LM_feature_dir"
        speech_encoder_type="resnet34_wespeaker"
        speaker_embedding_name_dir="ResNet34-TSTP-emb256-fbank80-num_frms200-aug0.6-spTrue-saFalse-ArcMargin-SGD-epoch150_exp2_on_CNCeleb1-2-LM_feature_dir_debug"
        speaker_embed_dim=256
        root_path=/home/maduo
        data_path=/data/alimeeting/

        exp_dir=${root_path}/exp/speaker_diarization/ts_vad
        ts_vad_path=${root_path}/codebase/fairseq_speechtext/examples/speaker_diarization
        spk_path=${root_path}/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding
        results_path=${exp_dir}/${exp_name}/inf/${gen_subset}

        rttm_dir=$root_path/model_hub/ts_vad
        sctk_tool_path=$ts_vad_path/SCTK-2.4.12
        rttm_name=alimeeting_test ## offer groud truth label
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
          --sctk_tool_path ${sctk_tool_path}\
          --rttm_name $rttm_name\
          --speaker_embedding_name_dir $speaker_embedding_name_dir\
          --speaker_embed_dim $speaker_embed_dim
fi
#Model DER:  0.14415879749166993
#Model ACC:  0.949006058772365
#100%|██████████| 60/60 [01:09<00:00,  1.16s/it]
#Eval for threshold 0.20: DER 11.29%, MS 0.98%, FA 9.62%, SC 0.69%
#
#Eval for threshold 0.30: DER 8.28%, MS 1.58%, FA 5.74%, SC 0.96%
#
#Eval for threshold 0.35: DER 7.40%, MS 1.91%, FA 4.40%, SC 1.09%
#
#Eval for threshold 0.40: DER 6.77%, MS 2.25%, FA 3.26%, SC 1.26%
#
#Eval for threshold 0.45: DER 6.46%, MS 2.70%, FA 2.39%, SC 1.36%
#
#Eval for threshold 0.50: DER 6.40%, MS 3.31%, FA 1.69%, SC 1.41%
#
#Eval for threshold 0.55: DER 6.55%, MS 4.05%, FA 1.21%, SC 1.30%
#
#Eval for threshold 0.60: DER 7.05%, MS 4.97%, FA 0.89%, SC 1.19%
#
#Eval for threshold 0.70: DER 8.67%, MS 7.32%, FA 0.45%, SC 0.90%
#
#Eval for threshold 0.80: DER 11.38%, MS 10.57%, FA 0.22%, SC 0.58%

if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ];then
   echo "infer alimeeting eval dataset via ts_vad model trained on alimeeting train_ali_far dataset with musan and rirs noise"

        exp_name=baseline_ResNet34-TSTP-emb256-fbank80-num_frms200-aug0.6-spTrue-saFalse-ArcMargin-SGD-epoch150_exp2_on_CNCeleb1-2-LM_with_musan_rirs_exp2_debug2
        rs_len=4
        segment_shift=1
        gen_subset=Eval ## choice from "Eval" and "Test"
        #speech_encoder_type="ecapa_wespeaker"
        #speaker_embedding_name_dir="ecapa_tdnn_based_DINO_ft_on_CNCeleb1-2_devset-LM_feature_dir"
        speech_encoder_type="resnet34_wespeaker"
        speaker_embedding_name_dir="ResNet34-TSTP-emb256-fbank80-num_frms200-aug0.6-spTrue-saFalse-ArcMargin-SGD-epoch150_exp2_on_CNCeleb1-2-LM_feature_dir_debug"
        speaker_embed_dim=256
        root_path=/home/maduo
        data_path=/data/alimeeting/

        exp_dir=${root_path}/exp/speaker_diarization/ts_vad
        ts_vad_path=${root_path}/codebase/fairseq_speechtext/examples/speaker_diarization
        spk_path=${root_path}/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding
        results_path=${exp_dir}/${exp_name}/inf/${gen_subset}

        rttm_dir=$root_path/model_hub/ts_vad
        sctk_tool_path=$ts_vad_path/SCTK-2.4.12
        rttm_name=alimeeting_eval ## offer groud truth label
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
          --sctk_tool_path ${sctk_tool_path}\
          --rttm_name $rttm_name\
          --speaker_embedding_name_dir $speaker_embedding_name_dir\
          --speaker_embed_dim $speaker_embed_dim
#Model DER:  0.13885086978361086
#Model ACC:  0.9547113273433837
#100%|██████████| 25/25 [00:28<00:00,  1.15s/it]
#Eval for threshold 0.20: DER 9.10%, MS 0.84%, FA 7.98%, SC 0.28%
#
#Eval for threshold 0.30: DER 6.57%, MS 1.38%, FA 4.79%, SC 0.39%
#
#Eval for threshold 0.35: DER 5.86%, MS 1.63%, FA 3.82%, SC 0.42%
#
#Eval for threshold 0.40: DER 5.44%, MS 1.92%, FA 3.07%, SC 0.45%
#
#Eval for threshold 0.45: DER 5.20%, MS 2.29%, FA 2.42%, SC 0.48%
#
#Eval for threshold 0.50: DER 5.11%, MS 2.72%, FA 1.90%, SC 0.49%
#
#Eval for threshold 0.55: DER 5.12%, MS 3.16%, FA 1.49%, SC 0.47%
#
#Eval for threshold 0.60: DER 5.34%, MS 3.78%, FA 1.12%, SC 0.43%
#
#Eval for threshold 0.70: DER 6.46%, MS 5.47%, FA 0.72%, SC 0.28%
#
#Eval for threshold 0.80: DER 8.54%, MS 7.91%, FA 0.48%, SC 0.16%
fi

if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ];then
   echo "infer alimeeting eval dataset via ts_vad model trained on alimeeting train_ali_far dataset with musan and rirs noise"

        exp_name=baseline_ecapa_tdnn_based_DINO_ft_on_CNCeleb1-2_devset-LM_with_musan_rirs_debug
        rs_len=4
        segment_shift=1
        gen_subset=Eval ## choice from "Eval" and "Test"
        root_path=/home/maduo
        data_path=/data/alimeeting/

        speech_encoder_type="ecapa_wespeaker"
        speaker_embedding_name_dir="ecapa_tdnn_based_DINO_ft_on_CNCeleb1-2_devset-LM_feature_dir_debug"
        speaker_embed_dim=192


        exp_dir=${root_path}/exp/speaker_diarization/ts_vad
        ts_vad_path=${root_path}/codebase/fairseq_speechtext/examples/speaker_diarization
        spk_path=${root_path}/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding
        results_path=${exp_dir}/${exp_name}/inf/${gen_subset}

        rttm_dir=$root_path/model_hub/ts_vad
        sctk_tool_path=$ts_vad_path/SCTK-2.4.12
        rttm_name=alimeeting_eval ## offer groud truth label
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
          --sctk_tool_path ${sctk_tool_path}\
          --rttm_name $rttm_name\
          --speaker_embedding_name_dir $speaker_embedding_name_dir\
          --speaker_embed_dim $speaker_embed_dim
#Model DER:  0.14597518900698928
#Model ACC:  0.9524256824111829
#100%|██████████| 25/25 [00:28<00:00,  1.13s/it]
#Eval for threshold 0.20: DER 9.61%, MS 1.01%, FA 8.27%, SC 0.32%
#
#Eval for threshold 0.30: DER 6.98%, MS 1.62%, FA 4.92%, SC 0.43%
#
#Eval for threshold 0.35: DER 6.29%, MS 1.92%, FA 3.89%, SC 0.47%
#
#Eval for threshold 0.40: DER 5.81%, MS 2.24%, FA 3.09%, SC 0.48%
#
#Eval for threshold 0.45: DER 5.60%, MS 2.66%, FA 2.45%, SC 0.49%
#
#Eval for threshold 0.50: DER 5.52%, MS 3.10%, FA 1.90%, SC 0.52%
#
#Eval for threshold 0.55: DER 5.61%, MS 3.64%, FA 1.48%, SC 0.49%
#
#Eval for threshold 0.60: DER 5.78%, MS 4.27%, FA 1.08%, SC 0.42%
#
#Eval for threshold 0.70: DER 6.76%, MS 5.80%, FA 0.63%, SC 0.32%
#
#Eval for threshold 0.80: DER 9.00%, MS 8.44%, FA 0.41%, SC 0.15%


fi

if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ];then
   echo "infer alimeeting test dataset via ts_vad model trained on alimeeting train_ali_far dataset with musan and rirs noise"

        exp_name=baseline_ecapa_tdnn_based_DINO_ft_on_CNCeleb1-2_devset-LM_with_musan_rirs_debug
        rs_len=4
        segment_shift=1
        gen_subset=Test ## choice from "Eval" and "Test"
        root_path=/home/maduo
        data_path=/data/alimeeting/

        speech_encoder_type="ecapa_wespeaker"
        speaker_embedding_name_dir="ecapa_tdnn_based_DINO_ft_on_CNCeleb1-2_devset-LM_feature_dir_debug"
        speaker_embed_dim=192


        exp_dir=${root_path}/exp/speaker_diarization/ts_vad
        ts_vad_path=${root_path}/codebase/fairseq_speechtext/examples/speaker_diarization
        spk_path=${root_path}/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding
        results_path=${exp_dir}/${exp_name}/inf/${gen_subset}

        rttm_dir=$root_path/model_hub/ts_vad
        sctk_tool_path=$ts_vad_path/SCTK-2.4.12
        rttm_name=alimeeting_test ## offer groud truth label
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
          --sctk_tool_path ${sctk_tool_path}\
          --rttm_name $rttm_name\
          --speaker_embedding_name_dir $speaker_embedding_name_dir\
          --speaker_embed_dim $speaker_embed_dim
# Model DER:  0.15782986208484368
#Model ACC:  0.9440446472384837
#100%|██████████| 60/60 [01:08<00:00,  1.15s/it]
#Eval for threshold 0.20: DER 12.03%, MS 1.21%, FA 9.94%, SC 0.89%
#
#Eval for threshold 0.30: DER 8.83%, MS 1.92%, FA 5.75%, SC 1.16%
#
#Eval for threshold 0.35: DER 7.99%, MS 2.36%, FA 4.33%, SC 1.30%
#
#Eval for threshold 0.40: DER 7.45%, MS 2.81%, FA 3.23%, SC 1.42%
#
#Eval for threshold 0.45: DER 7.10%, MS 3.30%, FA 2.33%, SC 1.47%
#
#Eval for threshold 0.50: DER 7.10%, MS 3.96%, FA 1.63%, SC 1.50%
#
#Eval for threshold 0.55: DER 7.36%, MS 4.77%, FA 1.18%, SC 1.40%
#
#Eval for threshold 0.60: DER 7.90%, MS 5.81%, FA 0.86%, SC 1.23%
#
#Eval for threshold 0.70: DER 9.49%, MS 8.18%, FA 0.42%, SC 0.88%
#
#Eval for threshold 0.80: DER 12.39%, MS 11.64%, FA 0.20%, SC 0.55%

fi


if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ];then
   echo "infer alimeeting eval dataset via ts_vad model trained on alimeeting train_ali_far dataset with musan and rirs noise"

        exp_name=baseline_cam++_speech_campplus_sv_zh-cn_16k-common_200k_speakers_with_musan_rirs
        rs_len=4
        segment_shift=1
        gen_subset=Eval ## choice from "Eval" and "Test"
        root_path=/home/maduo
        data_path=/data/alimeeting/

        speech_encoder_type="cam++"
        speaker_embedding_name_dir="cam++_zh_cn_16k_common_feature_dir"
        speaker_embed_dim=192

        exp_dir=${root_path}/exp/speaker_diarization/ts_vad
        ts_vad_path=${root_path}/codebase/fairseq_speechtext/examples/speaker_diarization
        spk_path=${root_path}/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding
        results_path=${exp_dir}/${exp_name}/inf/${gen_subset}

        rttm_dir=$root_path/model_hub/ts_vad
        sctk_tool_path=$ts_vad_path/SCTK-2.4.12
        rttm_name=alimeeting_eval ## offer groud truth label
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
          --sctk_tool_path ${sctk_tool_path}\
          --rttm_name $rttm_name\
          --speaker_embedding_name_dir $speaker_embedding_name_dir\
          --speaker_embed_dim $speaker_embed_dim
# Model DER:  0.13374529108617633
#Model ACC:  0.9562667646281756
#100%|██████████| 25/25 [00:27<00:00,  1.09s/it]
#Eval for threshold 0.20: DER 8.72%, MS 0.83%, FA 7.57%, SC 0.31%
#
#Eval for threshold 0.30: DER 6.34%, MS 1.36%, FA 4.60%, SC 0.38%
#
#Eval for threshold 0.35: DER 5.70%, MS 1.67%, FA 3.64%, SC 0.40%
#
#Eval for threshold 0.40: DER 5.25%, MS 1.95%, FA 2.88%, SC 0.42%
#
#Eval for threshold 0.45: DER 4.96%, MS 2.24%, FA 2.25%, SC 0.47%
#
#Eval for threshold 0.50: DER 4.90%, MS 2.65%, FA 1.81%, SC 0.44%
#
#Eval for threshold 0.55: DER 4.90%, MS 3.05%, FA 1.44%, SC 0.42%
#
#Eval for threshold 0.60: DER 5.02%, MS 3.54%, FA 1.11%, SC 0.37%
#
#Eval for threshold 0.70: DER 5.92%, MS 4.88%, FA 0.75%, SC 0.28%
#
#Eval for threshold 0.80: DER 7.48%, MS 6.83%, FA 0.48%, SC 0.17%


fi

if [ ${stage} -le 16 ] && [ ${stop_stage} -ge 16 ];then
   echo "infer alimeeting test dataset via ts_vad model trained on alimeeting train_ali_far dataset with musan and rirs noise"

        exp_name=baseline_cam++_speech_campplus_sv_zh-cn_16k-common_200k_speakers_with_musan_rirs
        rs_len=4
        segment_shift=1
        gen_subset=Test ## choice from "Eval" and "Test"
        root_path=/home/maduo
        data_path=/data/alimeeting/
	
	speech_encoder_type="cam++"
        speaker_embedding_name_dir="cam++_zh_cn_16k_common_feature_dir"
        speaker_embed_dim=192

        exp_dir=${root_path}/exp/speaker_diarization/ts_vad
        ts_vad_path=${root_path}/codebase/fairseq_speechtext/examples/speaker_diarization
        spk_path=${root_path}/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding
        results_path=${exp_dir}/${exp_name}/inf/${gen_subset}

        rttm_dir=$root_path/model_hub/ts_vad
        sctk_tool_path=$ts_vad_path/SCTK-2.4.12
        rttm_name=alimeeting_test ## offer groud truth label
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
          --sctk_tool_path ${sctk_tool_path}\
          --rttm_name $rttm_name\
          --speaker_embedding_name_dir $speaker_embedding_name_dir\
          --speaker_embed_dim $speaker_embed_dim

# Model DER:  0.13549183565984604
#Model ACC:  0.9532326285518569
#100%|██████████| 60/60 [01:08<00:00,  1.14s/it]
#Eval for threshold 0.20: DER 9.07%, MS 1.07%, FA 7.54%, SC 0.47%
#
#Eval for threshold 0.30: DER 6.80%, MS 1.68%, FA 4.54%, SC 0.59%
#
#Eval for threshold 0.35: DER 6.19%, MS 2.01%, FA 3.53%, SC 0.64%
#
#Eval for threshold 0.40: DER 5.85%, MS 2.39%, FA 2.75%, SC 0.71%
#
#Eval for threshold 0.45: DER 5.68%, MS 2.80%, FA 2.13%, SC 0.76%
#
#Eval for threshold 0.50: DER 5.66%, MS 3.26%, FA 1.64%, SC 0.76%
#
#Eval for threshold 0.55: DER 5.76%, MS 3.84%, FA 1.21%, SC 0.71%
#
#Eval for threshold 0.60: DER 6.07%, MS 4.54%, FA 0.91%, SC 0.62%
#
#Eval for threshold 0.70: DER 7.15%, MS 6.18%, FA 0.47%, SC 0.49%
#
#Eval for threshold 0.80: DER 9.11%, MS 8.57%, FA 0.22%, SC 0.33%

fi

#
if [ ${stage} -le 17 ] && [ ${stop_stage} -ge 17 ];then
   
   echo "Do model average ..."
   root_path=/home/maduo
   exp_dir=${root_path}/exp/speaker_diarization/ts_vad
   exp_name=baseline_camppluse_zh_en_common_advanced_with_musan_rirs
   avg_model=$exp_dir/$exp_name/checkpoints/avg_model.pt
   num_avg=5
   python ts_vad/average_model.py \
	    --dst_model $avg_model \
            --src_path $exp_dir/$exp_name/checkpoints \
            --num ${num_avg}


fi
