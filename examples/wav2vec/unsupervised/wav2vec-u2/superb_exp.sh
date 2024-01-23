#!/usr/bin/env bash

stage=0
stop_stage=1000

#. path_for_fsq_speechtext.sh # py3.9 cuda11.8 pytorch=2.1.1 torchaudio=2.1.1 without s3prl
. path_for_fsq_sptt.sh ## py3.9 cuda11.8 pytorch=2.1.1 torchaudio=2.1.1 , s3prl, hydra=2.3.1
. utils/parse_options.sh



if [ ${stage} -le -1 ]&& [ ${stop_stage} -ge -1 ];then
   echo "covert fairseq model to s3prl format"
   pretrain_model=exp/pretrain/pretrain_on_base_imls-ssl_4gpu_8update_960h_400k_update/checkpoint_298_400000.pt
   dest_dir=exp/base_imls_ssl_downstream_task
   python  s3prl/s3prl/upstream/hubert/convert.py\
             --fairseq_ckpt $pretrain_model\
             --output_dir $dest_dir       


fi


## speaker related task: SID(Speaker Identification), 
##                       ASV(Automatic Speaker Verification)
##                       SD(Speaker Diarization)
if [ ${stage} -le 0 ]&& [ ${stop_stage} -ge 0 ];then
   echo "SID(Speaker Identification)"
   #python3 run_downstream.py -n ExpName -m train -u fbank -d voxceleb1
   exp_dir=exp/base_imls_ssl_downstream_task/sid ## output
   mkdir -p $exp_dir
   pretrain_model=exp/base_imls_ssl_downstream_task/checkpoint_298_400000.pt
   num_gpus=1
   downstream_task_config=s3prl/s3prl/downstream/voxceleb1/config.yaml
   CUDA_VISIBLE_DEVICES=1 torchrun \
          --nproc_per_node $num_gpus \
          --master_port=34567 \
          s3prl/s3prl/run_downstream.py\
           -m train \
           -u imls_ssl_local\
           -k $pretrain_model\
           -d voxceleb1\
           --config $downstream_task_config\
           -o config.runner.gradient_accumulate_steps=4\
           --auto_resume \
           --expdir $exp_dir

fi

if [ ${stage} -le 1 ]&& [ ${stop_stage} -ge 1 ];then
   exp_dir=exp/base_imls_ssl_downstream_task/sid  ## output 
   #pretrain_model=$exp_dir/checkpoint_289_400000.pt
   num_gpus=1
   downstream_task_config=s3prl/s3prl/downstream/voxceleb1/config.yaml
   CUDA_VISIBLE_DEVICES=1 torchrun --master_port=34567  \
      --nproc_per_node $num_gpus \
      s3prl/s3prl/run_downstream.py\
       -m evaluate \
       --config $downstream_task_config\
       -e $exp_dir/dev-best.ckpt
  ## train acc: 0.9403749704360962
  ## start evaluate: in train func
  ## dev acc: 0.6426709294319153
  ## start evaluate: in train func
  ## test acc: 0.6164101362228394
  
  ## test acc: 0.6164101362228394
fi


if [ ${stage} -le 2 ]&& [ ${stop_stage} -ge 2 ];then
   echo "ASV: Automatic Speaker Verification"
   #python3 run_downstream.py -n ExpName -m train -u fbank -d voxceleb1
   exp_dir=exp/base_imls_ssl_downstream_task/asv ## output
   mkdir -p $exp_dir
   pretrain_model=exp/base_imls_ssl_downstream_task/checkpoint_298_400000.pt
   num_gpus=1
   downstream_task_config=s3prl/s3prl/downstream/sv_voxceleb1/config.yaml
   CUDA_VISIBLE_DEVICES=4 torchrun \
          --nproc_per_node $num_gpus \
          --master_port=25678 \
          s3prl/s3prl/run_downstream.py\
           -m train \
           -u imls_ssl_local\
           -k $pretrain_model\
           -d sv_voxceleb1\
           --config $downstream_task_config\
           -o config.runner.gradient_accumulate_steps=4\
           --auto_resume \
           --expdir $exp_dir

fi

if [ ${stage} -le 3 ]&& [ ${stop_stage} -ge 3 ];then
   exp_dir=exp/base_imls_ssl_downstream_task/asv  ## output 
   ## there is no official validation set under VoxCeleb1 setting, we save checkpoints every 20000 updates and report the best EER.
   num_gpus=1
   downstream_task_config=s3prl/s3prl/downstream/sv_voxceleb1/config.yaml
   CUDA_VISIBLE_DEVICES=4 \
    torchrun  --master_port=25678 \
      --nproc_per_node $num_gpus \
      s3prl/s3prl/run_downstream.py\
       -m evaluate \
       --config $downstream_task_config\
       -e $exp_dir/states-20000.ckpt
   ## sv-voxceleb1/test-EER: 0.07396606574760925
  

fi
if [ ${stage} -le 4 ]&& [ ${stop_stage} -ge 4 ];then
   voxceleb1=dataset/downstreams_tasks/Speaker_Identification/VoxCeleb1 #"root directory of VoxCeleb1"
   exp_dir=exp/base_imls_ssl_downstream_task/asv  ## output 
   bash s3prl/s3prl/downstream/sv_voxceleb1/test_expdir.sh\
           $exp_dir $voxceleb1

 ## cat logs/superb_exp_ASV_infer_stage4.log
 ## Report the testing results...
 ## exp/base_imls_ssl_downstream_task/asv/states-20000/log.txt:sv-voxceleb1/test-EER: 0.07396606574760925
 ## exp/base_imls_ssl_downstream_task/asv/states-40000/log.txt:sv-voxceleb1/test-EER: 0.05896076352067579
 ## exp/base_imls_ssl_downstream_task/asv/states-60000/log.txt:sv-voxceleb1/test-EER: 0.05625662778222375
 ## exp/base_imls_ssl_downstream_task/asv/states-160000/log.txt:sv-voxceleb1/test-EER: 0.05599151643690351
 ## exp/base_imls_ssl_downstream_task/asv/states-120000/log.txt:sv-voxceleb1/test-EER: 0.05577942735948751
 ## exp/base_imls_ssl_downstream_task/asv/states-140000/log.txt:sv-voxceleb1/test-EER: 0.054400848356309824
 ## exp/base_imls_ssl_downstream_task/asv/states-200000/log.txt:sv-voxceleb1/test-EER: 0.05429480381760797
 ## exp/base_imls_ssl_downstream_task/asv/states-80000/log.txt:sv-voxceleb1/test-EER: 0.05387062566364719
 ## exp/base_imls_ssl_downstream_task/asv/states-180000/log.txt:sv-voxceleb1/test-EER: 0.05381760339342527
 ##exp/base_imls_ssl_downstream_task/asv/states-100000/log.txt:sv-voxceleb1/test-EER: 0.053658536585366436

 ##10 checkpoints evaluated.
 ##The best checkpoint achieves EER 0.053658536585366436

 ##Prepare prediction file for submission...
 ##The best prediction file has been prepared
 ##/workspace2/maduo/exp/base_imls_ssl_downstream_task/asv/states-100000/test_predict.txt -> /workspace2/maduo/exp/base_imls_ssl_downstream_task/asv/test_predict.txt
fi

if [ ${stage} -le 5 ]&& [ ${stop_stage} -ge 5 ];then
   echo "SD: Speaker Diarization"
   exp_dir=exp/base_imls_ssl_downstream_task/sd ## output
   mkdir -p $exp_dir
   pretrain_model=exp/base_imls_ssl_downstream_task/checkpoint_298_400000.pt
   num_gpus=1
   downstream_task_config=s3prl/s3prl/downstream/diarization/config.yaml 
   CUDA_VISIBLE_DEVICES=6 torchrun  --master_port=12345 \
          --nproc_per_node $num_gpus \
          s3prl/s3prl/run_downstream.py\
           -m train \
           -u imls_ssl_local\
           -k $pretrain_model\
           -d diarization \
           --config $downstream_task_config\
           -o config.runner.gradient_accumulate_steps=4\
           -o config.downstream_expert.datarc.frame_shift=160\
           --auto_resume \
           --expdir $exp_dir


fi

if [ ${stage} -le 6 ]&& [ ${stop_stage} -ge 6 ];then
   exp_dir=exp/base_imls_ssl_downstream_task/sd  ## output 
   #pretrain_model=$exp_dir/checkpoint_289_400000.pt
   num_gpus=1
   downstream_task_config=s3prl/s3prl/downstream/diarization/config.yaml
   CUDA_VISIBLE_DEVICES=6 \
   torchrun  --master_port=12345 \
      --nproc_per_node $num_gpus \
      s3prl/s3prl/run_downstream.py\
       -m evaluate \
       --config $downstream_task_config\
       -e $exp_dir/best-states-dev.ckpt
   
   ## result: cat logs/superb_exp_SD.log
   # mode test acc 0.951992928981781 der 0.06429737061262131

fi

# PR ASR OOD-ASR KS QbE
if [ ${stage} -le 7 ]&& [ ${stop_stage} -ge 7 ];then
    echo "PR: Phoneme Recognition"
    
   exp_dir=exp/base_imls_ssl_downstream_task/pr ## output
   mkdir -p $exp_dir
   pretrain_model=exp/base_imls_ssl_downstream_task/checkpoint_298_400000.pt
   num_gpus=1
   downstream_task_config=s3prl/s3prl/downstream/ctc/libriphone.yaml
   CUDA_VISIBLE_DEVICES=6 torchrun  --master_port=12358  \
          --nproc_per_node $num_gpus \
          s3prl/s3prl/run_downstream.py\
           -m train \
           -u imls_ssl_local\
           -k $pretrain_model\
           -d ctc \
           --config $downstream_task_config\
           -o config.runner.gradient_accumulate_steps=4\
           --auto_resume \
           --expdir $exp_dir
fi
if [ ${stage} -le 8 ]&& [ ${stop_stage} -ge 8 ];then
   exp_dir=exp/base_imls_ssl_downstream_task/pr  ## output
   #pretrain_model=$exp_dir/checkpoint_289_400000.pt
   num_gpus=1
   downstream_task_config=s3prl/s3prl/downstream/ctc/libriphone.yaml
   CUDA_VISIBLE_DEVICES=6 torchrun  --master_port=12358  \
      --nproc_per_node $num_gpus \
      s3prl/s3prl/run_downstream.py\
       -m evaluate \
       --config $downstream_task_config\
       -e $exp_dir/dev-best.ckpt   
# test loss: 0.1890551894903183
# test per: 0.03869764178771306
fi


#(TODO) we stop the task, later we run it
## OOD-ASR contain four language . they are e Mexican Spanish (es), Mandarin (zh), and Arabic (ar) subsets from CommonVoice 7.0 and the spontaneous speech
## task (spon), we use the Santa Barbara Corpus of Spoken American English (SBCSAE)
## so final submit WER ia average four testsets WER.  
if [ ${stage} -le 9 ]&& [ ${stop_stage} -ge 9 ];then
    echo "OOD-ASR:Out-of-domain Automatic Speech Recognition Tasks"
   #exp_dir=exp/base_imls_ssl_downstream_task/ood_asr ## output
   #mkdir -p $exp_dir
   #pretrain_model=exp/base_imls_ssl_downstream_task/checkpoint_298_400000.pt
   pretain_model=exp/pretrain/continue_pretain_base_hubert_on_train_360/backup/checkpoint_96_50000.pt
   upstream_model_name=hubert
   num_gpus=1
   for lang in es zh ar;do
    exp_dir=exp/base_imls_ssl_downstream_task/ood_asr_$lang ## output
    mkdir -p $exp_dir
    downstream_task_config=s3prl/s3prl/downstream/ctc/cv_config/cv_${lang}.yaml
    CUDA_VISIBLE_DEVICES=5  torchrun  --master_port=12458  \
          --nproc_per_node $num_gpus \
          s3prl/s3prl/run_downstream.py\
           -m train \
           -u $upstream_model_name\
           -k $pretrain_model\
           -d ctc \
           --config $downstream_task_config\
           -o config.runner.gradient_accumulate_steps=4\
           --auto_resume \
           --expdir $exp_dir
   done
fi


if [ ${stage} -le 10 ]&& [ ${stop_stage} -ge 10 ];then
   #exp_dir=exp/base_imls_ssl_downstream_task/ood_asr  ## output
   #pretrain_model=$exp_dir/checkpoint_289_400000.pt
   num_gpus=1
   for lang in es zh ar;do
    exp_dir=exp/base_imls_ssl_downstream_task/ood_asr_$lang ## output
    downstream_task_config=s3prl/s3prl/downstream/ctc/cv_config/cv_${lang}.yaml
    CUDA_VISIBLE_DEVICES=5 torchrun  --master_port=12458  \
      --nproc_per_node $num_gpus \
      s3prl/s3prl/run_downstream.py\
       -m evaluate \
       --config $downstream_task_config\
       -e $exp_dir/dev-best.ckpt
   done
fi

if [ ${stage} -le 11 ]&& [ ${stop_stage} -ge 11 ];then
    echo "OOD-ASR:Out-of-domain Automatic Speech Recognition Tasks, Spontaneous Speech"
   #exp_dir=exp/base_imls_ssl_downstream_task/ood_asr ## output
   #mkdir -p $exp_dir
   pretrain_model=exp/base_imls_ssl_downstream_task/checkpoint_298_400000.pt
   num_gpus=1
   for lang in sbcsae;do 
    exp_dir=exp/base_imls_ssl_downstream_task/ood_asr_$lang ## output
    mkdir -p $exp_dir
    downstream_task_config=s3prl/s3prl/downstream/ctc/${lang}.yaml
    CUDA_VISIBLE_DEVICES=7  torchrun  --master_port=12468  \
          --nproc_per_node $num_gpus \
          s3prl/s3prl/run_downstream.py\
           -m train \
           -u imls_ssl_local\
           -k $pretrain_model\
           -d ctc \
           --config $downstream_task_config\
           -o config.runner.gradient_accumulate_steps=4\
           --auto_resume \
           --expdir $exp_dir
   done
fi



if [ ${stage} -le 12 ]&& [ ${stop_stage} -ge 12 ];then
   #exp_dir=exp/base_imls_ssl_downstream_task/ood_asr  ## output
   #pretrain_model=$exp_dir/checkpoint_289_400000.pt
   num_gpus=1
   for lang in sbcsae;do
    exp_dir=exp/base_imls_ssl_downstream_task/ood_asr_$lang ## output
    downstream_task_config=s3prl/s3prl/downstream/ctc/${lang}.yaml
    CUDA_VISIBLE_DEVICES=7 torchrun  --master_port=12468  \
      --nproc_per_node $num_gpus \
      s3prl/s3prl/run_downstream.py\
       -m evaluate \
       --config $downstream_task_config\
       -e $exp_dir/dev-best.ckpt
   done
fi



## KS
if [ ${stage} -le 15 ]&& [ ${stop_stage} -ge 15 ];then
   echo "KS"
   #s3prl/s3prl/downstream/speech_commands/config.yaml   
   exp_dir=exp/base_imls_ssl_downstream_task/ks ## output
   mkdir -p $exp_dir
   pretrain_model=exp/base_imls_ssl_downstream_task/checkpoint_298_400000.pt
   num_gpus=1
   downstream_task_config=s3prl/s3prl/downstream/speech_commands/config.yaml
   CUDA_VISIBLE_DEVICES=6  torchrun  --master_port=12358   \
          --nproc_per_node $num_gpus \
          s3prl/s3prl/run_downstream.py\
           -m train \
           -u imls_ssl_local\
           -k $pretrain_model\
           -d speech_commands \
           --config $downstream_task_config\
           -o config.runner.gradient_accumulate_steps=4\
           --auto_resume \
           --expdir $exp_dir

fi


if [ ${stage} -le 16 ]&& [ ${stop_stage} -ge 16 ];then
   exp_dir=exp/base_imls_ssl_downstream_task/ks  ## output
   #pretrain_model=$exp_dir/checkpoint_289_400000.pt
   num_gpus=1
   downstream_task_config=s3prl/s3prl/downstream/speech_commands/config.yaml
   CUDA_VISIBLE_DEVICES=6  torchrun  --master_port=12358 \
      --nproc_per_node $num_gpus \
      s3prl/s3prl/run_downstream.py\
       -m evaluate \
       --config $downstream_task_config\
       -e $exp_dir/dev-best.ckpt
   ##  cat logs/superb_exp_KS.log
   ## test acc: 0.9626744563453424
fi

## QbE: Query-by-Example Spoken Term Detection
if [ ${stage} -le 17 ]&& [ ${stop_stage} -ge 17 ];then
    #s3prl/s3prl/downstream/quesst14_dtw/config.yaml
   echo "QbE: Query-by-Example Spoken Term Detection in dev"
   dist_fn=cosine
   for layer in 0 1 2 3 4 5 6 7 8 9 10 11;do
    exp_dir=exp/base_imls_ssl_downstream_task/qbe_layer_${layer}_dev ## output
    mkdir -p $exp_dir
    pretrain_model=exp/base_imls_ssl_downstream_task/checkpoint_298_400000.pt
    num_gpus=1
    downstream_task_config=s3prl/s3prl/downstream/quesst14_dtw/config.yaml
    CUDA_VISIBLE_DEVICES=3   torchrun  --master_port=12368   \
          --nproc_per_node $num_gpus \
          s3prl/s3prl/run_downstream.py\
           -m evaluate -t "dev" \
           -u imls_ssl_local\
           -l $layer\
           -k $pretrain_model\
           -d quesst14_dtw \
           --config $downstream_task_config\
           -o config.runner.gradient_accumulate_steps=4\
           -o config.downstream_expert.dtwrc.dist_method=$dist_fn\
           --auto_resume \
           --expdir $exp_dir   
   

  done 
fi
if [ ${stage} -le 18 ]&& [ ${stop_stage} -ge 18 ];then
    #s3prl/s3prl/downstream/quesst14_dtw/config.yaml
   echo "QbE: Query-by-Example Spoken Term Detection in test"
  dist_fn=cosine
  #for layer in 0 1 2 3 4 5 6 7 8 9 10 11;do
  for layer in 9 10 11;do
    exp_dir=exp/base_imls_ssl_downstream_task/qbe_layer_${layer}_test ## output
    mkdir -p $exp_dir
    pretrain_model=exp/base_imls_ssl_downstream_task/checkpoint_298_400000.pt
    num_gpus=1
    downstream_task_config=s3prl/s3prl/downstream/quesst14_dtw/config.yaml
    CUDA_VISIBLE_DEVICES=3 torchrun  --master_port=12368   \
          --nproc_per_node $num_gpus \
          s3prl/s3prl/run_downstream.py\
           -m evaluate -t "test" \
           -u imls_ssl_local\
           -l $layer\
           -k $pretrain_model\
           -d quesst14_dtw \
           --config $downstream_task_config\
           -o config.runner.gradient_accumulate_steps=4\
           -o config.downstream_expert.dtwrc.dist_method=$dist_fn\
           --auto_resume \
           --expdir $exp_dir


  done
fi

if [ ${stage} -le 19 ]&& [ ${stop_stage} -ge 19 ];then
 
 for layer in 0 1 2 3 4 5 6 7 8 9 10 11;do
  # dev
  exp_dir=exp/base_imls_ssl_downstream_task/qbe_layer_${layer}_dev
  ref_dev=dataset/downstreams_tasks/Query_by_Example_Spoken_Term_Detection/quesst14Database/quesst14Database/scoring/groundtruth_quesst14_dev
  bash dataset/downstreams_tasks/Query_by_Example_Spoken_Term_Detection/quesst14Database/quesst14Database/scoring/score-TWV-Cnxe.sh\
        $exp_dir $ref_dev -10
 done
  # cat  logs/superb_exp_QbE_test_part_2.log 
  # best maxTWX:  at qbe_layer_7_dev, it should be 8-th layer
  #actTWV: 0.059187762  maxTWV: 0.059875514  Threshold: 2.0188
  #actCnxe: 3.1934202  minCnxe: 0.9719019
 for layer in 0 1 2 3 4 5 6 7 8 9 10 11;do
  # test
  exp_dir=exp/base_imls_ssl_downstream_task/qbe_layer_${layer}_test
  ref_eval=dataset/downstreams_tasks/Query_by_Example_Spoken_Term_Detection/quesst14Database/quesst14Database/scoring/groundtruth_quesst14_eval
  bash dataset/downstreams_tasks/Query_by_Example_Spoken_Term_Detection/quesst14Database/quesst14Database/scoring/score-TWV-Cnxe.sh\
    $exp_dir  $ref_eval -10
 done

 # # cat  logs/superb_exp_QbE_test_part_2.log
 #actTWV: 0.07876372  maxTWV: 0.07993908  Threshold: 1.9302
 #actCnxe: 3.1753254  minCnxe: 0.9685478
fi


## IC: Intent Classification
if [ ${stage} -le 20 ] && [ ${stop_stage} -ge 20 ];then
   exp_dir=exp/base_imls_ssl_downstream_task/ic ## output
   mkdir -p $exp_dir
   pretrain_model=exp/base_imls_ssl_downstream_task/checkpoint_298_400000.pt
   num_gpus=1
   downstream_task_config=s3prl/s3prl/downstream/fluent_commands/config.yaml
   CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --use_env \
          --nproc_per_node $num_gpus \
          s3prl/s3prl/run_downstream.py\
           -m train \
           -u imls_ssl_local\
           -k $pretrain_model\
           -d fluent_commands\
           --config $downstream_task_config\
           -o config.runner.gradient_accumulate_steps=4\
           --auto_resume \
           --expdir $exp_dir
fi


if [ ${stage} -le 21 ] && [ ${stop_stage} -ge 21 ];then
   echo "evalute test type"
   exp_dir=exp/base_imls_ssl_downstream_task/ic ## output
   #pretrain_model=$exp_dir/checkpoint_289_400000.pt
   num_gpus=1
   downstream_task_config=s3prl/s3prl/downstream/fluent_commands/config.yaml
   CUDA_VISIBLE_DEVICES=4 \
   python -m torch.distributed.launch \
      --use_env \
      --nproc_per_node $num_gpus \
      s3prl/s3prl/run_downstream.py\
       -m evaluate \
       --config $downstream_task_config\
       -e $exp_dir/dev-best.ckpt
# test acc:
fi

# SF: End-to-end Slot Filling
if [ ${stage} -le 22 ] && [ ${stop_stage} -ge 22 ];then
   echo "SF: End-to-end Slot Filling"

   exp_dir=exp/base_imls_ssl_downstream_task/sf ## output
   mkdir -p $exp_dir
   pretrain_model=exp/base_imls_ssl_downstream_task/checkpoint_298_400000.pt
   num_gpus=1
   downstream_task_config=s3prl/s3prl/downstream/ctc/snips.yaml
   CUDA_VISIBLE_DEVICES=2  torchrun  --master_port=12378 \
          --nproc_per_node $num_gpus \
          s3prl/s3prl/run_downstream.py\
           -m train \
           -u imls_ssl_local\
           -k $pretrain_model\
           -d ctc \
           --config $downstream_task_config\
           -o config.runner.gradient_accumulate_steps=4\
           --auto_resume \
           --expdir $exp_dir
fi
if [ ${stage} -le 23 ]&& [ ${stop_stage} -ge 23 ];then
   exp_dir=exp/base_imls_ssl_downstream_task/sf  ## output
   #pretrain_model=$exp_dir/checkpoint_289_400000.pt
   num_gpus=1
   downstream_task_config=s3prl/s3prl/downstream/ctc/snips.yaml
   CUDA_VISIBLE_DEVICES=2  torchrun  --master_port=12378 \
      --nproc_per_node $num_gpus \
      s3prl/s3prl/run_downstream.py\
       -m evaluate \
       --config $downstream_task_config\
       -e $exp_dir/dev-best.ckpt

	#test loss: 0.4978642463684082
	#test slot_type_f1: 0.8927435580292717
	#test slot_value_cer: 0.22717633336867776
	#test slot_value_wer: 0.3584683357879234
	#test slot_edit_f1_full: 66.12454212454213
	#test slot_edit_f1_part: 66.92866676553463
	#test wer: 0.14525291444378582
	#test cer: 0.07371506671941877
     # submit: slot_type_f1: 0.8927435580292717, slot_value_cer: 0.22717633336867776
fi
### ST: Speech Translation
if [ ${stage} -le 24 ] && [ ${stop_stage} -ge 24 ];then
   exp_dir=exp/base_imls_ssl_downstream_task/st ## output
   pretrain_model=exp/base_imls_ssl_downstream_task/checkpoint_298_400000.pt
   num_gpus=1
   downstream_task_config=s3prl/s3prl/downstream/speech_translation/config.yaml
   CUDA_VISIBLE_DEVICES=1 torchrun  --master_port=12388 \
      --nproc_per_node $num_gpus \
         s3prl/s3prl/run_downstream.py\
           -m train \
           -u imls_ssl_local\
           -k $pretrain_model\
           -d speech_translation\
           --config $downstream_task_config\
           -o config.runner.gradient_accumulate_steps=4\
           --auto_resume \
           --expdir $exp_dir
fi

if [ ${stage} -le 25 ] && [ ${stop_stage} -ge 25 ];then
   echo "evalute test type" ## I have change sf to st in output dir.
   exp_dir=exp/base_imls_ssl_downstream_task/st ## output
   #pretrain_model=$exp_dir/checkpoint_289_400000.pt
   num_gpus=1
   downstream_task_config=s3prl/s3prl/downstream/speech_translation/config.yaml
   CUDA_VISIBLE_DEVICES=1 torchrun  --master_port=12388 \
      --nproc_per_node $num_gpus \
      s3prl/s3prl/run_downstream.py\
       -m evaluate \
       --config $downstream_task_config\
       -e $exp_dir/dev-best.ckpt
# cat logs/superb_exp_ST.log
# BLEU = 15.03 47.1/20.3/10.8/6.0 (BP = 0.953 ratio = 0.954 hyp_len = 157202 ref_len = 164820) 
fi

# ER: Emotion Recognition
if [ ${stage} -le 26 ] && [ ${stop_stage} -ge 26 ];then
   for test_fold in Session1  Session2  Session3  Session4  Session5;do
    # The default config is "downstream/emotion/config.yaml"
    exp_dir=exp/base_imls_ssl_downstream_task/er_$test_fold ## output
    pretrain_model=exp/base_imls_ssl_downstream_task/checkpoint_298_400000.pt
    num_gpus=1
    downstream_task_config=s3prl/s3prl/downstream/emotion/config.yaml
    CUDA_VISIBLE_DEVICES=0 torchrun  --master_port=12398 \
      --nproc_per_node $num_gpus \
          s3prl/s3prl/run_downstream.py\
          --expdir $exp_dir\
          -m train \
          -u imls_ssl_local\
          -k $pretrain_model\
          -d emotion \
          --config $downstream_task_config\
          -o config.downstream_expert.datarc.test_fold=$test_fold\
          --auto_resume 
   done

fi

if [ ${stage} -le 27 ] && [ ${stop_stage} -ge 27 ];then
  for test_fold in Session1  Session2  Session3  Session4  Session5;do
   exp_dir=exp/base_imls_ssl_downstream_task/er_$test_fold ## output
   #pretrain_model=$exp_dir/checkpoint_289_400000.pt
   num_gpus=1
   downstream_task_config=s3prl/s3prl/downstream/emotion/config.yaml
   CUDA_VISIBLE_DEVICES=0 torchrun  --master_port=12398 \
      --nproc_per_node $num_gpus \
      s3prl/s3prl/run_downstream.py\
       -m evaluate \
       --config $downstream_task_config\
       -e $exp_dir/dev-best.ckpt
  done
  ##   cat logs/superb_exp_ER.log
  ##  test_fold：Session1: test acc: 0.652534544467926
  #   test_fold：Session2: test acc: 0.6735092997550964
  ##  test_fold：Session3: test acc: 0.6281494498252869
  ##  test_fold：Session4: test acc: 0.6692531704902649
  ##  test_fold：Session4: test acc: 0.6406124234199524 
  ## so 5 testing scores will be averaged, final test acc: 0.6528 
fi


#ASR
if [ ${stage} -le 28 ]&& [ ${stop_stage} -ge 28 ];then
   echo "ASR"
   #python3 run_downstream.py -n ExpName -m train -u fbank -d voxceleb1
   exp_dir=exp/base_imls_ssl_downstream_task/asr ## output
   mkdir -p $exp_dir
   pretrain_model=exp/base_imls_ssl_downstream_task/checkpoint_298_400000.pt
   num_gpus=1
   downstream_task_config=s3prl/s3prl/downstream/asr/config.yaml
   CUDA_VISIBLE_DEVICES=4 torchrun \
          --nproc_per_node $num_gpus \
          --master_port=25678 \
          s3prl/s3prl/run_downstream.py\
           -m train \
           -u imls_ssl_local\
           -k $pretrain_model\
           -d asr\
           --config $downstream_task_config\
           -o config.runner.gradient_accumulate_steps=4\
           --auto_resume \
           --expdir $exp_dir

fi

if [ ${stage} -le 29 ]&& [ ${stop_stage} -ge 29 ];then
   exp_dir=exp/base_imls_ssl_downstream_task/asr  ## output
   #pretrain_model=$exp_dir/checkpoint_289_400000.pt
   num_gpus=1
   downstream_task_config=s3prl/s3prl/downstream/asr/config.yaml
   CUDA_VISIBLE_DEVICES=4 torchrun  --master_port=25678  \
      --nproc_per_node $num_gpus \
      s3prl/s3prl/run_downstream.py\
       -m evaluate \
       -t "test-clean"\
       --config $downstream_task_config\
       -e $exp_dir/dev-clean-best.ckpt

   # test-clean loss: 0.16249065101146698
   # test-clean uer: 1.449234559211684
   # test-clean wer: 5.1183049300060866
fi

#ASR mulit -gpu
if [ ${stage} -le 30 ]&& [ ${stop_stage} -ge 30 ];then
   echo "ASR"
   #python3 run_downstream.py -n ExpName -m train -u fbank -d voxceleb1
   exp_dir=exp/base_imls_ssl_downstream_task/asr_mutil_gpus ## output
   mkdir -p $exp_dir
   pretrain_model=exp/base_imls_ssl_downstream_task/checkpoint_298_400000.pt
   num_gpus=2
   downstream_task_config=s3prl/s3prl/downstream/asr/config.yaml
   CUDA_VISIBLE_DEVICES=0,2 torchrun \
          --nproc_per_node $num_gpus \
          --master_port=25778 \
          s3prl/s3prl/run_downstream_offical.py\
           -m train \
           -u imls_ssl_local\
           -k $pretrain_model\
           -d asr\
           --config $downstream_task_config\
           -o config.runner.gradient_accumulate_steps=1\
           --auto_resume \
           --expdir $exp_dir

fi

if [ ${stage} -le 31 ]&& [ ${stop_stage} -ge 31 ];then
   exp_dir=exp/base_imls_ssl_downstream_task/asr_mutil_gpus   ## output
   #pretrain_model=$exp_dir/checkpoint_289_400000.pt
   num_gpus=1
   downstream_task_config=s3prl/s3prl/downstream/asr/config.yaml
   CUDA_VISIBLE_DEVICES=1 torchrun  --master_port=25778  \
      --nproc_per_node $num_gpus \
      s3prl/s3prl/run_downstream.py\
       -m evaluate \
       --config $downstream_task_config\
       -e $exp_dir/dev-clean-best.ckpt
fi





#### the above experiment is same as the above,
### However, I follows  superb task hyparameter of wavlm paper
if [ ${stage} -le 40 ]&& [ ${stop_stage} -ge 40 ];then
   echo "SID(Speaker Identification)"
   #python3 run_downstream.py -n ExpName -m train -u fbank -d voxceleb1
   exp_dir=exp/base_imls_ssl_downstream_task/sid_md ## output
   mkdir -p $exp_dir
   pretrain_model=exp/base_imls_ssl_downstream_task/checkpoint_298_400000.pt
   num_gpus=2
   downstream_task_config=s3prl/s3prl/downstream/voxceleb1/config_md.yaml
   CUDA_VISIBLE_DEVICES=0,3 torchrun \
          --nproc_per_node $num_gpus \
          --master_port=34577 \
          s3prl/s3prl/run_downstream_offical.py\
           -m train \
           -u imls_ssl_local\
           -k $pretrain_model\
           -d voxceleb1\
           --config $downstream_task_config\
           --auto_resume \
           --expdir $exp_dir

fi

if [ ${stage} -le 41 ]&& [ ${stop_stage} -ge 41 ];then
   exp_dir=exp/base_imls_ssl_downstream_task/sid  ## output
   #pretrain_model=$exp_dir/checkpoint_289_400000.pt
   num_gpus=1
   downstream_task_config=s3prl/s3prl/downstream/voxceleb1/config_md.yaml
   CUDA_VISIBLE_DEVICES=0 torchrun --master_port=34577  \
      --nproc_per_node $num_gpus \
      s3prl/s3prl/run_downstream_offical.py\
       -m evaluate \
       --config $downstream_task_config\
       -e $exp_dir/dev-best.ckpt
  
fi


if [ ${stage} -le 42 ]&& [ ${stop_stage} -ge 42 ];then
   echo "ASV: Automatic Speaker Verification"
   #python3 run_downstream.py -n ExpName -m train -u fbank -d voxceleb1
   exp_dir=exp/base_imls_ssl_downstream_task/asv_md ## output
   mkdir -p $exp_dir
   pretrain_model=exp/base_imls_ssl_downstream_task/checkpoint_298_400000.pt
   num_gpus=2
   downstream_task_config=s3prl/s3prl/downstream/sv_voxceleb1/config_md.yaml
   CUDA_VISIBLE_DEVICES=6,7 torchrun \
          --nproc_per_node $num_gpus \
          --master_port=34587 \
          s3prl/s3prl/run_downstream_offical.py\
           -m train \
           -u imls_ssl_local\
           -k $pretrain_model\
           -d sv_voxceleb1\
           --config $downstream_task_config\
           --auto_resume \
           --expdir $exp_dir

fi

if [ ${stage} -le 43 ]&& [ ${stop_stage} -ge 43 ];then
   exp_dir=exp/base_imls_ssl_downstream_task/asv_md  ## output
   ## there is no official validation set under VoxCeleb1 setting, we save checkpoints every 20000 updates and report the best EER.
   num_gpus=1
   downstream_task_config=s3prl/s3prl/downstream/sv_voxceleb1/config_md.yaml
   CUDA_VISIBLE_DEVICES=6 \
    torchrun  --master_port=34587 \
      --nproc_per_node $num_gpus \
      s3prl/s3prl/run_downstream_offical.py\
       -m evaluate \
       --config $downstream_task_config\
       -e $exp_dir/states-20000.ckpt
   ## sv-voxceleb1/test-EER: 0.07396606574760925


fi
if [ ${stage} -le 44 ]&& [ ${stop_stage} -ge 44 ];then
   voxceleb1=dataset/downstreams_tasks/Speaker_Identification/VoxCeleb1 #"root directory of VoxCeleb1"
   exp_dir=exp/base_imls_ssl_downstream_task/asv_md  ## output
   bash s3prl/s3prl/downstream/sv_voxceleb1/test_expdir.sh\
           $exp_dir $voxceleb1

fi


if [ ${stage} -le 45 ]&& [ ${stop_stage} -ge 45 ];then
   echo "SD: Speaker Diarization"
   exp_dir=exp/base_imls_ssl_downstream_task/sd_md ## output
   mkdir -p $exp_dir
   pretrain_model=exp/base_imls_ssl_downstream_task/checkpoint_298_400000.pt
   num_gpus=1
   downstream_task_config=s3prl/s3prl/downstream/diarization/config_md.yaml
   CUDA_VISIBLE_DEVICES=5 torchrun  --master_port=12345 \
          --nproc_per_node $num_gpus \
          s3prl/s3prl/run_downstream_offical.py\
           -m train \
           -u imls_ssl_local\
           -k $pretrain_model\
           -d diarization \
           --config $downstream_task_config\
           -o config.runner.gradient_accumulate_steps=4\
           -o config.downstream_expert.datarc.frame_shift=160\
           --auto_resume \
           --expdir $exp_dir


fi

if [ ${stage} -le 46 ]&& [ ${stop_stage} -ge 46 ];then
   exp_dir=exp/base_imls_ssl_downstream_task/sd_md  ## output
   #pretrain_model=$exp_dir/checkpoint_289_400000.pt
   num_gpus=1
   downstream_task_config=s3prl/s3prl/downstream/diarization/config_md.yaml
   CUDA_VISIBLE_DEVICES=5 \
   torchrun  --master_port=12345 \
      --nproc_per_node $num_gpus \
      s3prl/s3prl/run_downstream_offical.py\
       -m evaluate \
       --config $downstream_task_config\
       -e $exp_dir/best-states-dev.ckpt


fi


## the below script will be running
if [ ${stage} -le 47 ]&& [ ${stop_stage} -ge 47 ];then
    echo "PR: Phoneme Recognition"

   exp_dir=exp/base_imls_ssl_downstream_task/pr_md ## output
   mkdir -p $exp_dir
   pretrain_model=exp/base_imls_ssl_downstream_task/checkpoint_298_400000.pt
   num_gpus=1
   downstream_task_config=s3prl/s3prl/downstream/ctc/libriphone_md.yaml
   CUDA_VISIBLE_DEVICES=6 torchrun  --master_port=12358  \
          --nproc_per_node $num_gpus \
          s3prl/s3prl/run_downstream_offical.py\
           -m train \
           -u imls_ssl_local\
           -k $pretrain_model\
           -d ctc \
           --config $downstream_task_config\
           --auto_resume \
           --expdir $exp_dir
fi
if [ ${stage} -le 48 ]&& [ ${stop_stage} -ge 48 ];then
   exp_dir=exp/base_imls_ssl_downstream_task/pr_md  ## output
   #pretrain_model=$exp_dir/checkpoint_289_400000.pt
   num_gpus=1
   downstream_task_config=s3prl/s3prl/downstream/ctc/libriphone_md.yaml
   CUDA_VISIBLE_DEVICES=6 torchrun  --master_port=12358  \
      --nproc_per_node $num_gpus \
      s3prl/s3prl/run_downstream_offical.py\
       -m evaluate \
       --config $downstream_task_config\
       -e $exp_dir/dev-best.ckpt
# test loss:
# test per: 
fi

#ASR
if [ ${stage} -le 49 ]&& [ ${stop_stage} -ge 49 ];then
   echo "ASR"
   #python3 run_downstream.py -n ExpName -m train -u fbank -d voxceleb1
   exp_dir=exp/base_imls_ssl_downstream_task/asr_md ## output
   mkdir -p $exp_dir
   pretrain_model=exp/base_imls_ssl_downstream_task/checkpoint_298_400000.pt
   num_gpus=4
   downstream_task_config=s3prl/s3prl/downstream/asr/config_md.yaml
   CUDA_VISIBLE_DEVICES=4 torchrun \
          --nproc_per_node $num_gpus \
          --master_port=25678 \
          s3prl/s3prl/run_downstream_offical.py\
           -m train \
           -u imls_ssl_local\
           -k $pretrain_model\
           -d asr\
           --config $downstream_task_config\
           --auto_resume \
           --expdir $exp_dir

fi

if [ ${stage} -le 50 ]&& [ ${stop_stage} -ge 50 ];then
   exp_dir=exp/base_imls_ssl_downstream_task/asr_md  ## output
   #pretrain_model=$exp_dir/checkpoint_289_400000.pt
   num_gpus=1
   downstream_task_config=s3prl/s3prl/downstream/asr/config_md.yaml
   CUDA_VISIBLE_DEVICES=4 torchrun --master_port=25678 \
      --nproc_per_node $num_gpus \
      s3prl/s3prl/run_downstream_offical.py\
       -m evaluate \
       --config $downstream_task_config\
       -e $exp_dir/dev-clean-best.ckpt
fi

## KS
if [ ${stage} -le 51 ]&& [ ${stop_stage} -ge 51 ];then
   echo "KS"
   #s3prl/s3prl/downstream/speech_commands/config.yaml
   exp_dir=exp/base_imls_ssl_downstream_task/ks_md ## output
   mkdir -p $exp_dir
   pretrain_model=exp/base_imls_ssl_downstream_task/checkpoint_298_400000.pt
   num_gpus=1
   downstream_task_config=s3prl/s3prl/downstream/speech_commands/config_md.yaml
   CUDA_VISIBLE_DEVICES=6  torchrun  --master_port=12358   \
          --nproc_per_node $num_gpus \
          s3prl/s3prl/run_downstream_offical.py\
           -m train \
           -u imls_ssl_local\
           -k $pretrain_model\
           -d speech_commands \
           --config $downstream_task_config\
           --auto_resume \
           --expdir $exp_dir

fi


if [ ${stage} -le 52 ]&& [ ${stop_stage} -ge 52 ];then
   exp_dir=exp/base_imls_ssl_downstream_task/ks_md  ## output
   #pretrain_model=$exp_dir/checkpoint_289_400000.pt
   num_gpus=1
   downstream_task_config=s3prl/s3prl/downstream/speech_commands/config_md.yaml
   CUDA_VISIBLE_DEVICES=6  torchrun  --master_port=12358 \
      --nproc_per_node $num_gpus \
      s3prl/s3prl/run_downstream_offical.py\
       -m evaluate \
       --config $downstream_task_config\
       -e $exp_dir/dev-best.ckpt
   ##  cat logs/superb_exp_KS_md.log
   ## test acc: 
fi


# 5e-5 128


## IC: Intent Classification
if [ ${stage} -le 53 ] && [ ${stop_stage} -ge 53 ];then
   exp_dir=exp/base_imls_ssl_downstream_task/ic_md ## output
   mkdir -p $exp_dir
   pretrain_model=exp/base_imls_ssl_downstream_task/checkpoint_298_400000.pt
   num_gpus=1
   downstream_task_config=s3prl/s3prl/downstream/fluent_commands/config_md.yaml
   CUDA_VISIBLE_DEVICES=4 torchrun  --master_port=12358 \
          --nproc_per_node $num_gpus \
          s3prl/s3prl/run_downstream_offical.py\
           -m train \
           -u imls_ssl_local\
           -k $pretrain_model\
           -d fluent_commands\
           --config $downstream_task_config\
           --auto_resume \
           --expdir $exp_dir
fi


if [ ${stage} -le 54 ] && [ ${stop_stage} -ge 54 ];then
   echo "evalute test type"
   exp_dir=exp/base_imls_ssl_downstream_task/ic_md ## output
   #pretrain_model=$exp_dir/checkpoint_289_400000.pt
   num_gpus=1
   downstream_task_config=s3prl/s3prl/downstream/fluent_commands/config_md.yaml
   CUDA_VISIBLE_DEVICES=4 \
   torchrun  --master_port=12358 \
      --nproc_per_node $num_gpus \
      s3prl/s3prl/run_downstream_offical.py\
       -m evaluate \
       --config $downstream_task_config\
       -e $exp_dir/dev-best.ckpt
# test acc:
fi


# SF: End-to-end Slot Filling
if [ ${stage} -le 55 ] && [ ${stop_stage} -ge 55 ];then
   echo "SF: End-to-end Slot Filling"

   exp_dir=exp/base_imls_ssl_downstream_task/sf_md ## output
   mkdir -p $exp_dir
   pretrain_model=exp/base_imls_ssl_downstream_task/checkpoint_298_400000.pt
   num_gpus=3
   downstream_task_config=s3prl/s3prl/downstream/ctc/snips_md.yaml
   CUDA_VISIBLE_DEVICES=2  torchrun  --master_port=12378 \
          --nproc_per_node $num_gpus \
          s3prl/s3prl/run_downstream_offical.py\
           -m train \
           -u imls_ssl_local\
           -k $pretrain_model\
           -d ctc \
           --config $downstream_task_config\
           --auto_resume \
           --expdir $exp_dir
fi
if [ ${stage} -le 56 ]&& [ ${stop_stage} -ge 56 ];then
   exp_dir=exp/base_imls_ssl_downstream_task/sf_md  ## output
   #pretrain_model=$exp_dir/checkpoint_289_400000.pt
   num_gpus=1
   downstream_task_config=s3prl/s3prl/downstream/ctc/snips_md.yaml
   CUDA_VISIBLE_DEVICES=2  torchrun  --master_port=12378 \
      --nproc_per_node $num_gpus \
      s3prl/s3prl/run_downstream_offical.py\
       -m evaluate \
       --config $downstream_task_config\
       -e $exp_dir/dev-best.ckpt

fi

# ER: Emotion Recognition
if [ ${stage} -le 57 ] && [ ${stop_stage} -ge 57 ];then
   for test_fold in Session1  Session2  Session3  Session4  Session5;do
    # The default config is "downstream/emotion/config.yaml"
    exp_dir=exp/base_imls_ssl_downstream_task/er_${test_fold}_md ## output
    pretrain_model=exp/base_imls_ssl_downstream_task/checkpoint_298_400000.pt
    num_gpus=1
    downstream_task_config=s3prl/s3prl/downstream/emotion/config_md.yaml
    CUDA_VISIBLE_DEVICES=0 torchrun  --master_port=12398 \
      --nproc_per_node $num_gpus \
          s3prl/s3prl/run_downstream_offical.py\
          --expdir $exp_dir\
          -m train \
          -u imls_ssl_local\
          -k $pretrain_model\
          -d emotion \
          --config $downstream_task_config\
          -o config.downstream_expert.datarc.test_fold=$test_fold\
          --auto_resume
   done

fi

if [ ${stage} -le 58 ] && [ ${stop_stage} -ge 58 ];then
  for test_fold in Session1  Session2  Session3  Session4  Session5;do
   exp_dir=exp/base_imls_ssl_downstream_task/er_${test_fold}_md ## output
   #pretrain_model=$exp_dir/checkpoint_289_400000.pt
   num_gpus=1
   downstream_task_config=s3prl/s3prl/downstream/emotion/config_md.yaml
   CUDA_VISIBLE_DEVICES=0 torchrun  --master_port=12398 \
      --nproc_per_node $num_gpus \
      s3prl/s3prl/run_downstream_offical.py\
       -m evaluate \
       --config $downstream_task_config\
       -e $exp_dir/dev-best.ckpt
  done

fi




### 2023-11-30 eval llamahubert on downstream task in new hltsz cluster server
if [ ${stage} -le 60 ]&& [ ${stop_stage} -ge 60 ];then
   echo "covert fairseq model to s3prl format"
   pretrain_model=exp/pretrain/continue_pretain_on_hubert_iter2_with_llama_on_train_360_ft_style/checkpoint_192_100000.pt
   dest_dir=exp/base_llamahubert_downstream_task
   python  codebase/s3prl/s3prl/upstream/hubert/convert.py\
             --fairseq_ckpt $pretrain_model\
             --output_dir $dest_dir


fi

# SF without llama
if [ ${stage} -le 65 ] && [ ${stop_stage} -ge 65 ];then
   exp_dir=exp/base_llamahubert_downstream_task/sf_wo_llama ## output
   mkdir -p $exp_dir
   pretrain_model=exp/base_llamahubert_downstream_task/checkpoint_192_100000.pt
   num_gpus=1
   downstream_task_config=codebase/s3prl/s3prl/downstream/ctc/snips_md_for_llamahubert_hltsz_cluster.yaml
   #echo $LD_LIBRARY_PATH
   #unset LD_LIBRARY_PATH
   #echo $LD_LIBRARY_PATH

   ## problem: in group_norm
   ## return torch.group_norm(input, num_groups, weight, bias, eps, torch.backends.cudnn.enabled)
   ##torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 6.80 GiB. GPU 0 has a total capacty of 23.69 GiB of which 2.26 GiB is free. Including non-PyTorch memory, this process has 21.43 GiB memory in use. Of the allocated memory 8.24 GiB is allocated by PyTorch, and 11.45 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
   ## the below command is soloved.
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
   #CUDA_VISIBLE_DEVICES=4 \
   torchrun  --nproc_per_node $num_gpus  --master_port 25642\
          --nproc_per_node $num_gpus \
          codebase/s3prl/s3prl/run_downstream.py\
           -m train \
           -u llamahubert_without_llama_local\
           -k $pretrain_model\
           -d ctc\
           --config $downstream_task_config\
           -o config.runner.gradient_accumulate_steps=4\
           --auto_resume \
           --expdir $exp_dir
fi


if [ ${stage} -le 66 ] && [ ${stop_stage} -ge 66 ];then
   echo "evalute test type"
   exp_dir=exp/base_llamahubert_downstream_task/sf_wo_llama  ## output
   #pretrain_model=$exp_dir/checkpoint_289_400000.pt
   num_gpus=1
   downstream_task_config=codebase/s3prl/s3prl/downstream/ctc/snips_md_for_llamahubert_hltsz_cluster.yaml
   #CUDA_VISIBLE_DEVICES=6 \
   torchrun  --nproc_per_node $num_gpus  --master_port 25642\
      --nproc_per_node $num_gpus \
      codebase/s3prl/s3prl/run_downstream.py\
       -m evaluate \
       --config $downstream_task_config\
       -e $exp_dir/dev-best.ckpt
#  slot-type F1 score
#  slot value CER
# 
# submit: slot_type_f1:, slot_value_cer:
#test loss: 0.6474696397781372
#test slot_type_f1: 0.8848453669346537
#test slot_value_cer: 0.24219329869579048
#test slot_value_wer: 0.3787187039764359
#test slot_edit_f1_full: 64.74820143884892
#test slot_edit_f1_part: 65.6347670784343
#test wer: 0.15891128235526575
#test cer: 0.08160989821242312


fi
