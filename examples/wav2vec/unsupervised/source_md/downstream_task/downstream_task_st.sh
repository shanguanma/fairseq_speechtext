#!/usr/bin/env bash
  
stage=-1
stop_stage=1000
. utils/parse_options.sh
. path_for_fairseq_speechtext.sh


## stage-1-1 for debug
if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ];then
   echo " prepare data"
   covo_root=dataset/downstreams_tasks/Speech_Translation/common_voice_corpus_4_en ## covot2 corpus directory
   code_dir=source_md/downstream_task/speech_translation/    ## codebase directory
   common_voice_4_en_dir=$covo_root  ## common_voice_4_en corpus directory e.g. /path/to/common_voice_corpus_4_en  
   bash $code_dir/prepare_data/prepare_covo.sh \
           --stage -1 --stop-stage 4\
           $covo_root\
           $code_dir\
           $common_voice_4_en_dir
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
    pretrain_model=exp/pretrain/pretrain_on_small_ils-ssl_4gpu_8update_960h_mfcc_250k_update/checkpoint_last.pt
    exp_dir=exp/downstream_task/downstream_task_st_test_small_imls_ssl
    python   s3prl/s3prl/upstream/hubert/convert.py\
               --fairseq_ckpt   $pretrain_model\
               --output_dir $exp_dir

fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
   exp_dir=exp/downstream_task/downstream_task_st_test_small_imls_ssl ## output 
   pretrain_model=$exp_dir/checkpoint_last.pt
   num_gpus=1
   downstream_task_config=s3prl/s3prl/downstream/speech_translation/config.yaml
   CUDA_VISIBLE_DEVICES=0 \
    python -m torch.distributed.launch \
          --use_env --nproc_per_node $num_gpus \
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

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
   echo "evalute test type"
   exp_dir=exp/downstream_task/downstream_task_st_test_small_imls_ssl ## output 
   #pretrain_model=$exp_dir/checkpoint_289_400000.pt
   num_gpus=1
   downstream_task_config=s3prl/s3prl/downstream/speech_translation/config.yaml
   CUDA_VISIBLE_DEVICES=0 \
   python -m torch.distributed.launch \
      --use_env \
      --nproc_per_node $num_gpus \
      s3prl/s3prl/run_downstream.py\
       -m evaluate \
       --config $downstream_task_config\
       -e $exp_dir/dev-best.ckpt
# test acc:
fi

