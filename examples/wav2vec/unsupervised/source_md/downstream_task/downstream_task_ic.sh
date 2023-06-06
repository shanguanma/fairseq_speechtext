#!/usr/bin/env bash

stage=-1
stop_stage=1000
. utils/parse_options.sh
. path_for_fairseq_speechtext.sh


## stage-1-1 for debug s3prl pipeline  using standard hubert 
if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ];then
    pretrain_model=exp/pretrain/pretrain_on_hubert_4gpu_8update_960h_mfcc_250k_update/checkpoint_289_400000.pt
    exp_dir=exp/downstream_task/downstream_task_ic
    python   s3prl/s3prl/upstream/hubert/convert.py\
               --fairseq_ckpt   $pretrain_model\
               --output_dir $exp_dir
                
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
   exp_dir=exp/downstream_task/downstream_task_ic ## output 
   pretrain_model=$exp_dir/checkpoint_289_400000.pt
   num_gpus=4
   downstream_task_config=s3prl/s3prl/downstream/fluent_commands/config.yaml
   CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --use_env --nproc_per_node $num_gpus s3prl/s3prl/run_downstream.py\
           -m train \
           -u hubert_local\
           -k $pretrain_model\
           -d fluent_commands\
           --config $downstream_task_config\
           -o config.runner.gradient_accumulate_steps=1\
           --auto_resume \
           --expdir $exp_dir
fi
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
   echo "evalute test type"
   exp_dir=exp/downstream_task/downstream_task_ic ## output 
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
# test acc: 0.9607170820236206
fi

## stage2-4 for debug imls_ssl upstream model code part
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
    pretrain_model=exp/pretrain/pretrain_on_small_ils-ssl_4gpu_8update_960h_mfcc_250k_update/checkpoint_last.pt
    exp_dir=exp/downstream_task/downstream_task_ic_test_small_imls_ssl
    python   s3prl/s3prl/upstream/hubert/convert.py\
               --fairseq_ckpt   $pretrain_model\
               --output_dir $exp_dir

fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
   exp_dir=exp/downstream_task/downstream_task_ic_test_small_imls_ssl ## output 
   pretrain_model=$exp_dir/checkpoint_last.pt
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


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
   echo "evalute test type"
   exp_dir=exp/downstream_task/downstream_task_ic_test_small_imls_ssl ## output 
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

