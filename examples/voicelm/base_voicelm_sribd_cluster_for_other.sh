#!/bin/bash
stage=0

stop_stage=1000
.  utils/parse_options.sh
. path_for_fairseq_speechtext.sh ## pytorch 2.0 fairseq flashnight-text flashnight sequence
#. path_for_fsq_speechtext.sh
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

## build c++ part using CUDA, however in head node of this slurm server system hasn't cuda gpu.
cd /mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
pip install ninja ## using fast  distutils backend training for pytorch, it is very important
#pip install --editable ./  ## for python package, it can be installed at local environment
python setup.py build_ext --inplace


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
   echo "iter: pretrain voicelm on 6layer of hubert pesudo label and librispeech monophncode from w2vu2-model "
   echo "training on 400k steps for train-960 of librispeech"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   #tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp
   #label_dir=$tsv_dir/offical_hubert_codes_and_librispeech_frame_monophncode_using_wav2vec-u2_model
   tsv_dir=/path/to/w2v_unsup_gan_xp_4gpu_8update_with_unpair_text_0.3M
   label_dir=/path/to/w2v_unsup_gan_xp_4gpu_8update_with_unpair_text_0.3M
   config_dir=$fairseq_dir/examples/voicelm
   model_name=pretrain_on_base_imls-ssl_4gpu_8update_960h_400k_update
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_dir
   world_size=4
   update_freq=8
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   python $fairseq_dir/fairseq_cli/hydra_train.py \
            --config-dir $config_dir/config/pretrain \
            --config-name voicelm_base_librispeech \
            task.data=$tsv_dir\
            task.label_dir=$label_dir\
            task.labels='["unsupphncode1","km"]' \
            model.label_rate=50\
            common.user_dir=$fairseq_dir/examples/voicelm\
            dataset.train_subset=train-960\
            dataset.valid_subset=\'dev-other,dev-clean\'\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_dir\
            checkpoint.save_dir=$exp_dir\
            hydra.run.dir=$fairseq_dir/examples/voicelm\
            hydra.job.name=$exp_dir/pretrain
fi


if [ ${stage} -le 28 ] && [ ${stop_stage} -ge 28 ];then
   echo "fine tune base voicelm model  using train-1h supervision data"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp
   config_dir=$fairseq_dir/examples/voicelm
   model_name=pretrain_on_base_voicelm_8gpu_2update_960h_400k_update_on_unpaired_text_0.3M
   exp_finetune_dir=$dir/finetune/${model_name}_1h_asr_finetune
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_finetune_dir
   world_size=2
   update_freq=4
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   python $fairseq_dir/fairseq_cli/hydra_train.py \
       --config-dir $config_dir/config/finetune \
            --config-name voicelm_base_1h \
            task.data=$tsv_dir\
            task.label_dir=$tsv_dir\
            task.labels='["ltr"]' \
            model.w2v_path=$exp_dir/checkpoint_best.pt\
            common.user_dir=$fairseq_dir/examples/voicelm\
            dataset.train_subset=train-1h\
            dataset.valid_subset=dev-other\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_finetune_dir\
            checkpoint.save_dir=$exp_finetune_dir\
            hydra.run.dir=$fairseq_dir/examples/voicelm\
            hydra.job.name=$exp_finetune_dir/finetune
fi


if [ ${stage} -le 29 ] && [ ${stop_stage} -ge 29 ];then
   echo "inference voicelm  model on dev-other, dev-clean, test-other, test-clean of librispeech"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp

   config_dir=$fairseq_dir/examples/hubert/
   model_name=pretrain_on_base_voicelm_8gpu_2update_960h_400k_update_on_unpaired_text_0.3M
   exp_finetune_dir=$dir/finetune/${model_name}_1h_asr_finetune
   #results_path=$exp_finetune_dir/decode_on_100h
   results_path=$exp_finetune_dir/decode_on_1h_viterbi_normalize_false
   mkdir -p $results_path
   testsets="dev-clean dev-other test-clean test-other"
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH

   for name in $testsets;do
       python $fairseq_dir/examples/speech_recognition/new/infer.py \
                --config-dir $fairseq_dir/examples/speech_recognition/new/conf\
                --config-name infer_viterbi_librispeech\
                task.data=$tsv_dir\
                task.label_dir=$tsv_dir\
                task.normalize=false\
                common_eval.results_path=$results_path\
                common_eval.path=$exp_finetune_dir/checkpoint_best.pt\
                dataset.gen_subset=$name

   done

   # grep -rn 'Word error rate' logs/voicelm_0.3M_text_pretrain_and_finetune.log
   #24785:[2024-02-19 14:25:30,224][__main__][INFO] - Word error rate: 28.7600
   # 29098:[2024-02-19 14:26:34,698][__main__][INFO] - Word error rate: 33.6212
   # 33090:[2024-02-19 14:27:42,441][__main__][INFO] - Word error rate: 29.2522
   # 37556:[2024-02-19 14:28:49,202][__main__][INFO] - Word error rate: 33.9345
   fi


if [ ${stage} -le 30 ] && [ ${stop_stage} -ge 30 ];then
   echo "fine tune base voicelm model  using train-1h supervision data"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp
   config_dir=$fairseq_dir/examples/voicelm
   model_name=pretrain_on_base_voicelm_8gpu_2update_960h_400k_update_on_unpaired_text_0.3M
   exp_finetune_dir=$dir/finetune/${model_name}_1h_asr_finetune_continue
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_finetune_dir
   world_size=2
   update_freq=4
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   python $fairseq_dir/fairseq_cli/hydra_train.py \
       --config-dir $config_dir/config/finetune \
            --config-name voicelm_base_1h \
            task.data=$tsv_dir\
            task.label_dir=$tsv_dir\
            task.labels='["ltr"]' \
            model.w2v_path=$exp_dir/checkpoint_last.pt\
            common.user_dir=$fairseq_dir/examples/voicelm\
            dataset.train_subset=train-1h\
            dataset.valid_subset=dev-other\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_finetune_dir\
            checkpoint.save_dir=$exp_finetune_dir\
            hydra.run.dir=$fairseq_dir/examples/voicelm\
            hydra.job.name=$exp_finetune_dir/finetune
fi



if [ ${stage} -le 31 ] && [ ${stop_stage} -ge 31 ];then
   echo "inference voicelm  model on dev-other, dev-clean, test-other, test-clean of librispeech"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp

   config_dir=$fairseq_dir/examples/hubert/
   model_name=pretrain_on_base_voicelm_8gpu_2update_960h_400k_update_on_unpaired_text_0.3M
   exp_finetune_dir=$dir/finetune/${model_name}_1h_asr_finetune_continue
   #results_path=$exp_finetune_dir/decode_on_100h
   results_path=$exp_finetune_dir/decode_on_1h_viterbi_normalize_false
   mkdir -p $results_path
   testsets="dev-clean dev-other test-clean test-other"
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH

   for name in $testsets;do
       python $fairseq_dir/examples/speech_recognition/new/infer.py \
                --config-dir $fairseq_dir/examples/speech_recognition/new/conf\
                --config-name infer_viterbi_librispeech\
                task.data=$tsv_dir\
                task.label_dir=$tsv_dir\
                task.normalize=false\
                common_eval.results_path=$results_path\
                common_eval.path=$exp_finetune_dir/checkpoint_best.pt\
                dataset.gen_subset=$name

   done

   # grep -rn 'Word error rate:' logs/voicelm_0.3M_text_pretrain_and_finetune_continue.log
   # 24217:[2024-02-23 18:32:29,619][__main__][INFO] - Word error rate: 27.0854
   # 28530:[2024-02-23 18:33:54,928][__main__][INFO] - Word error rate: 31.5778
   # 32522:[2024-02-23 18:35:15,505][__main__][INFO] - Word error rate: 27.6447
   # 36988:[2024-02-23 18:36:32,143][__main__][INFO] - Word error rate: 31.5827
fi

