#!/usr/bin/env bash
  
stage=0

stop_stage=1000
.  utils/parse_options.sh
. path_for_fairseq_speechtext.sh ## pytorch 2.0 fairseq flashnight-text flashnight sequence 

export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
## (TODO) first run it
if [ ${stage} -le 16 ] && [ ${stop_stage} -ge 16 ];then
   echo "iter: pretrain imls_ssl on 6layer of hubert pesudo label and librispeech monophncode from w2vu2-model " 
   echo "training on 400k steps for train-960 of librispeech"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   label_dir=$tsv_dir/offical_hubert_codes_and_librispeech_frame_monophncode_using_wav2vec-u2_model
   config_dir=$fairseq_dir/examples/imls_ssl
   dir=/workspace2/maduo/exp
   model_name=pretrain_on_base_imls-ssl_4gpu_8update_960h_400k_update
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_dir
   world_size=4
   update_freq=8
   CUDA_VISIBLE_DEVICES=4,5,6,7 python $fairseq_dir/fairseq_cli/hydra_train.py \
            --config-dir $config_dir/config/pretrain \
            --config-name imls_ssl_base_librispeech \
            task.data=$tsv_dir\
            task.label_dir=$label_dir\
            task.labels='["phncode,km"]' \
            model.label_rate=50\
            model.predict_layers='[7,12]'\
            model.phnkm7_km12=true\
            common.user_dir=$fairseq_dir/examples/imls_ssl\
            dataset.train_subset=train-960\
            dataset.valid_subset=\'dev-other,dev-clean\'\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_dir\
            checkpoint.save_dir=$exp_dir\
            hydra.run.dir=$fairseq_dir/examples/imls_ssl\
            hydra.job.name=$exp_dir/pretrain
### 4RTX3090: training about  day
###           400steps: about  minites
fi



if [ ${stage} -le 17 ] && [ ${stop_stage} -ge 17 ];then
   echo "fine tune base imls-ssl model  using train-clean-100 supervision data"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   config_dir=$fairseq_dir/examples/imls_ssl
   dir=/workspace2/maduo/exp
   model_name=pretrain_on_base_imls-ssl_4gpu_8update_960h_400k_update
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_finetune_dir
   world_size=4
   update_freq=2
   CUDA_VISIBLE_DEVICES=4,5,6,7 python $fairseq_dir/fairseq_cli/hydra_train.py \
       --config-dir $config_dir/config/finetune \
            --config-name imls_ssl_base_100h \
            task.data=$tsv_dir\
            task.label_dir=$tsv_dir\
            task.labels='["ltr"]' \
            model.w2v_path=$exp_dir/checkpoint_last.pt\
            common.user_dir=$fairseq_dir/examples/imls_ssl\
            dataset.train_subset=train-clean-100\
            dataset.valid_subset=dev-other\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_finetune_dir\
            checkpoint.save_dir=$exp_finetune_dir\
            hydra.run.dir=$fairseq_dir/examples/imls_ssl\
            hydra.job.name=$exp_dir/finetune
fi

if [ ${stage} -le 18 ] && [ ${stop_stage} -ge 18 ];then
   echo "inference imls-ssl  model on dev-other, dev-clean, test-other, test-clean of librispeech"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   config_dir=$fairseq_dir/examples/hubert/
   dir=/workspace2/maduo/exp
   model_name=pretrain_on_base_ils-ssl_4gpu_8update_960h_ils-ssl_400k_update
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune
   mkdir -p $exp_finetune_dir/decode_on_100h
   testsets="dev-clean dev-other test-clean test-other"
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH

   for name in $testsets;do
       python $fairseq_dir/examples/speech_recognition/new/infer.py \
                --config-dir $config_dir/config/decode\
                --config-name infer_viterbi\
                task.data=$tsv_dir\
                task.label_dir=$tsv_dir\
                task.normalize=true\
                common_eval.results_path=$exp_finetune_dir/decode_on_100h\
                common_eval.path=$exp_finetune_dir/checkpoint_best.pt\
                dataset.gen_subset=$name

   done
   # WER% is from $exp_finetune_dir/decode_on_100/viterbi/infer.log
   #  dev-clean   dev-other   test-clean   test-other


fi


