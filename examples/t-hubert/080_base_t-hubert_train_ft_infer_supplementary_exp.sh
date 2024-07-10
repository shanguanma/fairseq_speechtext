#!/usr/bin/env bash

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
   echo "fine tune base t-hubert model  using train-clean-360 supervision data"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntcephfs/lab_data/maduo/exp
   config_dir=$fairseq_dir/examples/t-hubert
   #model_name=pretrain_on_base_t-hubert_4gpu_8update_960h_400k_update
   model_name=pretrain_on_base_imls-ssl_4gpu_8update_960h_400k_update # imls-ssl is same as t-hubert.
   exp_finetune_dir=$dir/finetune/${model_name}_360h_asr_finetune
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_finetune_dir
   world_size=4
   update_freq=2
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   python $fairseq_dir/fairseq_cli/hydra_train.py \
       --config-dir $config_dir/config/finetune \
            --config-name t-hubert_base_100h \
            task.data=$tsv_dir\
            task.label_dir=$tsv_dir\
            task.labels='["ltr"]' \
            model.w2v_path=$exp_dir/checkpoint_298_400000.pt\
            common.user_dir=$fairseq_dir/examples/t-hubert\
            dataset.train_subset=train-clean-360\
            dataset.valid_subset=dev-other\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_finetune_dir\
            checkpoint.save_dir=$exp_finetune_dir\
            hydra.run.dir=$fairseq_dir/examples/t-hubert\
            hydra.job.name=$exp_finetune_dir/finetune
fi
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
   echo "fine tune base t-hubert model  using train-clean-360 supervision data"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntcephfs/lab_data/maduo/exp
   config_dir=$fairseq_dir/examples/t-hubert
   #model_name=pretrain_on_base_t-hubert_4gpu_8update_960h_400k_update
   model_name=pretrain_on_base_imls-ssl_4gpu_8update_960h_400k_update # imls-ssl is same as t-hubert.
   exp_finetune_dir=$dir/finetune/${model_name}_360h_asr_finetune_10ksteps
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_finetune_dir
   world_size=4
   update_freq=2
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   python $fairseq_dir/fairseq_cli/hydra_train.py \
       --config-dir $config_dir/config/finetune \
            --config-name t-hubert_base_360h \
            task.data=$tsv_dir\
            task.label_dir=$tsv_dir\
            task.labels='["ltr"]' \
            model.w2v_path=$exp_dir/checkpoint_298_400000.pt\
            common.user_dir=$fairseq_dir/examples/t-hubert\
            dataset.train_subset=train-clean-360\
            dataset.valid_subset=dev-other\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_finetune_dir\
            checkpoint.save_dir=$exp_finetune_dir\
            hydra.run.dir=$fairseq_dir/examples/t-hubert\
            hydra.job.name=$exp_finetune_dir/finetune
fi
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
   echo "fine tune base hubert model  using train-other-500 supervision data"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntcephfs/lab_data/maduo/exp
   config_dir=$fairseq_dir/examples/t-hubert
   #model_name=pretrain_on_base_t-hubert_4gpu_8update_960h_400k_update
   model_name=pretrain_on_base_imls-ssl_4gpu_8update_960h_400k_update # imls-ssl is same as t-hubert.
   exp_finetune_dir=$dir/finetune/${model_name}_500h_asr_finetune_12ksteps
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_finetune_dir
   world_size=4
   update_freq=2
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   python $fairseq_dir/fairseq_cli/hydra_train.py \
       --config-dir $config_dir/config/finetune \
            --config-name t-hubert_base_500h \
            task.data=$tsv_dir\
            task.label_dir=$tsv_dir\
            task.labels='["ltr"]' \
            model.w2v_path=$exp_dir/checkpoint_298_400000.pt\
            common.user_dir=$fairseq_dir/examples/t-hubert\
            dataset.train_subset=train-other-500\
            dataset.valid_subset=dev-other\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_finetune_dir\
            checkpoint.save_dir=$exp_finetune_dir\
            hydra.run.dir=$fairseq_dir/examples/t-hubert\
            hydra.job.name=$exp_finetune_dir/finetune
fi

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ];then
   echo "fine tune base hubert model  using train-other-500 supervision data"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntcephfs/lab_data/maduo/exp
   config_dir=$fairseq_dir/examples/t-hubert
   #model_name=pretrain_on_base_t-hubert_4gpu_8update_960h_400k_update
   model_name=hubert_base_librispeech_offical_no_finetune
   exp_finetune_dir=$dir/finetune/${model_name}_500h_asr_finetune_12ksteps
   exp_dir=/mntcephfs/lab_data/maduo/model_hub/librispeech/hubert_base_librispeech_offical_no_finetune
   mkdir -p $exp_finetune_dir
   world_size=4
   update_freq=2
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   python $fairseq_dir/fairseq_cli/hydra_train.py \
       --config-dir $config_dir/config/finetune \
            --config-name t-hubert_base_500h \
            task.data=$tsv_dir\
            task.label_dir=$tsv_dir\
            task.labels='["ltr"]' \
            model.w2v_path=$exp_dir/hubert_base_ls960.pt\
            common.user_dir=$fairseq_dir/examples/t-hubert\
            dataset.train_subset=train-other-500\
            dataset.valid_subset=dev-other\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_finetune_dir\
            checkpoint.save_dir=$exp_finetune_dir\
            hydra.run.dir=$fairseq_dir/examples/t-hubert\
            hydra.job.name=$exp_finetune_dir/finetune
fi

if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ];then
   echo "fine tune base hubert model  using train-clean-360 supervision data"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntcephfs/lab_data/maduo/exp
   config_dir=$fairseq_dir/examples/t-hubert
   #model_name=pretrain_on_base_t-hubert_4gpu_8update_960h_400k_update
   model_name=hubert_base_librispeech_offical_no_finetune
   exp_finetune_dir=$dir/finetune/${model_name}_360h_asr_finetune_12ksteps
   exp_dir=/mntcephfs/lab_data/maduo/model_hub/librispeech/hubert_base_librispeech_offical_no_finetune
   mkdir -p $exp_finetune_dir
   world_size=4
   update_freq=2
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   python $fairseq_dir/fairseq_cli/hydra_train.py \
       --config-dir $config_dir/config/finetune \
            --config-name t-hubert_base_360h \
            task.data=$tsv_dir\
            task.label_dir=$tsv_dir\
            task.labels='["ltr"]' \
            model.w2v_path=$exp_dir/hubert_base_ls960.pt\
            common.user_dir=$fairseq_dir/examples/t-hubert\
            dataset.train_subset=train-clean-360\
            dataset.valid_subset=dev-other\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_finetune_dir\
            checkpoint.save_dir=$exp_finetune_dir\
            hydra.run.dir=$fairseq_dir/examples/t-hubert\
            hydra.job.name=$exp_finetune_dir/finetune
fi


if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ];then
   echo "inference t-hubert  model on test-other, test-clean of librispeech"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntcephfs/lab_data/maduo/exp

   #config_dir=$fairseq_dir/examples/hubert/
   model_name=pretrain_on_base_imls-ssl_4gpu_8update_960h_400k_update # imls-ssl is same as t-hubert.
   exp_finetune_dir=$dir/finetune/${model_name}_360h_asr_finetune_10ksteps
   #results_path=$exp_finetune_dir/decode_on_100h
   results_path=$exp_finetune_dir/decode_on_360h_normalize_false
   mkdir -p $results_path
   #testsets="dev-clean dev-other test-clean test-other"
   testsets="test-clean test-other"
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
fi

if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ];then
   echo "inference t-hubert  model on test-other, test-clean of librispeech"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntcephfs/lab_data/maduo/exp

   #config_dir=$fairseq_dir/examples/hubert/
   model_name=pretrain_on_base_imls-ssl_4gpu_8update_960h_400k_update # imls-ssl is same as t-hubert.
   exp_finetune_dir=$dir/finetune/${model_name}_500h_asr_finetune_10ksteps
   #results_path=$exp_finetune_dir/decode_on_100h
   results_path=$exp_finetune_dir/decode_on_500h_normalize_false
   mkdir -p $results_path
   #testsets="dev-clean dev-other test-clean test-other"
   testsets="test-clean test-other"
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
fi

if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ];then
   echo "inference t-hubert  model on test-other, test-clean of librispeech with kenlm"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntcephfs/lab_data/maduo/exp

   ##config_dir=$fairseq_dir/examples/hubert/
   model_name=pretrain_on_base_imls-ssl_4gpu_8update_960h_400k_update # imls-ssl is same as t-hubert.
   exp_finetune_dir=$dir/finetune/${model_name}_360h_asr_finetune
   results_path=$exp_finetune_dir/decode_on_360h_with_kenlm
   mkdir -p $results_path
   path_to_lexicon=/mntcephfs/lab_data/maduo/datasets/librispeech/kenlm_files/librispeech_lexicon.lst #word2letter
   path_to_lm=/mntcephfs/lab_data/maduo/datasets/librispeech/kenlm_files/4-gram.arpa  ## word lm
   testsets="test-clean test-other"
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH

   for name in $testsets;do
       python $fairseq_dir/examples/speech_recognition/new/infer.py \
                --config-dir $fairseq_dir/examples/speech_recognition/new/conf\
                --config-name infer_kenlm_lirispeech\
                task.data=$tsv_dir\
                task.label_dir=$tsv_dir\
                task.normalize=false\
                common_eval.results_path=$results_path\
                common_eval.path=$exp_finetune_dir/checkpoint_best.pt\
                dataset.gen_subset=$name\
                decoding.type=kenlm\
                decoding.lexicon=$path_to_lexicon\
                decoding.lmpath=$path_to_lm\
                decoding.nbest=1\
                decoding.beam=1500 \
                decoding.lmweight=2 \
                decoding.wordscore=-1 \
                decoding.beamthreshold=100\
                common_eval.quiet=false

   done
 fi

if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ];then
   echo "inference t-hubert  model on test-other, test-clean of librispeech with kenlm"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntcephfs/lab_data/maduo/exp

   ##config_dir=$fairseq_dir/examples/hubert/
   model_name=pretrain_on_base_imls-ssl_4gpu_8update_960h_400k_update # imls-ssl is same as t-hubert.
   exp_finetune_dir=$dir/finetune/${model_name}_500h_asr_finetune
   results_path=$exp_finetune_dir/decode_on_500h_with_kenlm
   mkdir -p $results_path
   path_to_lexicon=/mntcephfs/lab_data/maduo/datasets/librispeech/kenlm_files/librispeech_lexicon.lst #word2letter
   path_to_lm=/mntcephfs/lab_data/maduo/datasets/librispeech/kenlm_files/4-gram.arpa  ## word lm
   testsets="test-clean test-other"
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH

   for name in $testsets;do
       python $fairseq_dir/examples/speech_recognition/new/infer.py \
                --config-dir $fairseq_dir/examples/speech_recognition/new/conf\
                --config-name infer_kenlm_lirispeech\
                task.data=$tsv_dir\
                task.label_dir=$tsv_dir\
                task.normalize=false\
                common_eval.results_path=$results_path\
                common_eval.path=$exp_finetune_dir/checkpoint_best.pt\
                dataset.gen_subset=$name\
                decoding.type=kenlm\
                decoding.lexicon=$path_to_lexicon\
                decoding.lmpath=$path_to_lm\
                decoding.nbest=1\
                decoding.beam=1500 \
                decoding.lmweight=2 \
                decoding.wordscore=-1 \
                decoding.beamthreshold=100\
                common_eval.quiet=false

   done
   fi


if [ ${stage} -le 16 ] && [ ${stop_stage} -ge 16 ];then
   echo "inference hubert  model on test-other, test-clean of librispeech"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntcephfs/lab_data/maduo/exp

   #config_dir=$fairseq_dir/examples/hubert/
   model_name=hubert_base_librispeech_offical_no_finetune
   exp_finetune_dir=$dir/finetune/${model_name}_360h_asr_finetune_10ksteps
   #results_path=$exp_finetune_dir/decode_on_100h
   results_path=$exp_finetune_dir/decode_on_360h_normalize_false
   mkdir -p $results_path
   #testsets="dev-clean dev-other test-clean test-other"
   testsets="test-clean test-other"
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
fi

if [ ${stage} -le 17 ] && [ ${stop_stage} -ge 17 ];then
   echo "inference hubert  model on test-other, test-clean of librispeech"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntcephfs/lab_data/maduo/exp

   #config_dir=$fairseq_dir/examples/hubert/
   model_name=hubert_base_librispeech_offical_no_finetune
   exp_finetune_dir=$dir/finetune/${model_name}_500h_asr_finetune_10ksteps
   #results_path=$exp_finetune_dir/decode_on_100h
   results_path=$exp_finetune_dir/decode_on_500h_normalize_false
   mkdir -p $results_path
   #testsets="dev-clean dev-other test-clean test-other"
   testsets="test-clean test-other"
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
fi
if [ ${stage} -le 18 ] && [ ${stop_stage} -ge 18 ];then
   echo "inference hubert  model on test-other, test-clean of librispeech with kenlm"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntcephfs/lab_data/maduo/exp

   ##config_dir=$fairseq_dir/examples/hubert/
   model_name=hubert_base_librispeech_offical_no_finetune
   exp_finetune_dir=$dir/finetune/${model_name}_500h_asr_finetune
   results_path=$exp_finetune_dir/decode_on_500h_with_kenlm
   mkdir -p $results_path
   path_to_lexicon=/mntcephfs/lab_data/maduo/datasets/librispeech/kenlm_files/librispeech_lexicon.lst #word2letter
   path_to_lm=/mntcephfs/lab_data/maduo/datasets/librispeech/kenlm_files/4-gram.arpa  ## word lm
   testsets="test-clean test-other"
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH

   for name in $testsets;do
       python $fairseq_dir/examples/speech_recognition/new/infer.py \
                --config-dir $fairseq_dir/examples/speech_recognition/new/conf\
                --config-name infer_kenlm_lirispeech\
                task.data=$tsv_dir\
                task.label_dir=$tsv_dir\
                task.normalize=false\
                common_eval.results_path=$results_path\
                common_eval.path=$exp_finetune_dir/checkpoint_best.pt\
                dataset.gen_subset=$name\
                decoding.type=kenlm\
                decoding.lexicon=$path_to_lexicon\
                decoding.lmpath=$path_to_lm\
                decoding.nbest=1\
                decoding.beam=1500 \
                decoding.lmweight=2 \
                decoding.wordscore=-1 \
                decoding.beamthreshold=100\
                common_eval.quiet=false
   done
   fi


   if [ ${stage} -le 19 ] && [ ${stop_stage} -ge 19 ];then
   echo "inference hubert  model on test-other, test-clean of librispeech with kenlm"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntcephfs/lab_data/maduo/exp

   ##config_dir=$fairseq_dir/examples/hubert/
   model_name=hubert_base_librispeech_offical_no_finetune
   exp_finetune_dir=$dir/finetune/${model_name}_360h_asr_finetune
   results_path=$exp_finetune_dir/decode_on_360h_with_kenlm
   mkdir -p $results_path
   path_to_lexicon=/mntcephfs/lab_data/maduo/datasets/librispeech/kenlm_files/librispeech_lexicon.lst #word2letter
   path_to_lm=/mntcephfs/lab_data/maduo/datasets/librispeech/kenlm_files/4-gram.arpa  ## word lm
   testsets="test-clean test-other"
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH

   for name in $testsets;do
       python $fairseq_dir/examples/speech_recognition/new/infer.py \
                --config-dir $fairseq_dir/examples/speech_recognition/new/conf\
                --config-name infer_kenlm_lirispeech\
                task.data=$tsv_dir\
                task.label_dir=$tsv_dir\
                task.normalize=false\
                common_eval.results_path=$results_path\
                common_eval.path=$exp_finetune_dir/checkpoint_best.pt\
                dataset.gen_subset=$name\
                decoding.type=kenlm\
                decoding.lexicon=$path_to_lexicon\
                decoding.lmpath=$path_to_lm\
                decoding.nbest=1\
                decoding.beam=1500 \
                decoding.lmweight=2 \
                decoding.wordscore=-1 \
                decoding.beamthreshold=100\
                common_eval.quiet=false
   done
   fi
