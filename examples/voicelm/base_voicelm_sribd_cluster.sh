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
   echo "iter: pretrain voicelm on 6layer of hubert pesudo label and librispeech monophncode from w2vu2-model " 
   echo "training on 400k steps for train-960 of librispeech"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp
   label_dir=$tsv_dir/offical_hubert_codes_and_librispeech_frame_monophncode_using_wav2vec-u2_model
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
            task.labels='["phncode","km"]' \
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
### 4V100: training about 30.6 day
###           200steps: about 22 minites
fi



if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
   echo "fine tune base imls-ssl model  using train-clean-100 supervision data"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp
   config_dir=$fairseq_dir/examples/voicelm
   model_name=pretrain_on_base_imls-ssl_4gpu_8update_960h_400k_update
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_finetune_dir
   world_size=4
   update_freq=2
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   python $fairseq_dir/fairseq_cli/hydra_train.py \
       --config-dir $config_dir/config/finetune \
            --config-name voicelm_base_100h \
            task.data=$tsv_dir\
            task.label_dir=$tsv_dir\
            task.labels='["ltr"]' \
            model.w2v_path=$exp_dir/checkpoint_298_400000.pt\
            common.user_dir=$fairseq_dir/examples/voicelm\
            dataset.train_subset=train-clean-100\
            dataset.valid_subset=dev-other\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_finetune_dir\
            checkpoint.save_dir=$exp_finetune_dir\
            hydra.run.dir=$fairseq_dir/examples/voicelm\
            hydra.job.name=$exp_dir/finetune
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
   echo "inference imls-ssl  model on dev-other, dev-clean, test-other, test-clean of librispeech"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp

   config_dir=$fairseq_dir/examples/hubert/
   model_name=pretrain_on_base_imls-ssl_4gpu_8update_960h_400k_update
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune
   #results_path=$exp_finetune_dir/decode_on_100h
   results_path=$exp_finetune_dir/decode_on_100h_normalize_false
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
   #
   # WER% is grep -rn "Word error rate" $exp_finetune_dir/decode_on_100/viterbi/infer.log
   ## or  grep -rn "Word error rate" logs/base_imls-ssl_stage3_infer.log
   #  dev-clean   dev-other   test-clean   test-other  ## task.normalize=true, however it is setting false at pretrain stage.
   #    4.4373     10.6254     4.4420      10.1601
   #  grep -rn "Word error rate" logs/base_imls-ssl_stage3_infer_normalize_false.log , ## task.normalize=false. it is same as pretain stage
   #    4.4576     10.4095     4.4420      10.1677


fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
   echo "inference imls-ssl  model on dev-other, dev-clean, test-other, test-clean of librispeech with kenlm"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp

   config_dir=$fairseq_dir/examples/hubert/
   model_name=pretrain_on_base_imls-ssl_4gpu_8update_960h_400k_update
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune
   results_path=$exp_finetune_dir/decode_on_100h_with_kenlm
   mkdir -p $results_path
   path_to_lexicon=/mntcephfs/lab_data/maduo/datasets/librispeech/kenlm_files/librispeech_lexicon.lst #word2letter
   path_to_lm=/mntcephfs/lab_data/maduo/datasets/librispeech/kenlm_files/4-gram.arpa  ## word lm
   testsets="dev-clean dev-other test-clean test-other"
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
   # WER% in kenlm and task.normalize=false
   # log is from grep -rn "Word error rate" logs/base_imls-ssl_stage4_infer_with_kenlm.log
   # dev-clean  dev-other test-clean test-other
   # 2.5459      7.2079      3.0666     7.2580
   ## NOTE: for 1h / 10h ft model , you should set decoding.lmweight=3 to decode. will get best WER%
 fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then
   echo "inference imls-ssl  model on dev-other, dev-clean, test-other, test-clean of librispeech with fairseqlm"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp

   config_dir=$fairseq_dir/examples/hubert/
   model_name=pretrain_on_base_imls-ssl_4gpu_8update_960h_400k_update
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune
   results_path=$exp_finetune_dir/decode_on_100h_with_fairseqlm_normalize_false
   mkdir -p $results_path
   path_to_lexicon=/mntcephfs/lab_data/maduo/datasets/librispeech/kenlm_files/librispeech_lexicon.lst #word2letter, it is upper style dictionary
   path_to_lm=/mntcephfs/lab_data/maduo/datasets/librispeech/fairseqlm_files/lm_librispeech_word_transformer.pt  ## word lm
   #word_dict=/mntcephfs/lab_data/maduo/datasets/librispeech/fairseqlm_files/dict.txt ## lower style word dictionary, first column is word 
                                                                                     ## second column is word frequence.
   testsets="dev-other test-clean test-other"
   #testsets="dev-clean"
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   for name in $testsets;do
       python $fairseq_dir/examples/speech_recognition/new/infer.py \
                --config-dir $fairseq_dir/examples/speech_recognition/new/conf\
                --config-name infer_fsqlm_librispeech\
                task.data=$tsv_dir\
                task.label_dir=$tsv_dir\
                task.normalize=false\
                common_eval.results_path=$results_path\
                common_eval.path=$exp_finetune_dir/checkpoint_best.pt\
                dataset.gen_subset=$name\
                decoding.type=fairseqlm\
                decoding.lexicon=$path_to_lexicon\
                decoding.lmpath=$path_to_lm\
                decoding.nbest=1\
                decoding.beam=500 \
                decoding.lmweight=2 \
                decoding.wordscore=-1 \
                decoding.beamthreshold=25\
                common_eval.quiet=false


   done
   ##WER% with fairseqlm(it is a transformer network lm) to decode
   ## dev-clean  dev-other test-clean test-other   lmweight=2
   ## 2.4448      6.3089    2.6861      6.3696 
   
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ];then
   echo "inference imls-ssl  model on dev-other, dev-clean, test-other, test-clean of librispeech with fairseqlm"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp

   config_dir=$fairseq_dir/examples/hubert/
   model_name=pretrain_on_base_imls-ssl_4gpu_8update_960h_400k_update
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune
   results_path=$exp_finetune_dir/decode_on_100h_with_fairseqlm_normalize_false
   mkdir -p $results_path
   path_to_lexicon=/mntcephfs/lab_data/maduo/datasets/librispeech/kenlm_files/librispeech_lexicon.lst #word2letter, it is upper style dictionary
   path_to_lm=/mntcephfs/lab_data/maduo/datasets/librispeech/fairseqlm_files/lm_librispeech_word_transformer.pt  ## word lm
   #word_dict=/mntcephfs/lab_data/maduo/datasets/librispeech/fairseqlm_files/dict.txt ## lower style word dictionary, first column is word
                                                                                     ## second column is word frequence.
   testsets="dev-clean dev-other test-clean test-other"
   #testsets="dev-clean"
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   for name in $testsets;do
       python $fairseq_dir/examples/speech_recognition/new/infer.py \
                --config-dir $fairseq_dir/examples/speech_recognition/new/conf\
                --config-name infer_fsqlm_librispeech\
                task.data=$tsv_dir\
                task.label_dir=$tsv_dir\
                task.normalize=false\
                common_eval.results_path=$results_path\
                common_eval.path=$exp_finetune_dir/checkpoint_best.pt\
                dataset.gen_subset=$name\
                decoding.type=fairseqlm\
                decoding.lexicon=$path_to_lexicon\
                decoding.lmpath=$path_to_lm\
                decoding.nbest=1\
                decoding.beam=500 \
                decoding.lmweight=3 \
                decoding.wordscore=-1 \
                decoding.beamthreshold=25\
                common_eval.quiet=false


   done
   ##WER% with fairseqlm(it is a transformer network lm) to decode
   ## dev-clean  dev-other  test-clean  test-other   lmweight=3
   ##  grep -rn "Word error rate:" logs/base_imls-ssl_stage6_infer_with_fairseqlm_normalize_false_lmweight3.log
   ##   3.2352    8.1187     3.3538    8.1732

fi

## 2023.6.27
### I want to use 9layer feature of iter2 offical hubert base model and kmeams model to get target code to train imls-ssl model
### compared stage1, I only use iter2 hubert 9layer code instead of iter1 hubert 6layer code
if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ];then
   echo "iter2: pretrain voicelm on 9layer of offical iter2 hubert pesudo label and librispeech monophncode from w2vu2-model "
   echo "training on 400k steps for train-960 of librispeech"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp
   label_dir=$tsv_dir/offical_iter2_hubert_9layer_codes_and_librispeech_frame_monophncode_using_wav2vec-u2_model
   config_dir=$fairseq_dir/examples/voicelm
   model_name=pretrain_on_base_imls-ssl_4gpu_8update_960h_400k_update_iter2_9layer_hubert_model_code
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
            task.labels='["unsupphncode","km"]' \
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
### 4V100: training about 30 day
###           200steps: about 22 minites
fi

if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ];then
   echo "fine tune base imls-ssl model  using train-clean-100 supervision data"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp
   config_dir=$fairseq_dir/examples/voicelm
   model_name=pretrain_on_base_imls-ssl_4gpu_8update_960h_400k_update_iter2_9layer_hubert_model_code
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_finetune_dir
   world_size=4
   update_freq=2
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   python $fairseq_dir/fairseq_cli/hydra_train.py \
       --config-dir $config_dir/config/finetune \
            --config-name voicelm_base_100h \
            task.data=$tsv_dir\
            task.label_dir=$tsv_dir\
            task.labels='["ltr"]' \
            model.w2v_path=$exp_dir/checkpoint_298_400000.pt\
            common.user_dir=$fairseq_dir/examples/voicelm\
   dataset.train_subset=train-clean-100\
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

if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ];then
   echo "inference imls-ssl  model on dev-other, dev-clean, test-other, test-clean of librispeech"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp

   config_dir=$fairseq_dir/examples/hubert/
   model_name=pretrain_on_base_imls-ssl_4gpu_8update_960h_400k_update_iter2_9layer_hubert_model_code
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune
   #results_path=$exp_finetune_dir/decode_on_100h
   results_path=$exp_finetune_dir/decode_on_100h_viterbi_normalize_false
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
   ## grep -rn "Word error rate:" exp/finetune/pretrain_on_base_imls-ssl_4gpu_8update_960h_400k_update_iter2_9layer_hubert_model_code_100h_asr_finetune/decode_on_100h_viterbi_normalize_false/viterbi/infer.log 
#8128:[2023-08-04 15:24:35,411][__main__][INFO] - Word error rate: 4.3344
#16727:[2023-08-04 15:25:40,226][__main__][INFO] - Word error rate: 10.2780
#24591:[2023-08-04 15:26:47,326][__main__][INFO] - Word error rate: 4.4248
#33424:[2023-08-04 15:27:52,894][__main__][INFO] - Word error rate: 10.1009
# so WER%
# dev-clean  dev-other  test-clean  test-other 
# 4.33         10.28       4.42      10.10
fi

if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ];then
   echo "inference imls-ssl  model on dev-other, dev-clean, test-other, test-clean of librispeech with kenlm"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp

   config_dir=$fairseq_dir/examples/hubert/
   model_name=pretrain_on_base_imls-ssl_4gpu_8update_960h_400k_update_iter2_9layer_hubert_model_code
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune
   results_path=$exp_finetune_dir/decode_on_100h_with_kenlm_normalize_false
   mkdir -p $results_path
   path_to_lexicon=/mntcephfs/lab_data/maduo/datasets/librispeech/kenlm_files/librispeech_lexicon.lst #word2letter
   path_to_lm=/mntcephfs/lab_data/maduo/datasets/librispeech/kenlm_files/4-gram.arpa  ## word lm
   testsets="dev-clean dev-other test-clean test-other"
   #testsets="dev-clean"
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
   # WER% in kenlm and task.normalize=false
   # log is from grep -rn "Word error rate"  logs/base_imls-ssl_stage12-14_infer_on_100h.log
   # dev-clean  dev-other test-clean test-other
   # 2.49        6.99       3.05      7.27
   ## NOTE: for 1h / 10h ft model , you should set decoding.lmweight=3 to decode. will get best WER%
   # 
   # 
 fi

if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ];then
   echo "inference imls-ssl  model on dev-other, dev-clean, test-other, test-clean of librispeech with fairseqlm"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp

   config_dir=$fairseq_dir/examples/hubert/
   model_name=pretrain_on_base_imls-ssl_4gpu_8update_960h_400k_update
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune
   results_path=$exp_finetune_dir/decode_on_100h_with_fairseqlm_normalize_false
   mkdir -p $results_path
   path_to_lexicon=/mntcephfs/lab_data/maduo/datasets/librispeech/kenlm_files/librispeech_lexicon.lst #word2letter, it is upper style dictionary
   path_to_lm=/mntcephfs/lab_data/maduo/datasets/librispeech/fairseqlm_files/lm_librispeech_word_transformer.pt  ## word lm
   #word_dict=/mntcephfs/lab_data/maduo/datasets/librispeech/fairseqlm_files/dict.txt ## lower style word dictionary, first column is word
                                                                                     ## second column is word frequence.
   testsets="dev-clean dev-other test-clean test-other"
   #testsets="dev-clean"
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   for name in $testsets;do
       python $fairseq_dir/examples/speech_recognition/new/infer.py \
                --config-dir $fairseq_dir/examples/speech_recognition/new/conf\
                --config-name infer_fsqlm_librispeech\
                task.data=$tsv_dir\
                task.label_dir=$tsv_dir\
                task.normalize=false\
                common_eval.results_path=$results_path\
                common_eval.path=$exp_finetune_dir/checkpoint_best.pt\
                dataset.gen_subset=$name\
                decoding.type=fairseqlm\
                decoding.lexicon=$path_to_lexicon\
                decoding.lmpath=$path_to_lm\
                decoding.nbest=1\
                decoding.beam=500 \
                decoding.lmweight=2 \
                decoding.wordscore=-1 \
                decoding.beamthreshold=25\
                common_eval.quiet=false


   done
   ##WER% with fairseqlm(it is a transformer network lm) to decode
   ## dev-clean  dev-other  test-clean  test-other   lmweight=2
   ##  2.48       6.32       2.60        6.40
   ##  grep -rn "Word error rate:" logs/base_imls-ssl_stage12-14_infer_on_100h.log 
   ## 

fi

# run2023-11-27 for voicelm 
## using half of unpaired text to train wav2vec-u2 model and get librispeech monophncode
if [ ${stage} -le 20 ] && [ ${stop_stage} -ge 20 ];then
   echo "iter: pretrain voicelm on 6layer of hubert pesudo label and librispeech monophncode from w2vu2-model "
   echo "training on 400k steps for train-960 of librispeech"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp
   label_dir=$tsv_dir/offical_hubert_codes_and_librispeech_frame_monophncode_using_wav2vec-u2_model_on_unpaired_text_half
   config_dir=$fairseq_dir/examples/voicelm
   model_name=pretrain_on_base_voicelm_4gpu_8update_960h_400k_update_on_unpaired_text_half
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
            hydra.run.dir=$exp_dir\
            hydra.job.name=$exp_dir/pretrain
### 4V100: training about 30.6 day
###           200steps: about 22 minites

###           200steps: about 23 minites using 4A800
fi
## using half of unpaired text to train wav2vec-u2 model and get librispeech monophncode
if [ ${stage} -le 21 ] && [ ${stop_stage} -ge 21 ];then
   echo "fine tune base voicelm model  using train-clean-100 supervision data"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp
   config_dir=$fairseq_dir/examples/voicelm
   model_name=pretrain_on_base_voicelm_4gpu_8update_960h_400k_update_on_unpaired_text_half
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_finetune_dir
   world_size=4
   update_freq=2
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   python $fairseq_dir/fairseq_cli/hydra_train.py \
       --config-dir $config_dir/config/finetune \
            --config-name voicelm_base_100h \
            task.data=$tsv_dir\
            task.label_dir=$tsv_dir\
            task.labels='["ltr"]' \
            model.w2v_path=$exp_dir/checkpoint_298_400000.pt\
            common.user_dir=$fairseq_dir/examples/voicelm\
            dataset.train_subset=train-clean-100\
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

## using half of unpaired text to train wav2vec-u2 model and get librispeech monophncode
if [ ${stage} -le 22 ] && [ ${stop_stage} -ge 22 ];then
   echo "inference voicelm  model on dev-other, dev-clean, test-other, test-clean of librispeech"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp

   config_dir=$fairseq_dir/examples/hubert/
   model_name=pretrain_on_base_voicelm_4gpu_8update_960h_400k_update_on_unpaired_text_half
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune
   #results_path=$exp_finetune_dir/decode_on_100h
   results_path=$exp_finetune_dir/decode_on_100h_viterbi_normalize_false
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
   # grep -rn 'Word error rate' logs/voicelm_half_text_infer.log
   #8239:[2024-01-09 10:24:35,890][__main__][INFO] - Word error rate: 4.5568
   #16857:[2024-01-09 10:25:44,171][__main__][INFO] - Word error rate: 10.4527
   #24740:[2024-01-09 10:26:56,014][__main__][INFO] - Word error rate: 4.6379
   #33592:[2024-01-09 10:28:10,015][__main__][INFO] - Word error rate: 10.1658
fi

if [ ${stage} -le 23 ] && [ ${stop_stage} -ge 23 ];then
   echo "inference voicelm  model on dev-other, dev-clean, test-other, test-clean of librispeech with kenlm"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp

   config_dir=$fairseq_dir/examples/hubert/
   model_name=pretrain_on_base_voicelm_4gpu_8update_960h_400k_update_on_unpaired_text_half
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune
   #results_path=$exp_finetune_dir/decode_on_100h_with_kenlm_half_normalize_false
   #mkdir -p $results_path
   #path_to_lexicon=/mntcephfs/lab_data/maduo/datasets/librispeech/kenlm_files/librispeech_lexicon.lst #word2letter
   #path_to_lm=/mntcephfs/lab_data/maduo/datasets/librispeech/kenlm_files/4-gram.arpa  ## word lm it is librispeech_lm full text lm.

   path_to_lexicon=/mntcephfs/lab_data/maduo/datasets/librispeech/kenlm_files/librispeech_lexicon.lst #word2letter

   #results_path=$exp_finetune_dir/decode_on_100h_with_kenlm_half_normalize_false
   #path_to_lm=/mntcephfs/lab_data/maduo/datasets/librispeech/kenlm_files/4-gram.arpa  ## word lm, it is librispeech_lm full text lm.

   results_path=$exp_finetune_dir/decode_on_100h_with_kenlm_half_normalize_false
   path_to_lm=/mntcephfs/lab_data/maduo/datasets/librispeech/kenlm_files/kenlm.wrd.o40003_half.bin # word lm, it is trained on half of librispeech_lm full text
   testsets="dev-clean dev-other test-clean test-other"
   #testsets="dev-clean"
   mkdir -p $results_path
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
   # WER% in kenlm and task.normalize=false
   # kenlm is 4-gram.arpa
   #  grep -rn "Word error rate" logs/voicelm_half_text_infer_kenlm.log
   # 8243:[2024-01-09 12:54:01,338][__main__][INFO] - Word error rate: 2.5753
   # 16865:[2024-01-09 14:17:18,970][__main__][INFO] - Word error rate: 7.1431
   # 24752:[2024-01-09 15:43:30,921][__main__][INFO] - Word error rate: 3.0551
   # 33608:[2024-01-09 17:09:46,039][__main__][INFO] - Word error rate: 7.1224
  
   
   # WER% in kenlm and task.normalize=false
   # kenlm is kenlm.wrd.o40003_half.bin
   # log is from grep -rn "Word error rate"
   # dev-clean  dev-other test-clean test-other
   #  grep -rn 'Word error rate' logs/voicelm_half_text_infer_half_kenlm.log
   # 8239:[2024-01-09 19:06:30,680][__main__][INFO] - Word error rate: 2.8161
   # 16857:[2024-01-09 20:15:50,599][__main__][INFO] - Word error rate: 7.3492
   # 24740:[2024-01-09 21:27:18,274][__main__][INFO] - Word error rate: 3.1179
   # 33592:[2024-01-09 22:39:43,373][__main__][INFO] - Word error rate: 7.3077

   ## NOTE: for 1h / 10h ft model , you should set decoding.lmweight=3 to decode. will get best WER%
   #
   #
 fi



 
## using half of unpaired text to train wav2vec-u2 model and get librispeech monophncode
if [ ${stage} -le 24 ] && [ ${stop_stage} -ge 24 ];then
   echo "fine tune base voicelm model  using train-10 supervision data"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp
   config_dir=$fairseq_dir/examples/voicelm
   model_name=pretrain_on_base_voicelm_4gpu_8update_960h_400k_update_on_unpaired_text_half
   exp_finetune_dir=$dir/finetune/${model_name}_10h_asr_finetune
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_finetune_dir
   world_size=2
   update_freq=4
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   python $fairseq_dir/fairseq_cli/hydra_train.py \
       --config-dir $config_dir/config/finetune \
            --config-name voicelm_base_10h \
            task.data=$tsv_dir\
            task.label_dir=$tsv_dir\
            task.labels='["ltr"]' \
            model.w2v_path=$exp_dir/checkpoint_298_400000.pt\
            common.user_dir=$fairseq_dir/examples/voicelm\
            dataset.train_subset=train-10h\
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

## using half of unpaired text to train wav2vec-u2 model and get librispeech monophncode
if [ ${stage} -le 25 ] && [ ${stop_stage} -ge 25 ];then
   echo "inference voicelm  model on dev-other, dev-clean, test-other, test-clean of librispeech"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp

   config_dir=$fairseq_dir/examples/hubert/
   model_name=pretrain_on_base_voicelm_4gpu_8update_960h_400k_update_on_unpaired_text_half
   exp_finetune_dir=$dir/finetune/${model_name}_10h_asr_finetune
   #results_path=$exp_finetune_dir/decode_on_100h
   results_path=$exp_finetune_dir/decode_on_10h_viterbi_normalize_false
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
   # grep -rn 'Word error rate' logs/voicelm_half_text_10h_infer.log
   # 8239:[2024-01-10 09:50:24,606][__main__][INFO] - Word error rate: 8.2258
   # 16857:[2024-01-10 09:51:30,821][__main__][INFO] - Word error rate: 13.1968
   # 24740:[2024-01-10 09:52:44,372][__main__][INFO] - Word error rate: 8.2942
   # 33592:[2024-01-10 09:53:54,106][__main__][INFO] - Word error rate: 13.5589
fi

if [ ${stage} -le 26 ] && [ ${stop_stage} -ge 26 ];then
   echo "inference voicelm  model on dev-other, dev-clean, test-other, test-clean of librispeech with kenlm"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp

   config_dir=$fairseq_dir/examples/hubert/
   model_name=pretrain_on_base_voicelm_4gpu_8update_960h_400k_update_on_unpaired_text_half
   exp_finetune_dir=$dir/finetune/${model_name}_10h_asr_finetune
   results_path=$exp_finetune_dir/decode_on_10h_with_kenlm_normalize_false
   mkdir -p $results_path
   path_to_lexicon=/mntcephfs/lab_data/maduo/datasets/librispeech/kenlm_files/librispeech_lexicon.lst #word2letter
   path_to_lm=/mntcephfs/lab_data/maduo/datasets/librispeech/kenlm_files/4-gram.arpa  ## word lm
   testsets="dev-clean dev-other test-clean test-other"
   #testsets="dev-clean"
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
                decoding.lmweight=3 \
                decoding.wordscore=-1 \
                decoding.beamthreshold=100\
                common_eval.quiet=false

   done
   # WER% in kenlm and task.normalize=false
   # log is from grep -rn "Word error rate"
   # dev-clean  dev-other test-clean test-other
   #  grep -rn 'Word error rate' logs/voicelm_half_text_10h_infer.log
   # 41744:[2024-01-10 11:13:24,100][__main__][INFO] - Word error rate: 3.6690
   # 50366:[2024-01-10 12:29:27,522][__main__][INFO] - Word error rate: 8.2326
   # 58253:[2024-01-10 13:48:47,879][__main__][INFO] - Word error rate: 4.0329
   # 67109:[2024-01-10 15:08:23,205][__main__][INFO] - Word error rate: 8.7501
   ## NOTE: for 1h / 10h ft model , you should set decoding.lmweight=3 to decode. will get best WER%
   #
   #
 fi

if [ ${stage} -le 27 ] && [ ${stop_stage} -ge 27 ];then
   echo "inference voicelm  model on dev-other, dev-clean, test-other, test-clean of librispeech with kenlm"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp

   config_dir=$fairseq_dir/examples/hubert/
   model_name=pretrain_on_base_voicelm_4gpu_8update_960h_400k_update_on_unpaired_text_half
   exp_finetune_dir=$dir/finetune/${model_name}_10h_asr_finetune
   #results_path=$exp_finetune_dir/decode_on_10h_with_kenlm_normalize_false
   
   path_to_lexicon=/mntcephfs/lab_data/maduo/datasets/librispeech/kenlm_files/librispeech_lexicon.lst #word2letter
   #path_to_lm=/mntcephfs/lab_data/maduo/datasets/librispeech/kenlm_files/4-gram.arpa  ## word lm
   results_path=$exp_finetune_dir/decode_on_10h_with_kenlm_half_normalize_false
   path_to_lm=/mntcephfs/lab_data/maduo/datasets/librispeech/kenlm_files/kenlm.wrd.o40003_half.bin # word lm, it is trained on half of librispeech_lm full text
   mkdir -p $results_path
   testsets="dev-clean dev-other test-clean test-other"
   #testsets="dev-clean"
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
                decoding.lmweight=3 \
                decoding.wordscore=-1 \
                decoding.beamthreshold=100\
                common_eval.quiet=false

   done
   # WER% in kenlm and task.normalize=false
   # log is from grep -rn "Word error rate"
   # dev-clean  dev-other test-clean test-other
   # grep -rn 'Word error rate' logs/voicelm_half_text_10h_infer.log 
   # 75257:[2024-01-10 16:25:12,943][__main__][INFO] - Word error rate: 4.2021
   # 83875:[2024-01-10 17:38:04,388][__main__][INFO] - Word error rate: 8.9216
   # 91758:[2024-01-10 18:53:50,464][__main__][INFO] - Word error rate: 4.4781
   # 100610:[2024-01-10 20:10:20,740][__main__][INFO] - Word error rate: 9.1647
   ## NOTE: for 1h / 10h ft model , you should set decoding.lmweight=3 to decode. will get best WER%
   #
   #
 fi

 ## using half of unpaired text to train wav2vec-u2 model and get librispeech monophncode
if [ ${stage} -le 28 ] && [ ${stop_stage} -ge 28 ];then
   echo "fine tune base voicelm model  using train-1h supervision data"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp
   config_dir=$fairseq_dir/examples/voicelm
   model_name=pretrain_on_base_voicelm_4gpu_8update_960h_400k_update_on_unpaired_text_half
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
            model.w2v_path=$exp_dir/checkpoint_298_400000.pt\
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

## using half of unpaired text to train wav2vec-u2 model and get librispeech monophncode
if [ ${stage} -le 29 ] && [ ${stop_stage} -ge 29 ];then
   echo "inference voicelm  model on dev-other, dev-clean, test-other, test-clean of librispeech"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp

   config_dir=$fairseq_dir/examples/hubert/
   model_name=pretrain_on_base_voicelm_4gpu_8update_960h_400k_update_on_unpaired_text_half
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
  # grep -rn 'Word error rate' logs/voicelm_half_text_1h_infer.log
  # 8239:[2024-01-10 16:19:50,450][__main__][INFO] - Word error rate: 25.4053
  # 16857:[2024-01-10 16:20:55,178][__main__][INFO] - Word error rate: 29.9603
  # 24740:[2024-01-10 16:22:03,090][__main__][INFO] - Word error rate: 25.8242
  # 33592:[2024-01-10 16:23:08,935][__main__][INFO] - Word error rate: 29.7085


fi

if [ ${stage} -le 30 ] && [ ${stop_stage} -ge 30 ];then
   echo "inference voicelm  model on dev-other, dev-clean, test-other, test-clean of librispeech with kenlm"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp

   config_dir=$fairseq_dir/examples/hubert/
   model_name=pretrain_on_base_voicelm_4gpu_8update_960h_400k_update_on_unpaired_text_half
   exp_finetune_dir=$dir/finetune/${model_name}_1h_asr_finetune
   results_path=$exp_finetune_dir/decode_on_1h_with_kenlm_normalize_false
   mkdir -p $results_path
   path_to_lexicon=/mntcephfs/lab_data/maduo/datasets/librispeech/kenlm_files/librispeech_lexicon.lst #word2letter
   path_to_lm=/mntcephfs/lab_data/maduo/datasets/librispeech/kenlm_files/4-gram.arpa  ## word lm
   testsets="dev-clean dev-other test-clean test-other"
   #testsets="dev-clean"
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
                decoding.lmweight=3 \
                decoding.wordscore=-1 \
                decoding.beamthreshold=100\
                common_eval.quiet=false

   done
   # WER% in kenlm and task.normalize=false
   # log is from grep -rn "Word error rate"
   # dev-clean  dev-other test-clean test-other
   # grep -rn 'Word error rate' logs/voicelm_half_text_1h_infer.log 
   # 41744:[2024-01-10 17:43:52,519][__main__][INFO] - Word error rate: 6.7350
   # 50366:[2024-01-10 19:03:27,324][__main__][INFO] - Word error rate: 13.7622
   # 58253:[2024-01-10 20:25:48,862][__main__][INFO] - Word error rate: 7.1889
   # 67109:[2024-01-10 21:48:54,647][__main__][INFO] - Word error rate: 14.4893
   ## NOTE: for 1h / 10h ft model , you should set decoding.lmweight=3 to decode. will get best WER%
   #
   #
 fi

 if [ ${stage} -le 31 ] && [ ${stop_stage} -ge 31 ];then
   echo "inference voicelm  model on dev-other, dev-clean, test-other, test-clean of librispeech with kenlm"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp

   config_dir=$fairseq_dir/examples/hubert/
   model_name=pretrain_on_base_voicelm_4gpu_8update_960h_400k_update_on_unpaired_text_half
   exp_finetune_dir=$dir/finetune/${model_name}_1h_asr_finetune
   
   path_to_lexicon=/mntcephfs/lab_data/maduo/datasets/librispeech/kenlm_files/librispeech_lexicon.lst #word2letter
  
   results_path=$exp_finetune_dir/decode_on_1h_with_kenlm_half_normalize_false
   path_to_lm=/mntcephfs/lab_data/maduo/datasets/librispeech/kenlm_files/kenlm.wrd.o40003_half.bin # word lm, it is trained on half of librispeech_lm full text
   mkdir -p $results_path
   testsets="dev-clean dev-other test-clean test-other"
   #testsets="dev-clean"
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
                decoding.lmweight=3 \
                decoding.wordscore=-1 \
                decoding.beamthreshold=100\
                common_eval.quiet=false

   done
   # WER% in kenlm and task.normalize=false
   # log is from grep -rn "Word error rate"
   # dev-clean  dev-other test-clean test-other
   # grep -rn 'Word error rate' logs/voicelm_half_text_1h_infer.log
   #  75257:[2024-01-10 23:09:56,569][__main__][INFO] - Word error rate: 8.3471
   # 83875:[2024-01-11 00:28:03,071][__main__][INFO] - Word error rate: 15.7565
   # 91758:[2024-01-11 01:48:35,653][__main__][INFO] - Word error rate: 8.5795
   # 100610:[2024-01-11 03:09:57,335][__main__][INFO] - Word error rate: 16.2871
   ## NOTE: for 1h / 10h ft model , you should set decoding.lmweight=3 to decode. will get best WER%
   #
   #
 fi
# (TODO) 
## this pretrain model is trained from xianghu.
 ## using half of unpaired text to train wav2vec-u2 model and get librispeech monophncode
if [ ${stage} -le 32 ] && [ ${stop_stage} -ge 32 ];then
   echo "fine tune base voicelm model  using train-1h supervision data"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntnfs/lee_data1/maduo/exp
   config_dir=$fairseq_dir/examples/voicelm
   model_name=pretrain_on_base_voicelm_4gpu_8update_960h_400k_update_on_unpaired_text_half
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
            model.w2v_path=$exp_dir/checkpoint_298_400000.pt\
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

