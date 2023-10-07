#!/usr/bin/env bash
  
stage=0

stop_stage=1000
.  utils/parse_options.sh
#. path_for_fairseq_speechtext.sh ## pytorch 2.0 fairseq flashnight-text flashnight sequence 
. path_for_fsq_speechtext.sh
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

### voicelm paper: (VoiceLM(only phn))
if [ ${stage} -le 20 ] && [ ${stop_stage} -ge 20 ];then

   echo "pretrain hubert on wav2vec-u2.0 15 layer pesudo label, label_rate=50"
   echo "training on 400k steps for train-960 of librispeech unpair speech"
   echo "called by voicelm(only phn)"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   label_dir=$tsv_dir/librispeech_lm_monophncode_using_monophn_dict_librispeech_frame_monophncode_using_wav2vec-u2_model # ##postfix *.speechphncode *.textphncode files folder
   #tsv_dir=$label_dir
   config_dir=$fairseq_dir/examples/sthubert/
   dir=/workspace2/maduo/exp
   model_name=pretrain_on_base_sthubert_4gpu_8update_400k_w2vu2_librispeech_monophncode
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_dir
   world_size=4
   update_freq=8
   CUDA_VISIBLE_DEVICES=0,5,6,7 python $fairseq_dir/fairseq_cli/hydra_train.py \
            --config-dir $config_dir/config/pretrain \
            --config-name hubert_base_librispeech\
            task.data=$tsv_dir\
            task.label_dir=$label_dir\
            task.labels='["speechphncode"]' \
            model.label_rate=50\
            common.user_dir=$fairseq_dir/examples/sthubert\
            dataset.train_subset=train-960\
            dataset.valid_subset=\'dev-other,dev-clean\'\
            dataset.max_tokens=1400000\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_dir\
            checkpoint.save_dir=$exp_dir\
            hydra.run.dir=$fairseq_dir/examples/sthubert\
            hydra.job.name=$exp_dir/pretrain
fi

## check pretrain model name
if [ ${stage} -le 21 ] && [ ${stop_stage} -ge 21 ];then
   echo "fine tune voicelm(only phn) model using train-clean-100 supervision data"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   #config_dir=source_md/wav2vec-u2/
   config_dir=$fairseq_dir/examples/sthubert/
   dir=/workspace2/maduo/exp
   model_name=pretrain_on_base_sthubert_4gpu_8update_400k_w2vu2_librispeech_monophncode
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_finetune_dir
   world_size=4
   update_freq=2
   CUDA_VISIBLE_DEVICES=4,5,6,7 python $fairseq_dir/fairseq_cli/hydra_train.py \
            --config-dir $config_dir/config/finetune \
            --config-name base_100h_for_hubert \
            task.data=$tsv_dir\
            task.label_dir=$tsv_dir\
            task.labels='["ltr"]' \
            model.w2v_path=$exp_dir/checkpoint_289_400000.pt\
            common.user_dir=$fairseq_dir/examples/sthubert\
            dataset.train_subset=train-clean-100\
            dataset.valid_subset=dev-other\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_finetune_dir\
            checkpoint.save_dir=$exp_finetune_dir\
            hydra.run.dir=$fairseq_dir/examples/sthubert\
            hydra.job.name=$exp_finetune_dir/finetune
fi

if [ ${stage} -le 22 ] && [ ${stop_stage} -ge 22 ];then
    echo "inference voicelm(only phn) model on dev-other, dev-clean, test-other, test-clean of librispeech"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   #config_dir=/workspace2/maduo/source_md/wav2vec-u2
   config_dir=$fairseq_dir/examples/sthubert/
   dir=/workspace2/maduo/exp
   model_name=pretrain_on_base_sthubert_4gpu_8update_400k_w2vu2_librispeech_monophncode
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune
   mkdir -p $exp_finetune_dir/decode_on_100h_test
   testsets="dev-clean dev-other test-clean test-other"
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   #cd $fairseq_dir
   for name in $testsets;do
   CUDA_VISIBLE_DEVICES=7       python $fairseq_dir/examples/speech_recognition/new/infer.py \
                --config-dir $config_dir/config/decode\
                --config-name infer_viterbi_librispeech\
                task.data=$tsv_dir\
                task.label_dir=$tsv_dir\
                task.normalize=true\
                common_eval.results_path=$exp_finetune_dir/decode_on_100h\
                common_eval.path=$exp_finetune_dir/checkpoint_best.pt\
                dataset.gen_subset=$name

   done
   #result: mfcc iter:400k@372epochs, finetune:80k@222epchs
   # ## without lm
   # WER% is from $exp_finetune_dir/decode_on_100/viterbi/infer.log
   #  dev-clean   dev-other   test-clean   test-other
   #  4.8454       11.4734     4.9613        11.2052
   # with 4-gram lm



fi


## 2023-9.26  I want to finetune voicelm on 100h , using speechlm tunning parameter.

if [ ${stage} -le 30 ] && [ ${stop_stage} -ge 30 ];then
   echo "fine tune base imls-ssl model  using train-clean-100 supervision data"
   #fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   #tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   #dir=/mntnfs/lee_data1/maduo/exp
   #config_dir=$fairseq_dir/examples/voicelm
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   config_dir=$fairseq_dir/examples/voicelm
   dir=/workspace2/maduo/exp
   model_name=pretrain_on_base_imls-ssl_4gpu_8update_960h_400k_update
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune_30ksteps_bz1.6m_lr1e_5
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_finetune_dir
   world_size=4
   update_freq=2
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   CUDA_VISIBLE_DEVICES=4,5,6,7  python $fairseq_dir/fairseq_cli/hydra_train.py \
            --config-dir $config_dir/config/finetune \
            --config-name  voicelm_base_100h_30ksteps_bz1.6m_lr1e_5\
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



if [ ${stage} -le 31 ] && [ ${stop_stage} -ge 31 ];then
   echo "inference imls-ssl  model on dev-other, dev-clean, test-other, test-clean of librispeech"
   #fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   #tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   #dir=/mntnfs/lee_data1/maduo/exp

   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   config_dir=$fairseq_dir/examples/voicelm
   dir=/workspace2/maduo/exp

   config_dir=$fairseq_dir/examples/hubert/
   model_name=pretrain_on_base_imls-ssl_4gpu_8update_960h_400k_update
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune_30ksteps_bz1.6m_lr1e_5
   #results_path=$exp_finetune_dir/decode_on_100h
   results_path=$exp_finetune_dir/decode_on_100h_normalize_false
   mkdir -p $results_path
   testsets="dev-clean dev-other test-clean test-other"
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH

   for name in $testsets;do
     CUDA_VISIBLE_DEVICES=7  python $fairseq_dir/examples/speech_recognition/new/infer.py \
                --config-dir $config_dir/config/decode\
                --config-name infer_viterbi_librispeech\
                task.data=$tsv_dir\
                task.label_dir=$tsv_dir\
                task.normalize=false\
                common_eval.results_path=$results_path\
                common_eval.path=$exp_finetune_dir/checkpoint_best.pt\
                dataset.gen_subset=$name

   done
   # grep -rn "Word error rate" exp/finetune/pretrain_on_base_imls-ssl_4gpu_8update_960h_400k_update_100h_asr_finetune_30ksteps_bz1.6m_lr1e_5/decode_on_100h_normalize_false/viterbi/infer.log
   # dev-clean dev-other test-clean test-other
   # 6.3711    11.8326    6.6810     11.7095
fi

if [ ${stage} -le 32 ] && [ ${stop_stage} -ge 32 ];then
   echo "inference imls-ssl  model on dev-other, dev-clean, test-other, test-clean of librispeech with kenlm"
   #fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   #tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   #dir=/mntnfs/lee_data1/maduo/exp
   #config_dir=$fairseq_dir/examples/voicelm/
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   config_dir=$fairseq_dir/examples/voicelm
   dir=/workspace2/maduo/exp

   model_name=pretrain_on_base_imls-ssl_4gpu_8update_960h_400k_update
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune_30ksteps_bz1.6m_lr1e_5
   results_path=$exp_finetune_dir/decode_on_100h_with_kenlm
   mkdir -p $results_path
   path_to_lexicon=/workspace2/maduo/dataset/librispeech/kenlm_files/librispeech_lexicon.lst #word2letter
   path_to_lm=/workspace2/maduo/dataset/librispeech/kenlm_files/4-gram.arpa  ## word lm
   testsets="dev-clean dev-other test-clean test-other"
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH

   for name in $testsets;do
      CUDA_VISIBLE_DEVICES=7 python $fairseq_dir/examples/speech_recognition/new/infer.py \
                --config-dir $config_dir/config/decode\
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
  #  grep -rn "Word error rate" exp/finetune/pretrain_on_base_imls-ssl_4gpu_8update_960h_400k_update_100h_asr_finetune_30ksteps_bz1.6m_lr1e_5/decode_on_100h_with_kenlm/*/infer.log
  # dev-clean dev-other test-clean test-other
  #  2.8216
fi

if [ ${stage} -le 33 ] && [ ${stop_stage} -ge 33 ];then
   echo "fine tune base imls-ssl model  using train-clean-100 supervision data"
   #fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   #tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   #dir=/mntnfs/lee_data1/maduo/exp
   #config_dir=$fairseq_dir/examples/voicelm
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   config_dir=$fairseq_dir/examples/voicelm
   dir=/workspace2/maduo/exp
   model_name=pretrain_on_base_imls-ssl_4gpu_8update_960h_400k_update
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune_80ksteps_bz4m_lr4e_5
   exp_dir=$dir/pretrain/${model_name}
   mkdir -p $exp_finetune_dir
   world_size=4
   update_freq=2
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   CUDA_VISIBLE_DEVICES=4,5,6,7  python $fairseq_dir/fairseq_cli/hydra_train.py \
            --config-dir $config_dir/config/finetune \
            --config-name  voicelm_base_100h_80ksteps_bz4m_lr4e_5\
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

if [ ${stage} -le 34 ] && [ ${stop_stage} -ge 34 ];then
   echo "inference imls-ssl  model on dev-other, dev-clean, test-other, test-clean of librispeech"
   #fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   #tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   #dir=/mntnfs/lee_data1/maduo/exp

   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   config_dir=$fairseq_dir/examples/voicelm
   dir=/workspace2/maduo/exp

   config_dir=$fairseq_dir/examples/hubert/
   model_name=pretrain_on_base_imls-ssl_4gpu_8update_960h_400k_update
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune_80ksteps_bz4m_lr4e_5
   #results_path=$exp_finetune_dir/decode_on_100h
   results_path=$exp_finetune_dir/decode_on_100h_normalize_false
   mkdir -p $results_path
   testsets="dev-clean dev-other test-clean test-other"
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH

   for name in $testsets;do
     CUDA_VISIBLE_DEVICES=7  python $fairseq_dir/examples/speech_recognition/new/infer.py \
                --config-dir $config_dir/config/decode\
                --config-name infer_viterbi_librispeech\
                task.data=$tsv_dir\
                task.label_dir=$tsv_dir\
                task.normalize=false\
                common_eval.results_path=$results_path\
                common_eval.path=$exp_finetune_dir/checkpoint_best.pt\
                dataset.gen_subset=$name

   done
   #grep -rn "Word error rate" logs/base_voicelm_local_cluster_ft_new_config2_on_100h.log
   # 12151:[2023-09-28 14:27:55,099][__main__][INFO] - Word error rate: 4.6487
   # 20770:[2023-09-28 14:29:03,640][__main__][INFO] - Word error rate: 10.4723
   # 28654:[2023-09-28 14:30:14,578][__main__][INFO] - Word error rate: 4.6759
   # 37507:[2023-09-28 14:31:24,727][__main__][INFO] - Word error rate: 10.2289   
fi
if [ ${stage} -le 35 ] && [ ${stop_stage} -ge 35 ];then
   echo "inference imls-ssl  model on dev-other, dev-clean, test-other, test-clean of librispeech with kenlm"
   #fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   #tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   #dir=/mntnfs/lee_data1/maduo/exp
   #config_dir=$fairseq_dir/examples/voicelm/
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   config_dir=$fairseq_dir/examples/voicelm
   dir=/workspace2/maduo/exp

   model_name=pretrain_on_base_imls-ssl_4gpu_8update_960h_400k_update
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune_80ksteps_bz4m_lr4e_5
   results_path=$exp_finetune_dir/decode_on_100h_with_kenlm
   mkdir -p $results_path
   path_to_lexicon=/workspace2/maduo/dataset/librispeech/kenlm_files/librispeech_lexicon.lst #word2letter
   path_to_lm=/workspace2/maduo/dataset/librispeech/kenlm_files/4-gram.arpa  ## word lm
   testsets="dev-clean dev-other test-clean test-other"
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH

   for name in $testsets;do
      CUDA_VISIBLE_DEVICES=7 python $fairseq_dir/examples/speech_recognition/new/infer.py \
                --config-dir $config_dir/config/decode\
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
