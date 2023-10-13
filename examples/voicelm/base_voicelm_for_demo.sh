#!/usr/bin/env bash
  
stage=0

stop_stage=1000
.  utils/parse_options.sh
#. path_for_fairseq_speechtext.sh ## pytorch 2.0 fairseq flashnight-text flashnight sequence 
. path_for_fsq_speechtext.sh
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1


## for demo page, with lm model to decoding
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
   echo "inference voicelm  model on test-clean of librispeech"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   config_dir=$fairseq_dir/examples/voicelm/
   model_name=pretrain_on_base_imls-ssl_4gpu_8update_960h_400k_update
   exp_finetune_dir=/workspace2/maduo/exp/finetune/${model_name}_100h_asr_finetune

   results_path=$exp_finetune_dir/decode_on_100h_normalize_false_for_demo
   path_to_lexicon=/workspace2/maduo/dataset/librispeech/kenlm_files/librispeech_lexicon.lst #word2letter
   path_to_lm=/workspace2/maduo/dataset/librispeech/kenlm_files/4-gram.arpa  ## word lm
   mkdir -p $results_path
   testsets="test-clean"
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   for name in $testsets;do
        CUDA_VISIBLE_DEVICES=7 python $fairseq_dir/examples/speech_recognition/new/infer.py \
                --config-dir $config_dir/config/decode\
                --config-name infer_kenlm_lirispeech\
                task.data=$tsv_dir\
                task.label_dir=$tsv_dir\
                task.normalize=false\
                common_eval.results_path=$results_path\
                common_eval.path=$exp_finetune_dir/checkpoint_best_new.pt\
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
## without lm to decode
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
    echo "inference voicelm(only phn) model on dev-other, dev-clean, test-other, test-clean of librispeech"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   #sv_dir=/workspace2/maduo/dataset/format/librispeech
   tsv_dir=/workspace2/maduo/tests
   #config_dir=/workspace2/maduo/source_md/wav2vec-u2
   config_dir=$fairseq_dir/examples/voicelm/
   dir=/workspace2/maduo/exp
   model_name=pretrain_on_base_imls-ssl_4gpu_8update_960h_400k_update
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune
   results_path=$exp_finetune_dir/decode_on_100h_debug_for_demo_test
   mkdir -p $results_path
   testsets="test-clean10"
   #testsets="dev-clean10"
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   #cd $fairseq_dir
   for name in $testsets;do
   CUDA_VISIBLE_DEVICES=4       python $fairseq_dir/examples/speech_recognition/new/infer_md.py \
                --config-dir $config_dir/config/decode\
                --config-name infer_viterbi_librispeech_10utt\
                task.data=$tsv_dir\
                task.label_dir=$tsv_dir\
                task.normalize=true\
                common_eval.quiet=false\
                common_eval.results_path=$results_path\
                common_eval.path=$exp_finetune_dir/checkpoint_best_new.pt\
                dataset.gen_subset=$name

   done
fi

## without lm to decode
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
    echo "inference voicelm(only phn) model on dev-other, dev-clean, test-other, test-clean of librispeech"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   #sv_dir=/workspace2/maduo/dataset/format/librispeech
   tsv_dir=/workspace2/maduo/tests
   #tsv_dir=/workspace2/maduo/tests/tests
   #config_dir=/workspace2/maduo/source_md/wav2vec-u2
   config_dir=$fairseq_dir/examples/voicelm/
   dir=/workspace2/maduo/exp
   model_name=pretrain_on_base_imls-ssl_4gpu_8update_960h_400k_update
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune
   results_path=$exp_finetune_dir/decode_on_100h_debug_for_demo_test
   dict_path=/workspace2/maduo/dataset/format/librispeech/dict.ltr.txt
   mkdir -p $results_path
   testsets="test-clean10"
   #testsets="test-clean1"
   #testsets="dev-clean10"
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   #cd $fairseq_dir
   for name in $testsets;do
    CUDA_VISIBLE_DEVICES=4     python $fairseq_dir/examples/speech_recognition/new/infer_simple.py\
          $tsv_dir --task audio_pretraining\
          --nbest 1 --path $exp_finetune_dir/checkpoint_best_new.pt\
          --gen-subset ${name}\
          --results-path ${results_path} \
          --w2l-decoder viterbi \
          --word-score -1 --sil-weight 0 \
          --criterion ctc --max-tokens 1100000\
          --lexicon $dict_path \
          --label_dir $tsv_dir\
          --post-process letter 
   done

fi
## for baseline system(i.e: hubert)
## it doesn't finetune 100h model, so I finetune offical hubert base model.

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then
   echo "fine tune base hubert model  using train-clean-100 supervision data"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   config_dir=$fairseq_dir/examples/hubert
   model_name=pretrain_on_base_imls-ssl_4gpu_8update_960h_400k_update

   dir=/workspace2/maduo
   model_name=hubert_base_librispeech_offical_no_finetune
   exp_finetune_dir=$dir/exp/finetune/${model_name}_100h_asr_finetune
   exp_dir=$dir/model_hub/librispeech/${model_name}
   mkdir -p $exp_finetune_dir
   world_size=4
   update_freq=2
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   CUDA_VISIBLE_DEVICES=4,5,6,7 python $fairseq_dir/fairseq_cli/hydra_train.py \
            --config-dir $config_dir/config/finetune \
            --config-name base_100h \
            task.data=$tsv_dir\
            task.label_dir=$tsv_dir\
            task.labels='["ltr"]' \
            model.w2v_path=$exp_dir/hubert_base_ls960.pt\
            common.user_dir=$fairseq_dir/examples/voicelm\
            dataset.train_subset=train-clean-100\
            dataset.valid_subset=dev-other\
            distributed_training.distributed_world_size=${world_size}\
            distributed_training.distributed_port=-1\
            distributed_training.ddp_backend=legacy_ddp\
            optimization.update_freq=[${update_freq}]\
            common.tensorboard_logdir=$exp_finetune_dir\
            checkpoint.save_dir=$exp_finetune_dir\
            hydra.run.dir=$fairseq_dir/examples/hubert\
            hydra.job.name=$exp_finetune_dir/finetune
fi

## without lm to decode
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ];then
    echo "inference offical hubert base  model for test-clean10 of librispeech"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   #sv_dir=/workspace2/maduo/dataset/format/librispeech
   tsv_dir=/workspace2/maduo/tests
   config_dir=$fairseq_dir/examples/hubert

   dir=/workspace2/maduo/exp
   model_name=hubert_base_librispeech_offical_no_finetune
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune
   results_path=$exp_finetune_dir/decode_on_100h_debug_for_demo_test
   dict_path=/workspace2/maduo/dataset/format/librispeech/dict.ltr.txt

   mkdir -p $results_path
   testsets="test-clean10"
   #testsets="test-clean1"
   #testsets="dev-clean10"
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   #cd $fairseq_dir
   for name in $testsets;do
    CUDA_VISIBLE_DEVICES=4     python $fairseq_dir/examples/speech_recognition/new/infer_simple.py\
          $tsv_dir --task audio_pretraining\
          --nbest 1 --path $exp_finetune_dir/checkpoint_best.pt\
          --gen-subset ${name}\
          --results-path ${results_path} \
          --w2l-decoder viterbi \
          --word-score -1 --sil-weight 0 \
          --criterion ctc --max-tokens 1100000\
          --lexicon $dict_path \
          --label_dir $tsv_dir\
          --post-process letter
   done
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ];then
   echo "inference offical hubert base  model on test-clean of librispeech"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   config_dir=$fairseq_dir/examples/voicelm   

   model_name=hubert_base_librispeech_offical_no_finetune
   exp_finetune_dir=/workspace2/maduo/exp/finetune/${model_name}_100h_asr_finetune

   results_path=$exp_finetune_dir/decode_on_100h_normalize_false_for_demo
   path_to_lexicon=/workspace2/maduo/dataset/librispeech/kenlm_files/librispeech_lexicon.lst #word2letter
   path_to_lm=/workspace2/maduo/dataset/librispeech/kenlm_files/4-gram.arpa  ## word lm
   mkdir -p $results_path
   testsets="test-clean"
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   for name in $testsets;do
        CUDA_VISIBLE_DEVICES=7 python $fairseq_dir/examples/speech_recognition/new/infer.py \
                --config-dir $config_dir/config/decode\
                --config-name infer_kenlm_lirispeech\
                task.data=$tsv_dir\
                task.label_dir=$tsv_dir\
                task.normalize=false\
                common_eval.results_path=$results_path\
                common_eval.path=$exp_finetune_dir/checkpoint_best_new.pt\
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
