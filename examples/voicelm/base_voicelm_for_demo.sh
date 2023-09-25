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



   results_path=$exp_finetune_dir/decode_on_100h_normalize_false
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
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   #config_dir=/workspace2/maduo/source_md/wav2vec-u2
   config_dir=$fairseq_dir/examples/voicelm/
   dir=/workspace2/maduo/exp
   model_name=pretrain_on_base_imls-ssl_4gpu_8update_960h_400k_update
   exp_finetune_dir=$dir/finetune/${model_name}_100h_asr_finetune
   mkdir -p $exp_finetune_dir/decode_on_100h_debug
   testsets="test-clean"
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   #cd $fairseq_dir
   for name in $testsets;do
   CUDA_VISIBLE_DEVICES=7       python $fairseq_dir/examples/speech_recognition/new/infer.py \
                --config-dir $config_dir/config/decode\
                --config-name infer_viterbi_librispeech\
                task.data=$tsv_dir\
                task.label_dir=$tsv_dir\
                task.normalize=true\
                common_eval.results_path=$exp_finetune_dir/decode_on_100h_debug\
                common_eval.path=$exp_finetune_dir/checkpoint_best_new.pt\
                dataset.gen_subset=$name

   done


fi

