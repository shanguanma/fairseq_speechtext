#!/bin/bash

stage=0

stop_stage=1000
.  utils/parse_options.sh
. path_for_fairseq_speechtext.sh

export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
    echo "inference voicelm(only phn) model on dev-other, dev-clean, test-other, test-clean of librispeech"
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
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
