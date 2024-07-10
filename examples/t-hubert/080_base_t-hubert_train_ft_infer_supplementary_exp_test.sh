#!/bin/bash
stage=0

stop_stage=1000
.  utils/parse_options.sh
. path_for_fairseq_speechtext.sh ## pytorch 2.0 fairseq flashnight-text flashnight sequence
#. path_for_fsq_speechtext.sh
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

 if [ ${stage} -le 32 ] && [ ${stop_stage} -ge 32 ];then
   echo "inference t-hubert  model on test-other, test-clean of librispeech"
   ## This recipe is using cpu mode to infer testset via setting `common.cpu=true`
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntcephfs/lab_data/maduo/exp/

   #config_dir=$fairseq_dir/examples/hubert/
   #model_name=pretrain_on_base_t-hubert_4gpu_8update_960h_400k_update
   model_name=pretrain_on_base_imls-ssl_4gpu_8update_960h_400k_update # i rename imls-ssl to t-hubert.
   exp_finetune_dir=$dir/finetune/${model_name}_360h_asr_finetune
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
                dataset.gen_subset=$name\
                common.cpu=true

   done
   #  grep -rn 'Word error rate:' logs/080_base_t-hubert_train_ft_infer_supplementary_exp_test_stage32.log
   # WER%
   # test-clean test-other
   #  3.8332       9.1896
fi

if [ ${stage} -le 33 ] && [ ${stop_stage} -ge 33 ];then
   echo "inference t-hubert  model on test-other, test-clean of librispeech"
   ## This recipe is using cpu mode to infer testset via setting `common.cpu=true`
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   tsv_dir=/mntcephfs/lab_data/maduo/datasets/format/librispeech/
   dir=/mntcephfs/lab_data/maduo/exp/

   #config_dir=$fairseq_dir/examples/hubert/
   #model_name=pretrain_on_base_t-hubert_4gpu_8update_960h_400k_update
   model_name=pretrain_on_base_imls-ssl_4gpu_8update_960h_400k_update # i rename imls-ssl to t-hubert.
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
                dataset.gen_subset=$name\
                common.cpu=true

   done
#  grep -rn 'Word error rate' logs/080_base_t-hubert_train_ft_infer_supplementary_exp_test_stage33.log
#7880:[2024-07-07 13:03:23,199][__main__][INFO] - Word error rate: 3.7647
#16728:[2024-07-07 13:18:10,635][__main__][INFO] - Word error rate: 9.1284

fi
