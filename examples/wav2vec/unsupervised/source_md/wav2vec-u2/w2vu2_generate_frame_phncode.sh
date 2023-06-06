#!/usr/bin/env bash

stage=0
stop_stage=1000
. utils/parse_options.sh
. path_for_fairseq_speechtext.sh

export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
if [ ${stage} -le 0 ]&& [ ${stop_stage} -ge 0 ];then
   echo "decode  dev-clean dev-other test-clean test-other using wav2vec-u2.0 model and compute PER "
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   feat_dir=$tsv_dir/wav2vec_large_feat_dir
   cp -r $tsv_dir/librispeech_lm_norm_phn_seq/phoness/dict.phn.txt $feat_dir
   config_dir=$fairseq_dir/examples/wav2vec/
   dir=/workspace2/maduo/exp   
   wav2vec_u2_dir=/workspace2/maduo/model_hub/librispeech/wav2vec-u2.0-trained_model_using_librispeech_libirspeech_lm_text
   dest_dir=$dir/wav2vec-u2/hyps
   mkdir -p $dest_dir
   testsets="dev-clean dev-other test-clean test-other"
   #testsets="test-other"
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH 
   for name in $testsets;do
       cp -r $tsv_dir/${name}.wrd $feat_dir
       cp -r $tsv_dir/${name}.phn $feat_dir
       python $fairseq_dir/examples/wav2vec/unsupervised/w2vu_generate.py\
               --config-dir $config_dir/unsupervised/config/generate \
               --config-name viterbi \
               fairseq.common.user_dir=${fairseq_dir}/examples/wav2vec/unsupervised \
               fairseq.task.data=$feat_dir \
               fairseq.common_eval.path=$wav2vec_u2_dir/checkpoint_24_41000.pt \
               fairseq.common_eval.quiet=true \
               fairseq.dataset.gen_subset=$name\
               results_path=$dest_dir\
               decode_stride=2
   done
    ## PER% its audio with silence, it means that the audio is not applied vad.
    #  dev-clean  dev-other  test-clean test-other
    #   7.3       10.28        7.45      10.22
    ## PER% its audio not silence, it means that the audio is applied vad, its result at the source_md/wav2vec-u2/wav2vec-u2_from_scratch.sh --stage 8 --stop-stage 8 
    ## 8.1928     11.01        8.35      11.18 
fi


## 2023.5.17 the above way (decode librispeech dataset from pretrain wav2vec-u2 model) is wrong.
## why, because its decoder phnoneme sequence doesn't four specify symbols(dictionary bos index: 0, dictionary pad index: 1, dictionary eos index: 2, dictionary unk index: 3)
## the four symbols are auto adding the dictionary on the pretraining via fairseq framework. so, the trainset don't the above four symbols. it should be wrong at prepare trainset stage.
## in other words. At decoding librispeech dataset stage, it should directly output phoneme id sequence (at this stage, its contain four specify symbols),not covert phoneme id into phoneme(at this stage, it will remove four specify symbols)
## the below we will directly output phoneme id sequence from pretrain wav2vec-u2 model.
if [ ${stage} -le 20 ]&& [ ${stop_stage} -ge 20 ];then
   echo "decode dev-clean dev-other train-960 using wav2vec-u2.0 model to get frame level phonemem id sequence "
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   feat_dir=$tsv_dir/wav2vec_large_feat_dir
   cp -r $tsv_dir/librispeech_lm_norm_phn_seq/phoness/dict.phn.txt $feat_dir
   config_dir=$fairseq_dir/examples/wav2vec/
   dir=/workspace2/maduo/exp
   wav2vec_u2_dir=/workspace2/maduo/model_hub/librispeech/wav2vec-u2.0-trained_model_using_librispeech_libirspeech_lm_text
   dest_dir=$dir/wav2vec-u2/frame_phonecode_hyps_newer
   mkdir -p $dest_dir
   testsets="dev-clean dev-other train-960"
   #testsets="dev-clean"
   #testsets="test-other"
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   for name in $testsets;do
       cp -r $tsv_dir/${name}.wrd $feat_dir
       cp -r $tsv_dir/${name}.phn $feat_dir
       python $fairseq_dir/examples/wav2vec/unsupervised/w2vu_generate_frame_phncode.py\
               --config-dir $config_dir/unsupervised/config/generate \
               --config-name viterbi_frame \
               fairseq.common.user_dir=${fairseq_dir}/examples/wav2vec/unsupervised \
               fairseq.task.data=$feat_dir \
               fairseq.common_eval.path=$wav2vec_u2_dir/checkpoint_24_41000.pt \
               fairseq.common_eval.quiet=false \
               fairseq.dataset.gen_subset=$name\
               results_path=$dest_dir\
               decode_stride=1
   done
fi



## in order to compare different number of unpaired text, I will use half of unpaired text to train wav2vec-u2 model, then generate dev-clean dev-other train phoneseq
if [ ${stage} -le 40 ]&& [ ${stop_stage} -ge 40 ];then
   echo "decode dev-clean dev-other train-960 using wav2vec-u2.0 model to get frame level phonemem id sequence "
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   feat_dir=$tsv_dir/wav2vec_large_feat_dir
   cp -r $tsv_dir/librispeech_lm_norm_phn_seq/phoness/dict.phn.txt $feat_dir
   config_dir=$fairseq_dir/examples/wav2vec/
   dir=/workspace2/maduo/exp
   #wav2vec_u2_dir=/workspace2/maduo/model_hub/librispeech/wav2vec-u2.0-trained_model_using_librispeech_libirspeech_lm_text
   #dest_dir=$dir/wav2vec-u2/frame_phonecode_hyps_newer
   wav2vec_u2_dir=$dir/wav2vec-u2_gan_from_scratch/w2v_unsup_gan_xp_4gpu_8update_with_unpair_text_half
   dest_dir=$wav2vec_u2_dir/frame_phonecode_hyps_newer_using_half_text_wav2vec-u2
   mkdir -p $dest_dir
   testsets="dev-clean dev-other train-960"
   #testsets="dev-clean"
   #testsets="test-other"
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   for name in $testsets;do
       cp -r $tsv_dir/${name}.wrd $feat_dir
       cp -r $tsv_dir/${name}.phn $feat_dir
       python $fairseq_dir/examples/wav2vec/unsupervised/w2vu_generate_frame_phncode.py\
               --config-dir $config_dir/unsupervised/config/generate \
               --config-name viterbi_frame \
               fairseq.common.user_dir=${fairseq_dir}/examples/wav2vec/unsupervised \
               fairseq.task.data=$feat_dir \
               fairseq.common_eval.path=$wav2vec_u2_dir/checkpoint_last.pt \
               fairseq.common_eval.quiet=false \
               fairseq.dataset.gen_subset=$name\
               results_path=$dest_dir\
               decode_stride=1
   done
fi
