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
   ## dict.phn.txt contain two columns ,first column is monophone(total is 41 mono phones, last monophone is <SIL>), second column is index(0-base),
   cp -r $tsv_dir/librispeech_lm_norm_phn_seq/phoness/dict.phn.txt $feat_dir 
   config_dir=$fairseq_dir/examples/wav2vec/
   dir=/workspace2/maduo/exp   
   wav2vec_u2_dir=/workspace2/maduo/model_hub/librispeech/wav2vec-u2.0-trained_model_using_librispeech_libirspeech_lm_text
   dest_dir=$dir/wav2vec-u2/hyps
   mkdir -p $dest_dir
   testsets="dev-clean dev-other test-clean test-other"
   testsets="dev-clean"
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
    ## PER%
    #  dev-clean  dev-other  test-clean test-other
    #   7.3       10.28        7.45      10.22

fi

### 
if [ ${stage} -le 2 ]&& [ ${stop_stage} -ge 2 ];then
   echo "decode  dev-clean dev-other test-clean test-other using wav2vec-u2.0 model and compute PER "
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   root_dir=/workspace2/maduo/dataset/format/librispeech
   tsv_dir=$root_dir ### it contains *.wrd and *.phn, because it doesn't matter whether the wav contains silence or not
   feat_dir=$root_dir/wav2vec_large_feat_dir
   ## dict.phn.txt contain two columns ,first column is monophone(total is 41 mono phones, last monophone is <SIL>), second column is index(0-base),
   cp -r $root_dir/librispeech_lm_norm_phn_seq/phoness/dict.phn.txt $feat_dir
   config_dir=$fairseq_dir/examples/wav2vec/
   dir=/workspace2/maduo/exp/
   #model_name=w2v_unsup_gan_xp_4gpu_8update
   #exp_dir=$dir/wav2vec-u2_gan_from_scratch/${model_name}
   wav2vec_u2_dir=/workspace2/maduo/model_hub/librispeech/wav2vec-u2.0-trained_model_using_librispeech_libirspeech_lm_text
   dest_dir=$dir/wav2vec-u2/hyps_debug_for_apply_withoutvad
   mkdir -p $dest_dir
   #testsets="dev-clean dev-other test-clean test-other"
   testsets="dev-clean"
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   for name in $testsets;do
       cp -r $tsv_dir/${name}.wrd $feat_dir
       cp -r $tsv_dir/${name}.phn $feat_dir
       python $fairseq_dir/examples/wav2vec/unsupervised/w2vu_generate.py\
               --config-dir $config_dir/unsupervised/config/generate \
               --config-name viterbi_md \
               fairseq.common.user_dir=${fairseq_dir}/examples/wav2vec/unsupervised \
               fairseq.task.data=$feat_dir \
               fairseq.common_eval.path=$wav2vec_u2_dir/checkpoint_24_41000.pt \
               fairseq.common_eval.quiet=true \
               fairseq.dataset.gen_subset=$name\
               results_path=$dest_dir\
               decode_stride=2\
               beam_threshold=100.0\
               beam=1500\
               word_score=1.0\
               sil_weight=0.0

   done
   ## dev-clean
   ##  7.33886
fi


## 2023.5.17
## Geting phone sequence of the above way via set decode_stride=1 and config-name=viterbi_frame(decode librispeech dataset from pretrain wav2vec-u2 model) is wrong.
## why, because its decoder phnoneme sequence doesn't four specify symbols(dictionary bos index: 0, dictionary pad index: 1, dictionary eos index: 2, dictionary unk index: 3)
## the four symbols are auto adding the dictionary on the pretraining via fairseq framework. so, the trainset don't the above four symbols. it should be wrong at prepare trainset stage.
## in other words. At decoding librispeech dataset stage, it should directly output phoneme id sequence (at this stage, its contain four specify symbols),not covert phoneme id into phoneme(at this stage, it will remove four specify symbols)
## the below we will directly output phoneme id sequence from pretrain wav2vec-u2 model.
## it is correct way.
if [ ${stage} -le 20 ]&& [ ${stop_stage} -ge 20 ];then
   echo "decode dev-clean dev-other train-960 using wav2vec-u2.0 model to get frame level phonemem id sequence "
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   feat_dir=$tsv_dir/wav2vec_large_feat_dir
   ## dict.phn.txt contain two columns ,first column is monophone(total is 41 mono phones, last monophone is <SIL>), second column is index(0-base),
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

### copy the above speech unsupphoncode into dest directory
if [ ${stage} -le 21 ]&& [ ${stop_stage} -ge 21 ];then
   echo "copy speech unsupphoncode into dest directory"
   dest_dir=dataset/format/librispeech/librispeech_frame_monophncode_using_wav2vec-u2_model_newer
   mkdir -p $dest_dir
   dir=/workspace2/maduo/exp
   input_dir=$dir/wav2vec-u2/frame_phonecode_hyps_newer
   cp -r $input_dir/*.unsupphncode $dest_dir/
   ## code dictionary it total is 41 
   cp -r   dataset/format/librispeech/librispeech_lm_norm_phn_seq/phoness/dict.phncode.txt  $dest_dir/
   mv $dest_dir/dict.phncode.txt $dest_dir/dict.unsupphncode.txt
   ls $dest_dir/
fi



