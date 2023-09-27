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
    ## PER%
    #  dev-clean  dev-other  test-clean test-other
    #   7.3       10.28        7.45      10.22

fi
#### phoneme-unit tokenizer for unpaired speech part
#### in order to get speech frame level tokens using wav2vec-u2.0 
if [ ${stage} -le 1 ]&& [ ${stop_stage} -ge 1 ];then
   echo "decode dev-clean dev-other train-960 using wav2vec-u2.0 model "
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   feat_dir=$tsv_dir/wav2vec_large_feat_dir
   cp -r $tsv_dir/librispeech_lm_norm_phn_seq/phoness/dict.phn.txt $feat_dir
   config_dir=$fairseq_dir/examples/wav2vec/
   dir=/workspace2/maduo/exp
   wav2vec_u2_dir=/workspace2/maduo/model_hub/librispeech/wav2vec-u2.0-trained_model_using_librispeech_libirspeech_lm_text
   dest_dir=$dir/wav2vec-u2/frame_hyps
   mkdir -p $dest_dir
   #testsets="dev-clean dev-other train-960"
   #testsets="train-clean-100"
   #testsets="dev-clean"
   #testsets="test-other"
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   for name in $testsets;do
       cp -r $tsv_dir/${name}.wrd $feat_dir
       cp -r $tsv_dir/${name}.phn $feat_dir
       python $fairseq_dir/examples/wav2vec/unsupervised/w2vu_generate.py\
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
if [ ${stage} -le 2 ]&& [ ${stop_stage} -ge 2 ];then
   echo "covert librispeech speech  phoneme sequence to code sequence using phn2id dictionary"
   ### this phn2id dictionary is from librilm via g2p. this dictionary contains 41 mono phones including silence phones.
   #### more detail , you can reference : source_md/wav2vec-u2/prepared_text_for_sthubert.sh --stage 1 --stop-stage 6
   root_dir=dataset/format/librispeech/
   dict=$root_dir/librispeech_lm_norm_phn_seq/phoness/dict.phn.txt
   data_dir=exp/wav2vec-u2/frame_hyps
   output_dir=$root_dir/librispeech_frame_monophncode_using_wav2vec-u2_model
   mkdir -p $output_dir
   for name in dev-clean dev-other train-960;do
     python source_md/wav2vec-u2/text/phn_to_code.py\
            $data_dir/${name}_units.txt\
            $dict\
            $output_dir/${name}.phncode
   done
   echo "finish !!!!!!!!!!"
fi

# phoneme-unit tokenizer for unpaired text part
if [ ${stage} -le 3 ]&& [ ${stop_stage} -ge 3 ];then
   echo "covert librilm text  phoneme sequence to code sequence using phn2id dictionary"
   ### this phn2id dictionary is from librilm via g2p. this dictionary contains 41 mono phones including silence phones.
   #### more detail , you can reference : source_md/wav2vec-u2/prepared_text_for_sthubert.sh --stage 1 --stop-stage 6
   ### librilm phoneme sequenece: you can reference: source_md/wav2vec-u2/prepared_text_for_sthubert.sh --stage 1 --stop-stage 5 
   root_dir=dataset/format/librispeech/
   dict=$root_dir/librispeech_lm_norm_phn_seq/phoness/dict.phn.txt
   data_dir=$root_dir/librispeech_lm_norm_phn_seq/phones
   output_dir=$root_dir/librispeech_lm_monophncode_using_monophn_dict
   mkdir -p $output_dir
   for name in lm.phones.filtered;do
     python source_md/wav2vec-u2/text/phn_to_code.py\
            $data_dir/$name.txt\
            $dict\
            $output_dir/librilm.phncode
   done
   echo "finish !!!!!!!!!!"
fi

# phoneme-unit tokenizer for unpaired text part
if [ ${stage} -le 4 ]&& [ ${stop_stage} -ge 4 ];then
   echo "covert librilm text  phoneme sequence to code sequence using phn2id dictionary including four specfiy symbols"
   ### this phn2id dictionary is from librilm via g2p. this dictionary contains 41 mono phones including silence phones and '<s>','<pad>','</s>','<unk>'
   #### more detail , you can reference : source_md/wav2vec-u2/prepared_text_for_sthubert.sh --stage 1 --stop-stage 8
   ### librilm phoneme sequenece: you can reference: source_md/wav2vec-u2/prepared_text_for_sthubert.sh --stage 1 --stop-stage 5
   root_dir=dataset/format/librispeech/
   dict=$root_dir/librispeech_lm_norm_phn_seq/phoness/dict.phn_for_text.txt
   data_dir=$root_dir/librispeech_lm_norm_phn_seq/phones
   output_dir=$root_dir/librispeech_lm_monophncode_using_monophn_dict_newer
   mkdir -p $output_dir
   for name in lm.phones.filtered;do
     python source_md/wav2vec-u2/text/phn_to_code.py\
            $data_dir/$name.txt\
            $dict\
            $output_dir/librilm.phncode
   done
   echo "finish !!!!!!!!!!"
fi



#### phoneme-unit tokenizer for unpaired speech part
#### debug, Why is the last sample of each data set lost and not decoded
if [ ${stage} -le 10 ]&& [ ${stop_stage} -ge 10 ];then
   echo "decode dev-clean dev-other train-960 using wav2vec-u2.0 model "
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   feat_dir=$tsv_dir/wav2vec_large_feat_dir
   cp -r $tsv_dir/librispeech_lm_norm_phn_seq/phoness/dict.phn.txt $feat_dir
   config_dir=$fairseq_dir/examples/wav2vec/
   dir=/workspace2/maduo/exp
   wav2vec_u2_dir=/workspace2/maduo/model_hub/librispeech/wav2vec-u2.0-trained_model_using_librispeech_libirspeech_lm_text
   dest_dir=$dir/wav2vec-u2/frame_hyps_debug
   mkdir -p $dest_dir
   #testsets="dev-clean dev-other train-960"
   testsets="dev-clean"
   #testsets="test-other"
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   for name in $testsets;do
       cp -r $tsv_dir/${name}.wrd $feat_dir
       cp -r $tsv_dir/${name}.phn $feat_dir
       python $fairseq_dir/examples/wav2vec/unsupervised/w2vu_generate_debug.py\
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

## 2023.8.10.update, compared to stage20, I still keep reduce four specify symbols, because it will add four sysmbol into dictionary (e.g. dict.*.txt) and rewrite every utterance code id( it adds 4) in fairseq prepared dataset defult.
## No post-processing(rewrite every utterance code id( it adds 4)) 
## of sentence codes unless you manually intervene. 
if [ ${stage} -le 21 ]&& [ ${stop_stage} -ge 21 ];then
   echo "decode dev-clean dev-other train-960 using wav2vec-u2.0 model to get frame level phonemem id sequence "
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   feat_dir=$tsv_dir/wav2vec_large_feat_dir
   ## dict.phn.txt contain two columns ,first column is monophone(total is 41 mono phones, last monophone is <SIL>), second column is index(0-base),
   cp -r $tsv_dir/librispeech_lm_norm_phn_seq/phoness/dict.phn.txt $feat_dir
   config_dir=$fairseq_dir/examples/wav2vec/
   dir=/workspace2/maduo/exp
   wav2vec_u2_dir=/workspace2/maduo/model_hub/librispeech/wav2vec-u2.0-trained_model_using_librispeech_libirspeech_lm_text
   dest_dir=$dir/wav2vec-u2/frame_phonecode_hyps_newer_final
   mkdir -p $dest_dir
   testsets="dev-clean dev-other train-960"
   #testsets="dev-clean"
   #testsets="test-other"
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   for name in $testsets;do
       cp -r $tsv_dir/${name}.wrd $feat_dir
       cp -r $tsv_dir/${name}.phn $feat_dir
       python $fairseq_dir/examples/wav2vec/unsupervised/w2vu_generate_frame_phncode_final.py\
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


