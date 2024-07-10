#!/usr/bin/env bash

stage=0
stop_stage=1000
. utils/parse_options.sh
. path_for_fairseq_speechtext.sh

export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

## I still keep reduce four specify symbols, because it will add four sysmbol into dictionary (e.g. dict.*.txt) and rewrite every utterance code id( it adds 4) in fairseq prepared dataset defult.
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
   #wav2vec_u2_dir=/workspace2/maduo/model_hub/librispeech/wav2vec-u2.0-trained_model_using_librispeech_libirspeech_lm_text
   #model_path=$wav2vec_u2_dir/checkpoint_24_41000.pt # this checkpoint is reported at submit paper.
   model_path=gan_english_checkpoint/checkpoint_24_41000.pt
   # main config of the above gan model : 'smoothness_weight': 1.5, 'smoothing': 0.0, 'smoothing_one_sided': False, 'gradient_penalty': 1.5, 'probabilistic_grad_penalty_slicing': False, 'code_penalty': 0.0, 'mmi_weight': 0.5, 'target_dim': 64, 'target_downsample_rate': 2, 'gumbel': False, 'hard_gumbel': False, 'seed': 1,
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
               fairseq.common_eval.path=$model_path\
               fairseq.common_eval.quiet=false \
               fairseq.dataset.gen_subset=$name\
               results_path=$dest_dir\
               decode_stride=1
   done
fi



