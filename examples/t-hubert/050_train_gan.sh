#!/usr/bin/env bash


stage=0
stop_stage=1000
. utils/parse_options.sh
#. path_for_fairseq_speechtext.sh
#. path_for_fsq_speechtext.sh
. path_for_fsq_sptt.sh
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

#Note: This gan model training requires a lot of parameter adjustment to converge to a good result.
## the below config is better parameter on one RTX3090.
## The following script is just to show how to train such a model.
## You can change these parameter model.code_penalty=2.0\
#     model.gradient_penalty=1.5\
#     model.smoothness_weight=0.5\
#     model.mmi_weight=0.5\
#     common.seed=0\
# dependent your server and data.
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
  fairseq_dir=/home/maduo/codebase/fairseq_speechtext
  des_dir=/home/maduo/dataset/format/librispeech
  TASK_DATA=$des_dir/wav2vec_large_feat_dir_no_silence/ ## it stores wav2vec2 large model 15 layer representation of raw librispeech speech,
                                                        ## it  removes  silence. it offers feature of speech.
  cp -r dataset/format/librispeech/librispeech_no_silence/mfcc_no_silence/mfcc_lab/* $TASK_DATA  ## it offers hubert mfcc pesudo label

  # Unpaired text input
  TEXT_DATA=$des_dir/librispeech_lm_norm_phn_seq/unpair_text_all ## it  offers unpair trainset(train.bin, train.idx), devset (dev-clean.bin.dev-clean.idx, dev-other.bin, dev-other.idx)
  # it also offers phone dictionary, its type is as follow:
  ##head  dataset/format/librispeech/librispeech_lm_norm_phn_seq/phonesss/dict.phn.txt
  ## AH 336212
  ## N 238016
  ## S 209100
  ## T 194878
  ## L 188633
  ## IH 182116
  ## R 172703
  ## K 154411
  ## IY 138376
  ## Z 128619
  KENLM_PATH=$des_dir/librispeech_lm_norm_phn_seq/lm.phones.filtered.04.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)
  config_dir=$fairseq_dir/examples/wav2vec/unsupervised

  dir=/home/maduo/exp/
  model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_all_md
  exp_dir=$dir/wav2vec-u2_gan_from_scratch/${model_name}
  mkdir -p $exp_dir
  export PYTHONPATH=$fairseq_dir:$PYTHONPATH
  world_size=1
  update_freq=1
  #CUDA_VISIBLE_DEVICES=7  python $fairseq_dir/fairseq_cli/hydra_train.py  --multirun \
  python $fairseq_dir/fairseq_cli/hydra_train.py\
       --config-dir $config_dir/config/gan \
     --config-name w2vu2_local3_md_1 \
     task.data=${TASK_DATA} \
     task.text_data=${TEXT_DATA} \
     task.kenlm_path=${KENLM_PATH} \
     dataset.train_subset=train\
     dataset.valid_subset=\'dev-other,dev-clean\'\
     dataset.batch_size=160\
     dataset.num_workers=6\
     model.code_penalty=2.0\
     model.gradient_penalty=1.5\
     model.smoothness_weight=0.5\
     model.mmi_weight=0.5\
     common.seed=0\
     common.user_dir=${fairseq_dir}/examples/wav2vec/unsupervised \
     distributed_training.distributed_world_size=${world_size}\
     distributed_training.distributed_port=-1\
     distributed_training.ddp_backend=legacy_ddp\
     optimization.update_freq=[${update_freq}]\
     common.tensorboard_logdir=$exp_dir\
     hydra.run.dir=$exp_dir\
     hydra.job.name=$exp_dir/w2v_unsup_gan_xp
fi

if [ ${stage} -le 2 ]&& [ ${stop_stage} -ge 2 ];then
   echo "decode  dev-clean dev-other test-clean test-other using wav2vec-u2.0 model and compute PER "
   fairseq_dir=/home/maduo/codebase/fairseq_speechtext
   root_dir=/home/maduo/dataset/format/librispeech
   tsv_dir=$root_dir/librispeech_no_silence ### it contains *.wrd and *.phn, because it doesn't matter whether the wav contains silence or not
   feat_dir=$root_dir/wav2vec_large_feat_dir_no_silence
   ## dict.phn.txt contain two columns ,first column is monophone(total is 41 mono phones, last monophone is <SIL>), second column is index(0-base),
   #cp -r $root_dir/librispeech_lm_norm_phn_seq/phoness/dict.phn.txt $feat_dir
   config_dir=$fairseq_dir/examples/wav2vec/
   dir=/home/maduo/exp/
   model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_all_md
   wav2vec_u2_dir=$dir/wav2vec-u2_gan_from_scratch/${model_name}
   model_path=$wav2vec_u2_dir/checkpoint_18_30000.pt ## It is assumed here that the best checkpoint name is checkpoint_18_30000.pt , I select model via best dev-other_uer.
   #wav2vec_u2_dir=$dir/wav2vec-u2_gan_from_scratch/w2v_unsup_gan_xp_4gpu_8update_with_unpair_text_0.3M/
   dest_dir=$wav2vec_u2_dir/hyps_debug_for_apply_vad
   mkdir -p $dest_dir
   #testsets="dev-clean dev-other test-clean test-other train"
   testsets="dev-clean"
   #testsets="dev-clean dev-other test-clean test-other"
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   for name in $testsets;do
       cp -r $tsv_dir/${name}.wrd $feat_dir
       cp -r $tsv_dir/${name}.phn $feat_dir
       python $fairseq_dir/examples/wav2vec/unsupervised/w2vu_generate.py\
               --config-dir $config_dir/unsupervised/config/generate \
               --config-name viterbi_md \
               fairseq.common.user_dir=${fairseq_dir}/examples/wav2vec/unsupervised \
               fairseq.task.data=$feat_dir \
               fairseq.common_eval.path=$model_path \
               fairseq.common_eval.quiet=false \
               fairseq.dataset.gen_subset=$name\
               results_path=$dest_dir\
               decode_stride=2\
               beam_threshold=100.0\
               beam=1500\
               word_score=1.0\
               sil_weight=0.0

   done
fi



