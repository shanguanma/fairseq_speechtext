#!/usr/bin/env bash


stage=0
stop_stage=1000
. utils/parse_options.sh
#. path_for_fairseq_speechtext.sh
#. path_for_fsq_speechtext.sh
. path_for_fsq_sptt.sh 
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1


## this script is display how to train GAN for SPL paper(https://arxiv.org/abs/2402.15725)
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
  fairseq_dir=/home/maduo/codebase/fairseq_speechtext
  des_dir=/home/maduo/dataset/format/librispeech
  TASK_DATA=$des_dir/wav2vec_large_feat_dir_no_silence/ ## it stores wav2vec2 large model 15 layer representation of raw librispeech speech,
                                                        ## it  removes  silence. it offers feature of speech.
  #cp -r dataset/format/librispeech/librispeech_no_silence/mfcc_no_silence/mfcc_lab/* $TASK_DATA  ## it offers hubert mfcc pesudo label

  # Unpaired text input
  TEXT_DATA=$des_dir/librispeech_lm_norm_phn_seq/unpair_text_0.3M ## it  offers unpair trainset(train.bin, train.idx), devset (dev-clean.bin.dev-clean.idx, dev-other.bin, dev-other.idx)
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
  model_name=w2v_unsup_gan_xp_4gpu_8update_with_unpair_text_0.3M
  exp_dir=$dir/wav2vec-u2_gan_from_scratch/${model_name}
  mkdir -p $exp_dir
  export PYTHONPATH=$fairseq_dir:$PYTHONPATH
  world_size=1
  update_freq=1
  #CUDA_VISIBLE_DEVICES=7  python $fairseq_dir/fairseq_cli/hydra_train.py  --multirun \
  python $fairseq_dir/fairseq_cli/hydra_train.py\
       --config-dir $config_dir/config/gan \
     --config-name w2vu2_local3_md \
     task.data=${TASK_DATA} \
     task.text_data=${TEXT_DATA} \
     task.kenlm_path=${KENLM_PATH} \
     dataset.train_subset=train\
     dataset.valid_subset=\'dev-other,dev-clean\'\
     dataset.batch_size=160\
     dataset.num_workers=6\
     common.user_dir=${fairseq_dir}/examples/wav2vec/unsupervised \
     distributed_training.distributed_world_size=${world_size}\
     distributed_training.distributed_port=-1\
     distributed_training.ddp_backend=legacy_ddp\
     optimization.update_freq=[${update_freq}]\
     common.tensorboard_logdir=$exp_dir\
     hydra.run.dir=$exp_dir\
     hydra.job.name=$exp_dir/w2v_unsup_gan_xp

  # grep -rn '"dev-other_uer":' logs/w2vu2_with_unpair_text_0.3M_train.log | awk -F ":" '{print $12 $13}' | awk -F "," '{print $2}' | sort
  ## grep -rn '"dev-other_uer": "23.0154"' logs/w2vu2_with_unpair_text_0.3M_train.log
#688:[2024-01-07 05:20:00,840][dev-other][INFO] - {"epoch": 18, "dev-other_loss": "0.23", "dev-other_ntokens": "9989.44", "dev-other_nsentences": "159.111", "dev-other_sample_size": "9989.44", "dev-other_lm_score_sum": "-279054", "dev-other_num_pred_chars": "180370", "dev-other_vocab_seen_pct": "0.878049", "dev-other_uer": "23.0154", "dev-other_weighted_lm_ppl": "43.2413", "dev-other_lm_ppl": "33.3377", "dev-other_wps": "4823.4", "dev-other_wpb": "9989.4", "dev-other_bsz": "159.1", "dev-other_num_updates": "30000", "dev-other_best_weighted_lm_ppl": "43.2413"}

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
   #model_name=w2v_unsup_gan_xp_4gpu_8update
   #exp_dir=$dir/wav2vec-u2_gan_from_scratch/${model_name}
   wav2vec_u2_dir=$dir/wav2vec-u2_gan_from_scratch/w2v_unsup_gan_xp_4gpu_8update_with_unpair_text_0.3M/
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
               fairseq.common_eval.path=$wav2vec_u2_dir/checkpoint_18_30000.pt \
               fairseq.common_eval.quiet=false \
               fairseq.dataset.gen_subset=$name\
               results_path=$dest_dir\
               decode_stride=2\
               beam_threshold=100.0\
               beam=1500\
               word_score=1.0\
               sil_weight=0.0

   done
   #  grep -rn 'WER:' logs/w2vu2_with_unpair_text_0.3M_infer_again.log
#10848:[2024-01-22 17:12:07,077][__main__][INFO] - WER: 17.611222034505708
#10850:[2024-01-22 17:12:07,078][__main__][INFO] - | Generate dev-clean with beam=1500, lm_weight=2.0, word_score=1.0, sil_weight=0.0, blank_weight=0.0, WER: 17.611222034505708, LM_PPL: inf, num feats: 825119, length: 207584, UER to viterbi: 0, score: 17.611222034505708
fi

### this script don't eval, it should be work.
if [ ${stage} -le 3 ]&& [ ${stop_stage} -ge 3 ];then
   echo "decode  dev-clean dev-other test-clean test-other using wav2vec-u2.0 model and compute WER with hlg graph(phn2word)"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   root_dir=/workspace2/maduo/dataset/format/librispeech
   tsv_dir=$root_dir/librispeech_no_silence ### it contains *.wrd and *.phn, because it doesn't matter whether the wav contains silence or not
   feat_dir=$root_dir/wav2vec_large_feat_dir_no_silence
   ## dict.phn.txt contain two columns ,first column is monophone(total is 41 mono phones, last monophone is <SIL>), second column is index(0-base),
   cp -r $root_dir/librispeech_lm_norm_phn_seq/phoness/dict.phn.txt $feat_dir
   config_dir=$fairseq_dir/examples/wav2vec/
   dir=/workspace2/maduo/exp/

   ##kenlm_path=$root_dir/librispeech_lm_norm_phn_seq/kenlm.wrd.o40003.bin  ## must be absolute path
   ##lexicon_path=$root_dir/librispeech_lm_norm_phn_seq/lexicon_filtered.lst ## word2phn dictionary
   hlg_graph_path=$root_dir/librispeech_lm_norm_phn_seq/fst/phn_to_words_sil/HLG.phn.kenlm.wrd.o40003.fst

   wav2vec_u2_dir=$dir/wav2vec-u2_gan_from_scratch/w2v_unsup_gan_xp_4gpu_8update_with_unpair_text_half/
   dest_dir=$wav2vec_u2_dir/hyps_debug_for_apply_vad_and_kaldi_decoder
   mkdir -p $dest_dir
   testsets="dev-clean dev-other test-clean test-other"
   #testsets="dev-clean" ## for debug
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   for name in $testsets;do
       cp -r $tsv_dir/${name}.wrd $feat_dir
       cp -r $tsv_dir/${name}.phn $feat_dir
       python $fairseq_dir/examples/wav2vec/unsupervised/w2vu_generate.py\
               --config-dir $config_dir/unsupervised/config/generate \
               --config-name kaldi_decode \
               fairseq.common.user_dir=${fairseq_dir}/examples/wav2vec/unsupervised \
               fairseq.task.data=$feat_dir \
               fairseq.common_eval.path=$wav2vec_u2_dir/checkpoint_18_30000.pt \
               fairseq.common_eval.quiet=false \
               fairseq.dataset.gen_subset=$name\
               w2l_decoder=KALDI\
               results_path=$dest_dir\
               decode_stride=2\
               targets=wrd\
               beam_threshold=100.0\
               beam=1500\
               word_score=1.0\
               sil_weight=0.0

   done
   

fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
  fairseq_dir=/home/maduo/codebase/fairseq_speechtext
  des_dir=/home/maduo/dataset/format/librispeech
  TASK_DATA=$des_dir/wav2vec_large_feat_dir_no_silence/ ## it stores wav2vec2 large model 15 layer representation of raw librispeech speech,
                                                        ## it  removes  silence. it offers feature of speech.
  #cp -r dataset/format/librispeech/librispeech_no_silence/mfcc_no_silence/mfcc_lab/* $TASK_DATA  ## it offers hubert mfcc pesudo label

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
  model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_all
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
     model.code_penalty=0.0\
     model.gradient_penalty=1.5\
     model.smoothness_weight=1.5\
     model.mmi_weight=0.5\
     common.seed=1\
     common.user_dir=${fairseq_dir}/examples/wav2vec/unsupervised \
     distributed_training.distributed_world_size=${world_size}\
     distributed_training.distributed_port=-1\
     distributed_training.ddp_backend=legacy_ddp\
     optimization.update_freq=[${update_freq}]\
     common.tensorboard_logdir=$exp_dir\
     hydra.run.dir=$exp_dir\
     hydra.job.name=$exp_dir/w2v_unsup_gan_xp

fi

## 2023.8.10.update, compared to stage20, I still keep reduce four specify symbols, because it will add four sysmbol into dictionary (e.g. dict.*.txt) and rewrite every utterance code id( it adds 4) in fairseq prepared dataset defult.
## No post-processing(rewrite every utterance code id( it adds 4))
## of sentence codes unless you manually intervene.
if [ ${stage} -le 21 ]&& [ ${stop_stage} -ge 21 ];then
   echo "decode dev-clean dev-other train-960 using wav2vec-u2.0 model to get frame level phonemem id sequence "
   fairseq_dir=/home/maduo/codebase/fairseq_speechtext
   tsv_dir=/home/maduo/dataset/format/librispeech/
   #tsv_dir=/workspace2/maduo/tests
   feat_dir=$tsv_dir/wav2vec_large_feat_dir
   ## dict.phn.txt contain two columns ,first column is monophone(total is 41 mono phones, last monophone is <SIL>), second column is index(0-base),
   #cp -r $tsv_dir/librispeech_lm_norm_phn_seq/phoness/dict.phn.txt $feat_dir
   config_dir=$fairseq_dir/examples/wav2vec/
   dir=/home/maduo/exp
   wav2vec_u2_dir=$dir/wav2vec-u2_gan_from_scratch/w2v_unsup_gan_xp_4gpu_8update_with_unpair_text_0.3M
   model_name=w2v_unsup_gan_xp_4gpu_8update_with_unpair_text_0.3M
   wav2vec_u2_dir=$dir/wav2vec-u2_gan_from_scratch/$model_name

   #dest_dir=$dir/wav2vec-u2/frame_phonecode_hyps_newer_final
   dest_dir=$wav2vec_u2_dir/frame_phonecode_hyps_newer_final
   mkdir -p $dest_dir
   testsets="dev-clean dev-other train-960"
   #testsets="dev-clean "
   #testsets="test-other"
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   for name in $testsets;do
       cp -r $tsv_dir/tsv_dir/${name}.wrd $feat_dir
       cp -r $tsv_dir/tsv_dir/${name}.phn $feat_dir
       python $fairseq_dir/examples/wav2vec/unsupervised/w2vu_generate_frame_phncode_final.py\
               --config-dir $config_dir/unsupervised/config/generate \
               --config-name viterbi_frame \
               fairseq.common.user_dir=${fairseq_dir}/examples/wav2vec/unsupervised \
               fairseq.task.data=$feat_dir \
               fairseq.common_eval.path=$wav2vec_u2_dir/checkpoint_18_30000.pt \
               fairseq.common_eval.quiet=false \
               fairseq.dataset.gen_subset=$name\
               results_path=$dest_dir\
               decode_stride=1
   done
fi


if [ ${stage} -le 22 ]&& [ ${stop_stage} -ge 22 ];then
   echo "generate phoneme code"
   # generated phoneme code base on dataset/format/librispeech/librispeech_lm_norm_phn_seq/dict.phn_for_wav2vec-u2.txt
   #  wc -l dataset/format/librispeech/librispeech_lm_norm_phn_seq/dict.phn_for_wav2vec-u2.txt
   #  41 dataset/format/librispeech/librispeech_lm_norm_phn_seq/dict.phn_for_wav2vec-u2.txt  
   lab_dir=dataset/format/librispeech/w2v_unsup_gan_xp_4gpu_8update_with_unpair_text_0.3M
   n_cluster=41
   for x in $(seq 0 $((n_cluster - 1 )));do
   echo "$x 1"
   done>>$lab_dir/dict.unsupphncode1.txt
   

fi
