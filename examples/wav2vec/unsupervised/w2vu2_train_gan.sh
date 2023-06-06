#!/usr/bin/env bash


stage=1

stop_stage=1000
.  utils/parse_options.sh
. path_for_fairseq_speechtext.sh

export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
export OC_CAUSE=1
RVAD_ROOT=/workspace2/maduo/rVADfast
fairseq_dir=/workspace2/maduo/fairseq_speechtext
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
   echo "prepare tsv file for librispeech"
   raw_wav_dir=/workspace/xianghu/datasets/LibriSpeech
   des_dir=/workspace2/maduo/dataset/format/librispeech
   for name in dev-clean dev-other test-clean test-other train-clean-100  train-clean-360  train-other-500;do
     python3  source_md/wav2vec-u2/wav2vec_manifest_md.py\
           $raw_wav_dir\
           --dest_file $des_dir/$name.tsv\
           --ext flac
   done

fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
   echo "get vad file"
   input_dir=/workspace2/maduo/dataset/format/librispeech
   des_dir=/workspace2/maduo/dataset/format/librispeech/librispeech_no_silence/
   #datasets="dev-clean dev-other test-clean test-other train-clean-100  train-clean-360  train-other-500"
   #datasets="dev-clean"
   datasets="dev-other test-clean test-other train-clean-100  train-clean-360  train-other-500"
   for name in $datasets;do
     python $fairseq_dir/examples/wav2vec/unsupervised/scripts/vads.py\
             -r $RVAD_ROOT < $input_dir/$name.tsv > $des_dir/${name}.vads   
   done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
   echo "apply vad and remove silence"
   input_dir=/workspace2/maduo/dataset/format/librispeech
   des_dir=dataset/format/librispeech/librispeech_no_silence
   #datasets="dev-clean dev-other test-clean test-other train-clean-100  train-clean-360  train-other-500"
   #datasets="dev-clean"
   datasets="dev-other test-clean test-other train-clean-100  train-clean-360  train-other-500"
   for name in $datasets;do
     python $fairseq_dir/examples/wav2vec/unsupervised/scripts/remove_silence.py \
          --tsv $input_dir/${name}.tsv --vads $des_dir/${name}.vads --out $des_dir/$name
     echo "finish $name !!!!!!!!!!!!!!!!!!!!"
   done
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
   echo "prepare tsv file for no silence librispeech"
   raw_wav_dir=/workspace2/maduo/dataset/format/librispeech/librispeech_no_silence
   des_dir=/workspace2/maduo/dataset/format/librispeech/librispeech_no_silence
   #datasets="dev-clean"
   datasets="dev-other test-clean test-other train-clean-100  train-clean-360  train-other-500"
   for name in $datasets;do
     python3  source_md/wav2vec-u2/wav2vec_manifest_md.py\
           $raw_wav_dir/$name\
           --dest_file $des_dir/$name.tsv\
           --ext flac
     echo "finish $name !!!!!!!!!!!!!!!!!!!!"
   done

fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
   echo "merger train part into one"
   des_dir=/workspace2/maduo/dataset/format/librispeech/librispeech_no_silence
   python source_md/wav2vec-u2/three_tsv2one.py \
          $des_dir/train-clean-100.tsv\
          $des_dir/train-clean-360.tsv\
          $des_dir/train-other-500.tsv\
          $des_dir/train.tsv

   des_raw_dir=/workspace2/maduo/dataset/format/librispeech/
   python source_md/wav2vec-u2/three_tsv2one.py \
          $des_raw_dir/train-clean-100.tsv\
          $des_raw_dir/train-clean-360.tsv\
          $des_raw_dir/train-other-500.tsv\
          $des_raw_dir/train.tsv   
   ## in order to same as wav2vec-u2 offer script, eval the hold pipeline is correct. 
   #des_dir=/workspace2/maduo/dataset/format/librispeech/librispeech_no_silence
   #python source_md/wav2vec-u2/two_tsv2one.py\
   #       $des_dir/dev-clean.tsv\
   #       $des_dir/dev-other.tsv\
   #       $des_dir/valid.tsv
    
   #des_raw_dir=/workspace2/maduo/dataset/format/librispeech
   #python source_md/wav2vec-u2/two_tsv2one.py\
   #       $des_raw_dir/dev-clean.tsv\
   #       $des_raw_dir/dev-other.tsv\
   #       $des_raw_dir/valid.tsv
   echo "finish !!!!!!!!!!!!!!!!!!!!"
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then
   echo "using remove silence audio pass into wav2vec_larger model and output specify layer represent as audio feature of wav2vec-u2"
   export PYTHONPATH=/workspace2/maduo/fairseq_speechtext:$PYTHONPATH

   des_dir=/workspace2/maduo/dataset/format/librispeech/librispeech_no_silence
   feat_dir=/workspace2/maduo/dataset/format/librispeech/wav2vec_large_feat_dir_no_silence/ ## it stores wav2vec2 large model 15 layer representation of raw librispeech speech,
                                                        ## it  removes  silence.
   model=/workspace2/maduo/model_hub/librispeech/wav2vec2_Large_LV-60_no_finetune_offical_offer/wav2vec_vox_new.pt
   mkdir -p $feat_dir
   layer=14 #0-based index
   datasets="dev-other test-clean test-other train"
   #datasets="dev-clean"
   for name in  $datasets;do
     python  source_md/wav2vec-u2/wav2vec_extract_features.py\
           $des_dir\
           --split $name\
           --save-dir $feat_dir\
           --checkpoint $model\
           --layer $layer
     echo "finish $name !!!!!!!!!!!!!!!!!!!!"
   done   
fi
### prepared no silence data .wrd .ltr files
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ];then
   tsv_dir=/workspace2/maduo/dataset/format/librispeech/librispeech_no_silence
   withsiltsv_dir=/workspace2/maduo/dataset/format/librispeech/
   datasets="dev-other test-clean test-other train"
   #datasets="dev-other test-clean test-other"
   #datasets="dev-clean"
   #datasets="train"  ## for self-train using kaldi
   for name in $datasets;do
     python3 source_md/wav2vec-u2/text/libri_labels_for_no_silence.py\
         $tsv_dir/${name}.tsv\
         $withsiltsv_dir/${name}.tsv\
         --output-dir $tsv_dir\
         --output-name $name
    echo "finish $name !!!!"
   done
fi


### prepared no silence data .phn files
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ];then
   datasets="dev-other test-clean test-other train"
   #datasets="dev-other test-clean test-other"
   #datasets="dev-clean"
   #datasets="train"
   tsv_dir=dataset/format/librispeech/librispeech_no_silence
   for name in $datasets;do
     ## it outputs *.phn file
    cat $tsv_dir/${name}.wrd |  python source_md/wav2vec-u2/text/g2p_wrd_to_phn.py >$tsv_dir/${name}.phn
   done
fi

### in order to eval whether the vad is correct or not.
if [ ${stage} -le 8 ]&& [ ${stop_stage} -ge 8 ];then
   echo "decode  dev-clean dev-other test-clean test-other using wav2vec-u2.0 model and compute PER "
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   root_dir=/workspace2/maduo/dataset/format/librispeech
   tsv_dir=$root_dir/librispeech_no_silence ### it contains *.wrd and *.phn, because it doesn't matter whether the wav contains silence or not
   feat_dir=$root_dir/wav2vec_large_feat_dir_no_silence
   ## dict.phn.txt contain two columns ,first column is monophone(total is 41 mono phones, last monophone is <SIL>), second column is index(0-base),
   cp -r $root_dir/librispeech_lm_norm_phn_seq/phoness/dict.phn.txt $feat_dir
   config_dir=$fairseq_dir/examples/wav2vec/
   dir=/workspace2/maduo/exp/
   #model_name=w2v_unsup_gan_xp_4gpu_8update
   #exp_dir=$dir/wav2vec-u2_gan_from_scratch/${model_name}
   wav2vec_u2_dir=/workspace2/maduo/model_hub/librispeech/wav2vec-u2.0-trained_model_using_librispeech_libirspeech_lm_text
   dest_dir=$dir/wav2vec-u2/hyps_debug_for_apply_vad_final_phoneme
   mkdir -p $dest_dir
   #testsets="dev-clean dev-other test-clean test-other train"
   #testsets="dev-clean"
   #testsets="train"
   testsets="dev-clean dev-other test-clean test-other"
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
               fairseq.common_eval.quiet=false \
               fairseq.dataset.gen_subset=$name\
               results_path=$dest_dir\
               decode_stride=2\
               beam_threshold=100.0\
               beam=1500\
               word_score=1.0\
               sil_weight=0.0

   done
   ## PER%, log: grep -rn "WER:" logs/wav2vec-u2_from_scratch_stage8_generate_phoneme_and_PER.log
   ## beam=1500, beam_threshold=100.0
   ## dev-clean dev-other  test-clean test-other
   ## 8.1928      11.01     8.35         11.18

fi
### in order to eval whether the vad is correct or not.
if [ ${stage} -le 9 ]&& [ ${stop_stage} -ge 9 ];then
   echo "decode  dev-clean dev-other test-clean test-other using wav2vec-u2.0 model and compute PER "
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   root_dir=/workspace2/maduo/dataset/format/librispeech
   tsv_dir=$root_dir/librispeech_no_silence ### it contains *.wrd and *.phn, because it doesn't matter whether the wav contains silence or not
   feat_dir=$root_dir/wav2vec_large_feat_dir_no_silence
   ## dict.phn.txt contain two columns ,first column is monophone(total is 41 mono phones, last monophone is <SIL>), second column is index(0-base),
   cp -r $root_dir/librispeech_lm_norm_phn_seq/phoness/dict.phn.txt $feat_dir
   config_dir=$fairseq_dir/examples/wav2vec/
   dir=/workspace2/maduo/exp/
   #model_name=w2v_unsup_gan_xp_4gpu_8update
   #exp_dir=$dir/wav2vec-u2_gan_from_scratch/${model_name}
   wav2vec_u2_dir=/workspace2/maduo/model_hub/librispeech/wav2vec-u2.0-trained_model_using_librispeech_libirspeech_lm_text
   dest_dir=$dir/wav2vec-u2/hyps_debug_for_apply_vad
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
               beam_threshold=50.0\
               beam=5\
               word_score=1.0\
               sil_weight=0.0

   done
   ## beam=5, beam_threshold=50
   ## dev_clean 
   ## 8.1928
   ## with 4-gram kenlm decoding
   ## 

fi

### kenlm rescore is not working, (TODO) md
if [ ${stage} -le 10 ]&& [ ${stop_stage} -ge 10 ];then
   echo "decode  dev-clean dev-other test-clean test-other using wav2vec-u2.0 model and compute WER using kenlm and kaldi_decoder "
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   root_dir=/workspace2/maduo/dataset/format/librispeech
   tsv_dir=$root_dir/librispeech_no_silence ### it contains *.wrd and *.phn, because it doesn't matter whether the wav contains silence or not
   feat_dir=$root_dir/wav2vec_large_feat_dir_no_silence
   ## dict.phn.txt contain two columns ,first column is monophone(total is 41 mono phones, last monophone is <SIL>), second column is index(0-base),
   cp -r $root_dir/librispeech_lm_norm_phn_seq/phoness/dict.phn.txt $feat_dir
   config_dir=$fairseq_dir/examples/wav2vec/
   dir=/workspace2/maduo/exp/
   #model_name=w2v_unsup_gan_xp_4gpu_8update
   #exp_dir=$dir/wav2vec-u2_gan_from_scratch/${model_name}
   kenlm_path=$root_dir/librispeech_lm_norm_phn_seq/kenlm.wrd.o40003.bin  ## must be absolute path
   lexicon_path=$root_dir/librispeech_lm_norm_phn_seq/lexicon_filtered.lst ## word2phn dictionary
   wav2vec_u2_dir=/workspace2/maduo/model_hub/librispeech/wav2vec-u2.0-trained_model_using_librispeech_libirspeech_lm_text
   dest_dir=$dir/wav2vec-u2/hyps_debug_for_apply_vad_with_4gramlm_and_kaldi_decode
   mkdir -p $dest_dir
   #testsets="dev-clean dev-other test-clean test-other"
   testsets="dev-clean"
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   for name in $testsets;do
       cp -r $tsv_dir/${name}.wrd $feat_dir
       cp -r $tsv_dir/${name}.phn $feat_dir
       python $fairseq_dir/examples/wav2vec/unsupervised/w2vu_generate.py\
               --config-dir $config_dir/unsupervised/config/generate \
               --config-name kaldi_decode \
               fairseq.common.user_dir=${fairseq_dir}/examples/wav2vec/unsupervised \
               fairseq.task.data=$feat_dir \
               fairseq.common_eval.path=$wav2vec_u2_dir/checkpoint_24_41000.pt \
               fairseq.common_eval.quiet=false \
               fairseq.dataset.gen_subset=$name\
               results_path=$dest_dir\
               decode_stride=2\
               lm_model=$kenlm_path\
               lexicon=$lexicon_path\
               w2l_decoder=KALDI\
               targets=wrd\
               beam_threshold=100.0\
               beam=1500\
               word_score=1.0\
               sil_weight=0.0

   done
   ## beam=1500, beam_threshold=100.0
   ## dev_clean(PER)
   ## 8.1928
   ## with 4-gram kenlm decoding
   ## 8.1928 (because, it compute still PER ,not WER, so, It will not apply word level kenlm model peformance, so when it compute PER, it is not work using word level kenlm model)

   
fi

### 
if [ ${stage} -le 11 ]&& [ ${stop_stage} -ge 11 ];then
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

   wav2vec_u2_dir=/workspace2/maduo/model_hub/librispeech/wav2vec-u2.0-trained_model_using_librispeech_libirspeech_lm_text
   #dest_dir=$dir/wav2vec-u2/hyps_apply_vad_and_kaldi_decoder
   dest_dir=$dir/wav2vec-u2/hyps_apply_vad_and_kaldi_decoder_debug
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
               fairseq.common_eval.path=$wav2vec_u2_dir/checkpoint_24_41000.pt \
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
               #kaldi_decoder_config=$config_dir/unsupervised/config/generate/kaldi_decoder_conf.yaml

   done
   ## beam=1500, beam_threshold=100.0
   ## dev_clean (PER)
   ## 8.1928
   ## using kaldi hlg decode graph to decoding, and compute WER
   ## 8.1928 (because, it compute still PER ,not WER, so, It will not apply word level kenlm model peformance, so when it compute PER, it is not work using word level kenlm model)
   ## dev-clean(WER), max_mem: 120, its means that in determinize-lattice: size exceeds maximum 120 bytes. it should be big number e.g. 20000000, you can reference kaldi recipe setting (kaldi/egs/wsj/s5/steps/nnet3/get_degs.sh) 
   ## 13.01
   ## WER
   ## dev-clean dev-other test-clean test-other at grep -rn "WER:" logs/wav2vec-u2_from_scratch_stage11_apply_vad_and_kaldi_decoder.log
   ##  13.01      15.92     12.95      17.44 

fi

### 
### so I will online get mfcc feature , then train kmeans model.
### it is same as hubert iter1 setting ,
#### however 1. the mfcc feature is from no silence audio of librispeech now.
####         2. k-means cluster is setting  64.
### this mfcc pseudo_label is used as wav2vec-u2 auxiliary ssl loss
if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ];then
  echo "step12: train k-means model using 10% no silence train-960.tsv mfcc feature"
  tsv_dir=/workspace2/maduo/dataset/format/librispeech/librispeech_no_silence
  #feat_dir=$tsv_dir/mfcc_no_silence
  km_path=$tsv_dir/mfcc_no_silence/no_silence_librispeech-960h_0.1_portion_mfcc_km_100_clusters.mdl
  n_cluster=64
  nj=35
  sample_rate=16000
  portion=0.1
  for name in  train;do
    python source_md/wav2vec-u2/sklearn_kmeans_on_mfcc.py \
      --tsvfile $tsv_dir/${name}.tsv\
      --n-clusters ${n_cluster}\
      --nj $nj \
      --sample-rate ${sample_rate}\
      --portion ${portion}\
      --km-path ${km_path}
  done
fi

if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ];then
  echo "step13: get k-means pesudo target from k-means model"
  tsv_dir=/workspace2/maduo/dataset/format/librispeech/librispeech_no_silence 
  km_path=$tsv_dir/mfcc_no_silence/no_silence_librispeech-960h_0.1_portion_mfcc_km_100_clusters.mdl
  nj=35
  sample_rate=16000
  lab_dir=$tsv_dir/mfcc_no_silence/mfcc_lab
  mkdir  -p $lab_dir
  for name in dev-clean dev-other train;do
    python source_md/wav2vec-u2/dump_pseudo_label_on_mfcc.py\
      --tsvfile $tsv_dir/${name}.tsv \
      --nj $nj \
      --sample-rate ${sample_rate}\
      --label-path ${lab_dir}/$name.km\
      --km-path $km_path

  done
fi

if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ];then
   echo "create a dummpy dictionary similar to hubert dictionary"
   dest_dir=/workspace2/maduo/dataset/format/librispeech/librispeech_no_silence
   lab_dir=$dest_dir/mfcc_no_silence/mfcc_lab
   n_cluster=64
   for x in $(seq 0 $((n_cluster - 1 )));do
     echo "$x 1"
   done>>$lab_dir/dict.km.txt
fi




if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ];then
    echo "get monophoneme dictionary and word n-gram and phnone n-gram from text(librispeech lm text)"
    bash source_md/wav2vec-u2/prepared_text_for_wav2vec-u2.sh --stage 1 --stop-stage 11
fi


if [ ${stage} -le 400 ] && [ ${stop_stage} -ge 400 ];then
  des_dir=/workspace2/maduo/dataset/format/librispeech
  TASK_DATA=$des_dir/wav2vec_large_feat_dir_no_silence/ ## it stores wav2vec2 large model 15 layer representation of raw librispeech speech,
                                                        ## it  removes  silence. it offers feature of speech.
  cp -r dataset/format/librispeech/librispeech_no_silence/mfcc_no_silence/mfcc_lab/* $TASK_DATA  ## it offers hubert mfcc pesudo label

  # Unpaired text input
  TEXT_DATA=$des_dir/librispeech_lm_norm_phn_seq/unpair_text_half ## it  offers unpair trainset(train.bin, train.idx), devset (dev-clean.bin.dev-clean.idx, dev-other.bin, dev-other.idx)
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
  KENLM_PATH=$des_dir/librispeech_lm_norm_phn_seq/phonesss/lm.phones.filtered.04.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)
  config_dir=$fairseq_dir/examples/wav2vec/unsupervised

  dir=/workspace2/maduo/exp/
  model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_half_second
  exp_dir=$dir/wav2vec-u2_gan_from_scratch/${model_name}
  mkdir -p $exp_dir
  export PYTHONPATH=$fairseq_dir:$PYTHONPATH
  world_size=1
  update_freq=1
  CUDA_VISIBLE_DEVICES=7  python $fairseq_dir/fairseq_cli/hydra_train.py  --multirun \
       --config-dir $config_dir/config/gan \
     --config-name w2vu2_local1 \
     task.data=${TASK_DATA} \
     task.text_data=${TEXT_DATA} \
     task.kenlm_path=${KENLM_PATH} \
     dataset.train_subset=train\
     dataset.valid_subset=\'dev-other,dev-clean\'\
     dataset.batch_size=160\
     dataset.num_workers=6\
     common.user_dir=${fairseq_dir}/examples/wav2vec/unsupervised \
     model.code_penalty=2,4 model.gradient_penalty=1.5,2.0 \
     model.smoothness_weight=0.5,0.75,1.0 'common.seed=range(0,5)'\
     distributed_training.distributed_world_size=${world_size}\
     distributed_training.distributed_port=-1\
     distributed_training.ddp_backend=legacy_ddp\
     optimization.update_freq=[${update_freq}]\
     common.tensorboard_logdir=$exp_dir\
     hydra.run.dir=$fairseq_dir/examples/wav2vec/unsupervised

fi
