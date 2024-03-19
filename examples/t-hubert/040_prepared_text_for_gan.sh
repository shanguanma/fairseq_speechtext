#!/usr/bin/env bash

stage=0

stop_stage=1000
.  utils/parse_options.sh
#. path_for_fairseq.sh
#. path_for_fsq_speechtext.sh
. path_for_fsq_sptt.sh
if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ];then
   echo "normalize librispeech-lm-norm using fasttext model"
   lg=en
   lid_path=dataset/librispeech/lid.176.bin
   input_text_dir=dataset/librispeech/
   output_text_dir=dataset/format/librispeech/librispeech_lm_norm_phn_seq/
   for name in librispeech-lm-norm;do
     python source_md/wav2vec-u2/text/normalize_and_filter_text.py\
              --lang $lg\
              --fasttext-model $lid_path\
              --text $input_text_dir/${name}.txt \
              --output $output_text_dir/${name}.lid.tmp.txt
     cat $input_text_dir/${name}.lid.tmp.txt | grep -v '\-\-\-'>$input_text_dir/${name}.lid.txt
   done  

fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   dest_dir=dataset/format/librispeech/librispeech_lm_norm_phn_seq/
   echo " get word list from libirspeechlm text"
   python $fairseq_dir/fairseq_cli/preprocess.py\
           --dataset-impl mmap\
           --trainpref $dest_dir/librispeech-lm-norm.lid.txt\
           --only-source \
           --destdir $dest_dir\
           --thresholdsrc 2\
           --padding-factor 1\
           --dict-only
    cut -f1 -d ' ' $dest_dir/dict.txt|\
                  grep -v -x '[[:punct:]]*' | \
                  grep -Pv '\d\d\d\d\d+' > $dest_dir/words.txt
   echo "finish get word list from libirspeechlm text !!!!!!!!!!"           
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
   echo "covert word to phones using g2p"
   dest_dir=dataset/format/librispeech/librispeech_lm_norm_phn_seq/
   python source_md/wav2vec-u2/text/g2p_wrd_to_phn.py\
      --compact true < $dest_dir/words.txt > $dest_dir/phones.txt 
  echo "finish covert word to phones using g2p !!!!!!!!!!!!!!"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
   echo "get lexicon and remove lower frequence phones to get phones set"
   min_phones=1000
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   dest_dir=dataset/format/librispeech/librispeech_lm_norm_phn_seq/
   paste $dest_dir/words.txt $dest_dir/phones.txt > $dest_dir/lexicon.txt
   python $fairseq_dir/fairseq_cli/preprocess.py\
           --dataset-impl mmap\
           --trainpref $dest_dir/phones.txt\
           --only-source \
           --destdir $dest_dir/phones\
           --thresholdsrc $min_phones\
           --padding-factor 1\
           --dict-only
  echo "finish get lexicon and remove lower frequence phones to get phones set !!!"
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
   echo "these phones don't contain lexicon and remove it from lexicon"
   dest_dir=dataset/format/librispeech/librispeech_lm_norm_phn_seq/
   python source_md/wav2vec-u2/text/filter_lexicon.py\
      -d $dest_dir/phones/dict.txt\
      < $dest_dir/lexicon.txt\
      > $dest_dir/lexicon_filtered.lst
   echo "finish these phones don't contain lexicon and remove it from lexicon !!!"
fi






if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then
   echo "add silence into phone sequence"
   sil_prob=0.5 ## the of setting of wav2vec-u is 0.25, however, the of setting of wav2vec-u2.0 is 05
   dest_dir=dataset/format/librispeech/librispeech_lm_norm_phn_seq/
   python  source_md/wav2vec-u2/text/phonemize_with_sil.py\
       -s $sil_prob --surround \
       --lexicon $dest_dir/lexicon_filtered.lst\
       < $dest_dir/librispeech-lm-norm.lid.txt\
       >$dest_dir/lm.phones.filtered.txt
       #>$dest_dir/phones/lm.phones.filtered.txt ## old style
  echo "finish add silence into phone sequence !!!!"
fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ];then
   echo "get final phone dictionary and compress lm.phones.filtered.txt into train.bin and train.idx for model training"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   dest_dir=dataset/format/librispeech/librispeech_lm_norm_phn_seq/
   python $fairseq_dir/fairseq_cli/preprocess.py\
           --dataset-impl mmap\
           --trainpref $dest_dir/phones/lm.phones.filtered.txt\
           --workers 70\
           --only-source \
           --destdir $dest_dir/phoness\
           --srcdict $dest_dir/phones/dict.txt
    
   echo "<SIL> 0" >>  $dest_dir/phoness/dict.txt
   mkdir -p $dest_dir/phonesss/
   cp $dest_dir/phoness/dict.txt $dest_dir/phonesss/dict.phn.txt ## for wav2vec-u2 
   #cut -f1 -d ' ' $dest_dir/phoness/dict.txt | awk '{print $0 " " NR-1}' > $dest_dir/phoness/dict.phn.txt
   
   echo "finish get final phone dictionary !!!!!!!!!!"
fi

## using kenlm(phn2word) to train 4-gram following wav2vec-u2
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ];then
   echo "make word level 4-gram lm using librispeech lm('librispeech-lm-norm.lid.txt')"
   kenlm_root=kenlm/build/bin
   dest_dir=dataset/format/librispeech/librispeech_lm_norm_phn_seq
   $kenlm_root/lmplz -o 4 < $dest_dir/librispeech-lm-norm.lid.txt\
        --discount_fallback --prune 0 0 0 3 > $dest_dir/kenlm.wrd.o40003.arpa
   $kenlm_root/build_binary $dest_dir/kenlm.wrd.o40003.arpa $dest_dir/kenlm.wrd.o40003.bin
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ];then
  echo "creat HLG phn2words graph "
  fairseq_dir=/workspace2/maduo/fairseq_speechtext
  dest_dir=/workspace2/maduo/dataset/format/librispeech/librispeech_lm_norm_phn_seq/
  kaldi_root=/workspace2/maduo/pykaldi/tools/kaldi   ### it is kaldi path of  pykaldi.
  #lg=en python $fairseq_dir/examples/speech_recognition/kaldi/kaldi_initializer_md.py\
  lg=en python $fairseq_dir/examples/speech_recognition/kaldi/kaldi_initializer.py\
       kaldi_root=$kaldi_root \
       fst_dir=$dest_dir/fst/phn_to_words_sil \
       lm_arpa=$dest_dir/kenlm.wrd.o40003.arpa \
       wav2letter_lexicon=$dest_dir/lexicon_filtered.lst \
       data_dir=$dest_dir/phonesss \
       in_labels=phn "blank_symbol='<SIL>'"

  echo "finish !!!!!!!!!!!!!"
fi
if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ];then
  
  fairseq_dir=/workspace2/maduo/fairseq_speechtext
  dest_dir=/workspace2/maduo/dataset/format/librispeech/librispeech_lm_norm_phn_seq/
  kaldi_root=/workspace2/maduo/pykaldi/tools/kaldi   ### it is kaldi path of  pykaldi.
  #lg=en python $fairseq_dir/examples/speech_recognition/kaldi/kaldi_initializer_md.py\
  lg=en python $fairseq_dir/examples/speech_recognition/kaldi/kaldi_initializer.py\
       kaldi_root=$kaldi_root \
       fst_dir=$dest_dir/fst/phn_to_words\
       lm_arpa=$dest_dir/kenlm.wrd.o40003.arpa \
       wav2letter_lexicon=$dest_dir/lexicon_filtered.lst \
       data_dir=$dest_dir/phonesss\
       in_labels=phn
  echo "finish !!!!!!!!!!!!!"

fi


if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ];then
  kenlm_root=kenlm/build/bin
  dest_dir=/workspace2/maduo/dataset/format/librispeech/librispeech_lm_norm_phn_seq/ ## it must be absolute path.
  $kenlm_root/lmplz -o 4 < $dest_dir/phones/lm.phones.filtered.txt --discount_fallback > $dest_dir/phonesss/lm.phones.filtered.04.arpa
  $kenlm_root/build_binary $dest_dir/phonesss/lm.phones.filtered.04.arpa $dest_dir/phonesss/lm.phones.filtered.04.bin
  $kenlm_root/lmplz -o 6 < $dest_dir/phones/lm.phones.filtered.txt --discount_fallback > $dest_dir/phonesss/lm.phones.filtered.06.arpa
  $kenlm_root/build_binary $dest_dir/phonesss/lm.phones.filtered.06.arpa $dest_dir/phonesss/lm.phones.filtered.06.bin
  echo "finish !!!!!!!!!!!!!"
fi

if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ];then
  echo "creat HLG phn2phn_sil decode graph"
  fairseq_dir=/workspace2/maduo/fairseq_speechtext
  dest_dir=/workspace2/maduo/dataset/format/librispeech/librispeech_lm_norm_phn_seq/
  #kaldi_root=/workspace2/maduo/kaldi   ### this offical kaldi , it works, it can replace pykaldi. 
  #lg=en python $fairseq_dir/examples/speech_recognition/kaldi/kaldi_initializer_md.py \
  kaldi_root=/workspace2/maduo/pykaldi/tools/kaldi   ### it is kaldi path of  pykaldi.
  #lg=en python $fairseq_dir/examples/speech_recognition/kaldi/kaldi_initializer_md.py\
  lg=en python $fairseq_dir/examples/speech_recognition/kaldi/kaldi_initializer.py\
     kaldi_root=$kaldi_root \
     fst_dir=$dest_dir/fst/phn_to_phn_sil \
     lm_arpa=$dest_dir/phonesss/lm.phones.filtered.06.arpa\
     data_dir=$dest_dir/phonesss \
     in_labels=phn "blank_symbol='<SIL>'"

  echo "finish !!!!!!!!!!!!!"
fi


if [ ${stage} -le 35 ] && [ ${stop_stage} -ge 35 ];then
   echo "split lirbispeechlm into three part(one trainset, two devsets)"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   dest_dir=dataset/format/librispeech/librispeech_lm_norm_phn_seq/
   textdata_dir=$dest_dir/phones/
   output_dir=$dest_dir/unpair_text_all
   mkdir -p $output_dir
   #maduo@lthpc-SYS-4029GP-TRT-BA001:/workspace2/maduo$ wc -l  dataset/format/librispeech/librispeech_lm_norm_phn_seq/phones/lm.phones.filtered.txt
   # 33390030 dataset/format/librispeech/librispeech_lm_norm_phn_seq/phones/lm.phones.filtered.txt
   ## 33390030/2  = 16695015.0
   cat $textdata_dir/lm.phones.filtered.txt > $output_dir/train.txt

fi
if [ ${stage} -le 36 ] && [ ${stop_stage} -ge 36 ];then
   echo "prepare text trainset"
   echo "get final phone dictionary and compress train.txt into train.bin and train.idx for model training"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   dest_dir=dataset/format/librispeech/librispeech_lm_norm_phn_seq/
   output_dir=$dest_dir/unpair_text_all
   python $fairseq_dir/fairseq_cli/preprocess.py\
           --dataset-impl mmap\
           --trainpref $output_dir/train.txt\
           --workers 70\
           --only-source \
           --destdir $output_dir\
           --srcdict $dest_dir/phones/dict.txt

   echo "<SIL> 0" >>  $output_dir/dict.txt
   mv $output_dir/dict.txt $output_dir/dict.phn.txt ## for wav2vec-u2

   echo "finish get final phone dictionary and compress train.txt into train.bin and train.idx for model training"
fi

if [ ${stage} -le 83 ] && [ ${stop_stage} -ge 83 ];then
   echo "prepared dict for voicelm"
   lab_dir=dataset/format/librispeech/librispeech_frame_monophncode_using_wav2vec-u2_model/
   n_cluster=41
   for x in $(seq 0 $((n_cluster - 1 )));do
     echo "$x 1"
   done>>$lab_dir/dict.phncode.txt
fi
