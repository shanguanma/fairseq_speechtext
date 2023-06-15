#!/usr/bin/env bash

stage=0

stop_stage=1000
.  utils/parse_options.sh
. path_for_fairseq.sh

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
      --compact < $dest_dir/words.txt > $dest_dir/phones.txt 
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
       >$dest_dir/phones/lm.phones.filtered.txt
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


## now I split lirbispeechlm into three part(one trainset, two devsets) for wav2vec-u2
## I will named them train.bin train.idx dev-clean.bin dev-clean.idx dev-other.bin dev-other.idx, 
## import note: here train or dev-clean or dev-clean contains text utterence can be difference from unpair speech (train.tsv, dev-clean.tsv , dev-other.tsv)
if [ ${stage} -le 19 ] && [ ${stop_stage} -ge 19 ];then
   echo "split lirbispeechlm into three part(one trainset, two devsets)"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   dest_dir=dataset/format/librispeech/librispeech_lm_norm_phn_seq/
   textdata_dir=$dest_dir/phones/
   output_dir=$dest_dir/spilt_three
   mkdir -p $output_dir
   #maduo@lthpc-SYS-4029GP-TRT-BA001:/workspace2/maduo$ wc -l  dataset/format/librispeech/librispeech_lm_norm_phn_seq/phones/lm.phones.filtered.txt
   # 33390030 dataset/format/librispeech/librispeech_lm_norm_phn_seq/phones/lm.phones.filtered.txt
   ## 16695015
   sed -n "2005001,33385030p" $textdata_dir/lm.phones.filtered.txt > $output_dir/train2.txt
   sed -n "1,5000p" $textdata_dir/lm.phones.filtered.txt > $output_dir/train1.txt
   sed -n "5001,10001p" $textdata_dir/lm.phones.filtered.txt > $output_dir/dev-clean.txt
   sed -n "10001,2000000p" $textdata_dir/lm.phones.filtered.txt > $output_dir/train3.txt
   sed -n "2000001, 2005001p" $textdata_dir/lm.phones.filtered.txt > $output_dir/dev-other.txt  ## 


   cat $output_dir/train2.txt $output_dir/train1.txt $output_dir/train3.txt> $output_dir/train.txt

fi
if [ ${stage} -le 20 ] && [ ${stop_stage} -ge 20 ];then
   echo "prepare text trainset"
   echo "get final phone dictionary and compress train.txt into train.bin and train.idx for model training"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   dest_dir=dataset/format/librispeech/librispeech_lm_norm_phn_seq/
   output_dir=$dest_dir/spilt_three
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



if [ ${stage} -le 21 ] && [ ${stop_stage} -ge 21 ];then
   echo "prepare text dev-clean"
   echo "compress dev-clean into train.bin and train.idx for model training"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   dest_dir=dataset/format/librispeech/librispeech_lm_norm_phn_seq/
   output_dir=$dest_dir/spilt_three
   mkdir -p $output_dir/dev-clean
   python $fairseq_dir/fairseq_cli/preprocess.py\
           --dataset-impl mmap\
           --trainpref $output_dir/dev-clean.txt\
           --workers 70\
           --only-source \
           --destdir $output_dir/dev-clean\
           --srcdict $dest_dir/phones/dict.txt
   mv $output_dir/dev-clean/train.idx $output_dir/dev-clean/dev-clean.idx
   mv $output_dir/dev-clean/train.bin $output_dir/dev-clean/dev-clean.bin
   mv $output_dir/dev-clean/dev-clean.bin $output_dir/
   mv $output_dir/dev-clean/dev-clean.idx $output_dir/
   rm -rf $output_dir/dev-clean
   echo "compress dev-clean into train.bin and train.idx for model training"
fi

if [ ${stage} -le 22 ] && [ ${stop_stage} -ge 22 ];then
   echo "prepare text dev-other"
   echo "compress dev-other into train.bin and train.idx for model training"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   dest_dir=dataset/format/librispeech/librispeech_lm_norm_phn_seq/
   output_dir=$dest_dir/spilt_three
   mkdir -p $output_dir/dev-other
   python $fairseq_dir/fairseq_cli/preprocess.py\
           --dataset-impl mmap\
           --trainpref $output_dir/train.txt\
           --workers 70\
           --only-source \
           --destdir $output_dir/dev-other\
           --srcdict $dest_dir/phones/dict.txt
   mv $output_dir/dev-other/train.idx $output_dir/dev-other/dev-other.idx
   mv $output_dir/dev-other/train.bin $output_dir/dev-other/dev-other.bin
   mv $output_dir/dev-other/dev-other.bin $output_dir/
   mv $output_dir/dev-other/dev-other.idx $output_dir/
   rm -rf $output_dir/dev-other
   echo "  compress dev-other into train.bin and train.idx for model training"
fi
if [ ${stage} -le 23 ] && [ ${stop_stage} -ge 23 ];then
  echo "copy phone kenlm into specify target"
  
  dest_dir=/workspace2/maduo/dataset/format/librispeech/librispeech_lm_norm_phn_seq/
  target_dir=$dest_dir/spilt_three ## they are used to train wav2vec-u2, 
                                   ## they contains dict.phn.txt(41monophone dictionary), phone level 4-gram kenlm(lm.phones.filtered.04.bin), 
                                   ## three phone level text datsets(train.bin train.idx dev-clean.bin dev-clean.idx dev-other.bin dev-other.idx)
  cp -r $dest_dir/phonesss/lm.phones.filtered.04.bin $target_dir
  echo "target files"
  ls $target_dir
fi


### in order to eval text utterance effect wav2vec-u2 performance
### now I cut half of librispeechlm utterance number and prepare
### three part(one trainset, two devsets) for wav2vec-u2
## I will named them train.bin train.idx dev-clean.bin dev-clean.idx dev-other.bin dev-other.idx,
## import note: here train or dev-clean or dev-clean contains text utterence can be difference from unpair speech (train.tsv, dev-clean.tsv , dev-other.tsv)
#16695015
if [ ${stage} -le 25 ] && [ ${stop_stage} -ge 25 ];then
   echo "split lirbispeechlm into three part(one trainset, two devsets)"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   dest_dir=dataset/format/librispeech/librispeech_lm_norm_phn_seq/
   textdata_dir=$dest_dir/phones/
   output_dir=$dest_dir/spilt_three_half
   mkdir -p $output_dir
   #maduo@lthpc-SYS-4029GP-TRT-BA001:/workspace2/maduo$ wc -l  dataset/format/librispeech/librispeech_lm_norm_phn_seq/phones/lm.phones.filtered.txt
   # 33390030 dataset/format/librispeech/librispeech_lm_norm_phn_seq/phones/lm.phones.filtered.txt
   ## 33390030/2  = 16695015.0
   sed -n "2005002,16695015p" $textdata_dir/lm.phones.filtered.txt > $output_dir/train2.txt
   sed -n "1,5000p" $textdata_dir/lm.phones.filtered.txt > $output_dir/train1.txt
   sed -n "5001,10001p" $textdata_dir/lm.phones.filtered.txt > $output_dir/dev-clean.txt
   sed -n "10001,2000000p" $textdata_dir/lm.phones.filtered.txt > $output_dir/train3.txt
   sed -n "2000001, 2005001p" $textdata_dir/lm.phones.filtered.txt > $output_dir/dev-other.txt  ##   
   cat $output_dir/train2.txt $output_dir/train1.txt $output_dir/train3.txt > $output_dir/train.txt
fi

if [ ${stage} -le 26 ] && [ ${stop_stage} -ge 26 ];then
   echo "prepare text trainset"
   echo "get final phone dictionary and compress train.txt into train.bin and train.idx for model training"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   dest_dir=dataset/format/librispeech/librispeech_lm_norm_phn_seq/
   output_dir=$dest_dir/spilt_three_half
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

if [ ${stage} -le 27 ] && [ ${stop_stage} -ge 27 ];then
   echo "prepared target dir"
   dest_dir=dataset/format/librispeech/librispeech_lm_norm_phn_seq/
   output_dir=$dest_dir/spilt_three_half  #### they are used to train wav2vec-u2,
                                   ## they contains dict.phn.txt(41monophone dictionary), phone level 4-gram kenlm(lm.phones.filtered.04.bin),
                                   ## three phone level text datsets(train.bin train.idx dev-clean.bin dev-clean.idx dev-other.bin dev-other.idx)
   cp -r $dest_dir/spilt_three/{dev-clean.idx,dev-clean.bin,dev-other.idx,dev-other.bin} $output_dir
   cp -r $dest_dir/phonesss/lm.phones.filtered.04.bin $output_dir
   echo "target files"
   ls $output_dir

fi



if [ ${stage} -le 30 ] && [ ${stop_stage} -ge 30 ];then
   echo "split lirbispeechlm into three part(one trainset, two devsets)"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   dest_dir=dataset/format/librispeech/librispeech_lm_norm_phn_seq/
   textdata_dir=$dest_dir/phones/
   output_dir=$dest_dir/unpair_text_half
   mkdir -p $output_dir
   #maduo@lthpc-SYS-4029GP-TRT-BA001:/workspace2/maduo$ wc -l  dataset/format/librispeech/librispeech_lm_norm_phn_seq/phones/lm.phones.filtered.txt
   # 33390030 dataset/format/librispeech/librispeech_lm_norm_phn_seq/phones/lm.phones.filtered.txt
   ## 33390030/2  = 16695015.0
   sed -n "1,16695015p" $textdata_dir/lm.phones.filtered.txt > $output_dir/train.txt

fi

if [ ${stage} -le 31 ] && [ ${stop_stage} -ge 31 ];then
   echo "prepare text trainset"
   echo "get final phone dictionary and compress train.txt into train.bin and train.idx for model training"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   dest_dir=dataset/format/librispeech/librispeech_lm_norm_phn_seq/
   output_dir=$dest_dir/unpair_text_half
   python $fairseq_dir/fairseq_cli/preprocess.py\
           --dataset-impl mmap\
           --trainpref $output_dir/train.txt\
           --workers 70\
           --only-source \
           --destdir $output_dir\
           --srcdict $dest_dir/phones/dict.txt

   echo "<SIL> 0" >>  $output_dir/dict.txt
   mv $output_dir/dict.txt $output_dir/dict.phn.txt ## for wav2vec-u2
   #cp -r 
   echo "finish get final phone dictionary and compress train.txt into train.bin and train.idx for model training"
fi

if [ ${stage} -le 33 ] && [ ${stop_stage} -ge 33 ];then
   echo "split lirbispeechlm into three part(one trainset, two devsets)"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   dest_dir=dataset/format/librispeech/librispeech_lm_norm_phn_seq/
   textdata_dir=$dest_dir/phones/
   output_dir=$dest_dir/unpair_text_lasthalf
   mkdir -p $output_dir
   #maduo@lthpc-SYS-4029GP-TRT-BA001:/workspace2/maduo$ wc -l  dataset/format/librispeech/librispeech_lm_norm_phn_seq/phones/lm.phones.filtered.txt
   # 33390030 dataset/format/librispeech/librispeech_lm_norm_phn_seq/phones/lm.phones.filtered.txt
   ## 33390030/2  = 16695015.0
   sed -n "16695015,33390030p" $textdata_dir/lm.phones.filtered.txt > $output_dir/train.txt

fi

if [ ${stage} -le 34 ] && [ ${stop_stage} -ge 34 ];then
   echo "prepare text trainset"
   echo "get final phone dictionary and compress train.txt into train.bin and train.idx for model training"
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   dest_dir=dataset/format/librispeech/librispeech_lm_norm_phn_seq/
   output_dir=$dest_dir/unpair_text_lasthalf
   python $fairseq_dir/fairseq_cli/preprocess.py\
           --dataset-impl mmap\
           --trainpref $output_dir/train.txt\
           --workers 70\
           --only-source \
           --destdir $output_dir\
           --srcdict $dest_dir/phones/dict.txt

   echo "<SIL> 0" >>  $output_dir/dict.txt
   mv $output_dir/dict.txt $output_dir/dict.phn.txt ## for wav2vec-u2
   #cp -r
   echo "finish get final phone dictionary and compress train.txt into train.bin and train.idx for model training"
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


## using kenlm(word) to train 4-gram using half of librispeechlm, in order to eval imls-ssl model perfromance
if [ ${stage} -le 40 ] && [ ${stop_stage} -ge 40 ];then
   echo "make word level 4-gram lm using librispeech lm('librispeech-lm-norm.lid.txt')"
   kenlm_root=kenlm/build/bin
   dest_dir=dataset/format/librispeech/librispeech_lm_norm_phn_seq
   sed -n "1, 16857966p" $dest_dir/librispeech-lm-norm.lid.txt > $dest_dir/librispeech-lm-norm.lid_first_half.txt
   $kenlm_root/lmplz -o 4 < $dest_dir/librispeech-lm-norm.lid_first_half.txt\
        --discount_fallback --prune 0 0 0 3 > $dest_dir/kenlm.wrd_first_half.o40003.arpa
   $kenlm_root/build_binary $dest_dir/kenlm.wrd_first_half.o40003.arpa $dest_dir/kenlm.wrd_first_half.o40003.bin
fi
