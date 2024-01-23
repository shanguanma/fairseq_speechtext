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
   echo "get final phone dictionary"
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
   cut -f1 -d ' ' $dest_dir/phoness/dict.txt | awk '{print $0 " " NR-1}' > $dest_dir/phoness/dict.phn.txt
   
   echo "finish get final phone dictionary !!!!!!!!!!"
fi



if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ];then
   echo "create a dummpy dictionary similar to hubert dictionary"
   dest_dir=dataset/format/librispeech/librispeech_lm_norm_phn_seq/
   lab_dir=$dest_dir/phoness/
   n_cluster=41
   for x in $(seq 0 $((n_cluster - 1 )));do
     echo "$x 1"
   done>>$lab_dir/dict.phncode.txt
fi
## 2023.5.17
## I will add four specify symbols into mono phoneme dictionary, in order to text phonecode same as speech phonecode
## you  can also see source_md/wav2vec-u2/w2vu2_generate_frame_phncode_deprecated.sh --stage 3 --stop-stage 3  
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ];then
  dest_dir=dataset/format/librispeech/librispeech_lm_norm_phn_seq/
  cat $dest_dir/phoness/dict.txt> $dest_dir/phoness/dict_monophn.txt
  echo "<s> 0" > $dest_dir/phoness/dict_specify_symbols.txt
  echo "<pad> 1" >> $dest_dir/phoness/dict_specify_symbols.txt
  echo "</s> 2" >> $dest_dir/phoness/dict_specify_symbols.txt
  echo "<unk> 3" >> $dest_dir/phoness/dict_specify_symbols.txt
  cat $dest_dir/phoness/dict_specify_symbols.txt $dest_dir/phoness/dict_monophn.txt > $dest_dir/phoness/dict_for_text.txt
  head $dest_dir/phoness/dict_for_text.txt
  tail  $dest_dir/phoness/dict_for_text.txt
  cut -f1 -d ' ' $dest_dir/phoness/dict_for_text.txt | awk '{print $0 " " NR-1}' > $dest_dir/phoness/dict.phn_for_text.txt  
  echo "finish mono phone dictioanry including four specify symbols for converting text phone sequence into phonecode"
  
fi




## in order to get number of text utterances equal to number of wav utterances in sthubert method
if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ];then
   ### speech part
   # wc -l  dataset/format/librispeech/mfcc/mfcc_lab/train-960.km
   # 281241 dataset/format/librispeech/mfcc/mfcc_lab/train-960.km
   # wc -l  dataset/format/librispeech/mfcc/mfcc_lab/dev-other.km
   # 2864 dataset/format/librispeech/mfcc/mfcc_lab/dev-other.km
   # wc -l  dataset/format/librispeech/mfcc/mfcc_lab/dev-clean.km
   # 2703 dataset/format/librispeech/mfcc/mfcc_lab/dev-clean.km

   #### text part
   ## wc -l  dataset/librispeech/librispeech-lm-norm.txt
   ## 40418261 dataset/librispeech/librispeech-lm-norm.txt
   data_dir=dataset/format/librispeech/librispeech_lm_norm_phn_seq/phones
   sed -n "30000,311240p" $data_dir/lm.phones.filtered.txt > $data_dir/lm.phones.filtered.281241.txt
   sed -n "320000,322863p" $data_dir/lm.phones.filtered.txt > $data_dir/lm.phones.filtered.2864.txt
   sed -n "330000,332702p" $data_dir/lm.phones.filtered.txt > $data_dir/lm.phones.filtered.2703.txt
fi

if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ];then
   dest_dir=dataset/format/librispeech/librispeech_lm_norm_phn_seq/
   echo "convert phone utterance into phone index utterance"
   dict=$dest_dir/phoness/dict.phn.txt     
   for name in lm.phones.filtered.281241 lm.phones.filtered.2864 lm.phones.filtered.2703;do
     python source_md/wav2vec-u2/text/phn_to_code.py\
            $dest_dir/phones/${name}.txt\
            $dict\
            $dest_dir/phones/${name}_phncode.txt    
   done 
fi

if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ];then
  echo "in order to use same as train set name( e.g.train-960), dev set name (e.g. dev-clean dev-other)"
  dest_dir=dataset/format/librispeech/librispeech_lm_norm_phn_seq
  cat $dest_dir/phones/lm.phones.filtered.281241_phncode.txt> $dest_dir/train-960.phncode
  cat $dest_dir/phones/lm.phones.filtered.2864_phncode.txt > $dest_dir/dev-other.phncode
  cat $dest_dir/phones/lm.phones.filtered.2703_phncode.txt  > $dest_dir/dev-clean.phncode

fi

if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ];then
 data_dir=dataset/format/librispeech/librispeech_lm_norm_phn_seq
 sed -n "330000,335405p" $data_dir/phones/lm.phones.filtered.txt > $data_dir/phones/lm.phones.filtered.5406.txt
 dest_dir=$data_dir
 dict=$dest_dir/phoness/dict.phn.txt
 for name in lm.phones.filtered.5406;do
     python source_md/wav2vec-u2/text/phn_to_code.py\
            $dest_dir/phones/${name}.txt\
            $dict\
            $dest_dir/phones/${name}_phncode.txt
   done

 cat $dest_dir/phones/lm.phones.filtered.5406_phncode.txt  > $dest_dir/dev-clean-wav.phncode
fi



## using kenlm to train 4-gram following wav2vec-u2
if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ];then
   echo "make 4-gram lm using librispeech lm('librispeech-lm-norm.lid.txt')"
   kenlm_root=kenlm/build/bin
   dest_dir=dataset/format/librispeech/librispeech_lm_norm_phn_seq
   $kenlm_root/lmplz -o 4 < $dest_dir/librispeech-lm-norm.lid.txt\
        --discount_fallback --prune 0 0 0 3 > $dest_dir/kenlm.wrd.o40003.arpa
   $kenlm_root/build_binary $dest_dir/kenlm.wrd.o40003.arpa $dest_dir/kenlm.wrd.o40003.bin
   
fi


## I will get phone sequence of train-clean-100 
if [ ${stage} -le 20 ] && [ ${stop_stage} -ge 20 ];then
   fairseq_dir=/workspace2/maduo/fairseq_speechtext
   dest_dir=dataset/format/librispeech/train-clean-100_paired
   mkdir -p $dest_dir
   data_dir=dataset/format/librispeech/
   echo " get word list from train-clean-100 text"
    #cut -f1 -d ' ' $dest_dir/dict.txt|\
    #              grep -v -x '[[:punct:]]*' | \
    #              grep -Pv '\d\d\d\d\d+' > $dest_dir/words.txt
   python source_md/wav2vec-u2/text/utt2word.py\
           $data_dir/train-clean-100.wrd\
           $dest_dir/train-clean-100.words
   echo "finish get word list from libirspeechlm text !!!!!!!!!!"           
fi
if [ ${stage} -le 22 ] && [ ${stop_stage} -ge 22 ];then
   echo "covert word to phones using g2p"
   dest_dir=dataset/format/librispeech/train-clean-100_paired
   python source_md/wav2vec-u2/text/g2p_wrd_to_phn.py < $dest_dir/train-clean-100.words > $dest_dir/phones.txt
  echo "finish covert word to phones using g2p !!!!!!!!!!!!!!"
  paste $dest_dir/train-clean-100.words $dest_dir/phones.txt > $dest_dir/lexicon.txt
fi

if [ ${stage} -le 23 ] && [ ${stop_stage} -ge 23 ];then
   echo "add silence into phone sequence"
   sil_prob=0.5 ## the of setting of wav2vec-u is 0.25, however, the of setting of wav2vec-u2.0 is 05
   data_dir=dataset/format/librispeech/
   dest_dir=dataset/format/librispeech/train-clean-100_paired
   python  source_md/wav2vec-u2/text/phonemize_with_sil.py\
       -s $sil_prob --surround \
       --lexicon $dest_dir/lexicon.txt\
       < $data_dir/train-clean-100.wrd\
       >$data_dir/train-clean-100.phn
  echo "finish add silence into phone sequence !!!!"
  wc -l $data_dir/train-clean-100.phn
  wc -l $data_dir/train-clean-100.wrd
fi

if [ ${stage} -le 24 ] && [ ${stop_stage} -ge 24 ];then
    dest_dir=dataset/format/librispeech/train-clean-100_paired
    echo "create phone and id  mapping"
    python source_md/wav2vec-u2/phones2dict.py \
     $dest_dir/phones.txt  \
     $dest_dir/phn2id.txt
fi
## i found that phn2id.txt and dataset/format/librispeech/librispeech_lm_norm_phn_seq/phoness/dict.phn.txt contain same phone set
## in order to keep same phone id mapping  at warm up stage(using train-clean-100 paired data(speech code and its phn code))and pretrain stage
## I used dataset/format/librispeech/librispeech_lm_norm_phn_seq/phoness/dict.phn.txt  mapping 
if [ ${stage} -le 26 ] && [ ${stop_stage} -ge 26 ];then
 dest_dir=dataset/format/librispeech/
 dict=$dest_dir/librispeech_lm_norm_phn_seq/phoness/dict.phn.txt
 data_dir=dataset/format/librispeech/
 for name in train-clean-100;do
     python source_md/wav2vec-u2/text/phn_to_code.py\
            $data_dir/$name.phn\
            $dict\
            $data_dir/${name}.phncode
   done
 wc -l $data_dir/train-clean-100.{tsv,phn,phncode,wrd}
fi
