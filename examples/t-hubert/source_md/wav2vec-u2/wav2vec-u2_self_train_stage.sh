#!/usr/bin/env bash


stage=1

stop_stage=1000
. "/home/maduo/miniconda3/etc/profile.d/conda.sh"
conda activate kaldi
. path.sh ## it is offical kaldi, not pykaldi
. cmd.sh
.  utils/parse_options.sh
root_dir=/workspace2/maduo/
# it will offer {dev-clean,dev-other,train}.{npy,lengths,phn}, dictionary: dict.phn.txt (it is phone2id)
w2v_dir=$root_dir/dataset/format/librispeech/wav2vec_large_feat_dir_no_silence
## wav2vev-u2 generate phone sequence, it has removed silence.  it contains dev-clean.txt dev-other.txt train.txt
lab_dir=$root_dir/exp/wav2vec-u2/hyps_debug_for_apply_vad
out_dir=$root_dir/exp/wav2vec-u2/hyps_debug_for_apply_vad/self_train_dir
arpa_lm=$root_dir/dataset/format/librispeech/librispeech_lm_norm_phn_seq/phonesss/lm.phones.filtered.06.arpa  # phone LM
arpa_lm_bin=$root_dir/dataset/format/librispeech/librispeech_lm_norm_phn_seq/phonesss/lm.phones.filtered.06.bin  # (binary) phone LM for KenLM, used in unsupervised selection
label=phn
train_name="train"
valid_name="dev-clean dev-other"
data_dir=${out_dir}/data
mkdir -p ${out_dir}/exp
fairseq_dir=/workspace2/maduo/fairseq_speechtext/examples/wav2vec/unsupervised/kaldi_self_train/st

## start reproduct self-training + kaldi-lm decoding
## now I use other people trained wav2vec-u2 model to eval self-train + kaldi-lm decoding pipeline
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
   echo "prepared lang and phone lm using only phone2id dictionary"
   $fairseq_dir/local/prepare_lang.sh $w2v_dir/dict.${label}.txt $data_dir
   $fairseq_dir/local/prepare_lm.sh $arpa_lm $data_dir
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
  echo "prepare pseudo/ground truth kaldi format data, however, its text is phone seqeunce or word sequence"
  #for x in $train_name $valid_name; do
  for x in $valid_name; do
   x_gt=${x}_gt

   # prepare phone level pseudo data
   ## this must using python3 not python, because offical kaldi has two python env(python=2.7 and python=3.*)
   python3 $fairseq_dir/local/prepare_data_from_w2v.py $w2v_dir $data_dir $x
   steps/compute_cmvn_stats.sh $data_dir/$x $out_dir/exp/make_feat/$x $out_dir/feats/$x
   python3 $fairseq_dir/local/copy_aligned_text.py < $lab_dir/$x.txt > $data_dir/$x/text

   # prepare phone level ground truth data
   mkdir -p $data_dir/$x_gt
   cp $data_dir/$x/{feats.scp,cmvn.scp,utt2spk,spk2utt} $data_dir/$x_gt/
   python3 $fairseq_dir/local/copy_aligned_text.py < $w2v_dir/$x.$label > $data_dir/$x_gt/text
  done
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
   echo "train monophone hmm, triphone hmm"
   $fairseq_dir/local/train_subset_lgbeam.sh \
     --out_root ${out_dir} --out_name exp --train $train_name --valid $valid_name \
     --mono_size 2000 --tri1_size 5000 --tri2b_size -1 --tri3b_size -1 \
     --stage 1 --max_stage 4 $data_dir $data_dir/lang $data_dir/lang_test   
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
   echo "select phone sequence" 
   exp_root=$out_dir/exp 
   splits="dev_other dev_clean"
   get_best_wer=true
   dec_name="decode"
   graph_name="graph"
   #kenlm_path=/checkpoint/abaevski/data/speech/libri/librispeech_lm_novox.phnc_o6.bin
   kenlm_path=$root_dir/dataset/format/librispeech/librispeech_lm_norm_phn_seq/phonesss/lm.phones.filtered.06.bin
    echo "==== PER w.r.t. real transcript (select based on unsupervised metric)"
    for split in $splits;do
      ref_txt=$data_dir/${split}_gt/text # ground truth phone transcript path
      psd_txt=$data_dir/${split}/text    # pseudo phone transcript path
     for x in $exp_root/*/${dec_name}_${split}*; do
      lang=$(dirname $x)/$graph_name
     (
      for tra in $x/scoring/*.tra; do
        cat $tra | utils/int2sym.pl -f 2- $lang/words.txt | sed 's:<UNK>::g' | sed 's:<SIL>::g' > $tra.txt
        python $fairseq_dir/local/unsup_select.py $psd_txt $tra.txt --kenlm_path $kenlm_path --gt_tra $ref_txt $unsup_args
      done 2>/dev/null | grep "score=" | sed 's/=/ /g' | sed 's/;//g' | sort -k3n | head -n1
     ) &
    done
   done
wait   
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then
   echo "decoding dataset into phones sequence using valid PER to select best decode lmweight"
   out_dir=$out_dir 
   dec_lmparam=  # LM hyperparameters (e.g., 7.0.0) it is from the above stage4 output
   dec_exp=tri2b
   dec_script="steps/decode.sh"
   dec_splits="dev-clean dev-other train"
   dec_data_dir=$out_dir/dec_data  # where to write HMM output
   data_dir=${out_dir}/data

   $fairseq_dir/local/decode.sh --nj 40 --graph_name graph \
     --val_sets "$dec_splits" --decode_script $dec_script \
     $out_dir/exp/$dec_exp $data_dir $data_dir/lang_test

   if [ ! -z $dec_lmparam ]; then
    for x in $dec_splits; do
     mkdir -p $dec_data_dir/$x
     cp $data_dir/$x/{feats.scp,cmvn.scp,utt2spk,spk2utt} $dec_data_dir/$x/
  
     tra=$out_dir/exp/$dec_exp/decode_${x}/scoring/${dec_lmparam}.tra
     cat $tra | utils/int2sym.pl -f 2- $data_dir/lang/words.txt | \
      sed 's:<UNK>::g' | sed 's:<SIL>::g' > $dec_data_dir/${x}/text
     utils/fix_data_dir.sh $dec_data_dir/${x}
    echo "PER on ${x} is" $(compute-wer ark:$data_dir/${x}_gt/text ark:$dec_data_dir/$x/text | cut -d" " -f2-)
   done
  fi  
fi
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ];then
   echo "decoding dataset into phones sequence using valid PER to select best decode lmweight"
   out_dir=$out_dir
   dec_lmparam=  # LM hyperparameters (e.g., 7.0.0)
   dec_exp=tri3b
   dec_script="steps/decode_fmllr.sh"
   dec_splits="dev-clean dev-other train"
   dec_data_dir=$out_dir/dec_data_fmllr  # where to write HMM output
   data_dir=${out_dir}/data

   $fairseq_dir/local/decode.sh --nj 40 --graph_name graph \
     --val_sets "$dec_splits" --decode_script $dec_script \
     $out_dir/exp/$dec_exp $data_dir $data_dir/lang_test

   if [ ! -z $dec_lmparam ]; then
    for x in $dec_splits; do
     mkdir -p $dec_data_dir/$x
     cp $data_dir/$x/{feats.scp,cmvn.scp,utt2spk,spk2utt} $dec_data_dir/$x/

     tra=$out_dir/exp/$dec_exp/decode_${x}/scoring/${dec_lmparam}.tra
     cat $tra | utils/int2sym.pl -f 2- $data_dir/lang/words.txt | \
      sed 's:<UNK>::g' | sed 's:<SIL>::g' > $dec_data_dir/${x}/text
     utils/fix_data_dir.sh $dec_data_dir/${x}
    echo "PER on ${x} is" $(compute-wer ark:$data_dir/${x}_gt/text ark:$dec_data_dir/$x/text | cut -d" " -f2-)
   done
  fi
fi

#### the above is decoding them into phone sequence using kaldi
### here we are decoding them into word sequence using kaldi
if [ ${stage} -le 17 ] && [ ${stop_stage} -ge 17 ];then
   echo "prepare word lang and wordlm lang_test"
   lexicon=dataset/format/librispeech/librispeech_lm_norm_phn_seq/lexicon_filtered.lst  # word to phone mapping
   wrd_arpa_lm=dataset/format/librispeech/librispeech_lm_norm_phn_seq/kenlm.wrd.o40003.arpa  # word LM
   wrd_arpa_lm_bin=dataset/format/librispeech/librispeech_lm_norm_phn_seq/kenlm.wrd.o40003.bin  # word LM for KenLM, used in unsupervised selection
   phn_label=phn
   wrd_label=wrd

   data_dir=$out_dir/data
   wrd_data_dir=$out_dir/data_word

   lexicon_clean=$(mktemp)
    cat $lexicon | sort | uniq > $lexicon_clean
   $fairseq_dir/local/prepare_lang_word.sh $w2v_dir/dict.${phn_label}.txt $data_dir $lexicon_clean && rm $lexicon_clean
   $fairseq_dir/local/prepare_lm.sh --langdir $data_dir/lang_word --lmdir $data_dir/lang_test_word $wrd_arpa_lm $data_dir
    
fi


if [ ${stage} -le 18 ] && [ ${stop_stage} -ge 18 ];then
  echo "prepare dataset word level groud truth kaldi format "
  dec_splits="dev-clean dev-other train"
  wrd_data_dir=$out_dir/data_word
  data_dir=$out_dir/data
  wrd_label=wrd
  for x in $dec_splits; do
   x_gt=${x}_gt
   mkdir -p $wrd_data_dir/$x_gt
   cp $data_dir/$x_gt/{feats.scp,cmvn.scp,utt2spk,spk2utt} $wrd_data_dir/$x_gt/
   python $fairseq_dir/local/copy_aligned_text.py < $w2v_dir/$x.$wrd_label > $wrd_data_dir/$x_gt/text
  done
fi


if [ ${stage} -le 19 ] && [ ${stop_stage} -ge 19 ];then
   dec_suffix=word
   dec_splits="dev-clean dev-other train"
   dec_script="steps/decode_fmllr.sh"  # what decoding script to use (e.g., steps/decode_fmllr.sh)
   dec_exp=tri3b  # what HMM stage to decode (e.g., tri3b)
   $fairseq_dir/local/decode.sh --nj 40 --graph_name graph${dec_suffix} --decode_suffix $dec_suffix \
     --val_sets "$dec_splits" --decode_script $dec_script \
     $out_dir/exp/$dec_exp $data_dir $data_dir/lang_test_word 
fi

if [ ${stage} -le 19 ] && [ ${stop_stage} -ge 19 ];then
   echo "unsup select word sequence via word error rate (WER) of valid dataset  "
   graph_name=graphword
   dec_name=decodeword
   valid_splits="dev-clean dev-other"
   exp_root=$out_dir/exp
   phonemize_lexicon=$data_dir/local/dict_word/lexicon.txt
   echo "==== WER w.r.t. real transcript (select based on unsupervised metric)"
   for split in valid_splits; do
     ref_txt=$wrd_data_dir/${split}_gt/text # ground truth word transcript path
     psd_txt=$data_dir/${split}/text        # pseudo phone transcript path
     for x in $exp_root/*/${dec_name}_${split}*; do
      lang=$(dirname $x)/$graph_name
      for tra in $x/scoring/*.tra; do
        cat $tra | utils/int2sym.pl -f 2- $lang/words.txt | sed 's:\<UNK\>::g' > $tra.txt
        python $fairseq_dir/local/unsup_select.py $psd_txt $tra.txt \
           --kenlm_path $kenlm_path --gt_tra $ref_txt --phonemize \
           --phonemize_lexicon "$phonemize_lexicon"
       done | grep "score=" | sed 's/=/ /g' | sed 's/;//g' | sort -k3n | head -n1
    done
  done
fi


if [ ${stage} -le 20 ] && [ ${stop_stage} -ge 20 ];then  
  out_dir=$out_dir 
  dec_lmparam=  # LM hyperparameters (e.g., 7.0.0) it based on stage19 output to get final trainset(train) best word sequence
  dec_exp=tri3b  # what HMM stage to decode (e.g., tri3b)
  dec_suffix=word
  dec_splits="train dev-clean dev-other"
  dec_data_dir=$out_dir/dec_data_word  # where to write HMM output
  data_dir=$out_dir/data
  wrd_data_dir=$out_dir/data_word
  for x in $dec_splits; do
     mkdir -p $dec_data_dir/$x
     cp $data_dir/$x/{feats.scp,cmvn.scp,utt2spk,spk2utt} $dec_data_dir/$x/

     tra=$out_dir/exp/$dec_exp/decode${dec_suffix}_${x}/scoring/${dec_lmparam}.tra
     cat $tra | utils/int2sym.pl -f 2- $data_dir/lang_word/words.txt | \
      sed 's:<UNK>::g' | sed 's:<SIL>::g' > $dec_data_dir/$x/text
     utils/fix_data_dir.sh $dec_data_dir/$x
    echo "WER on $x is" $(compute-wer ark:$wrd_data_dir/${x}_gt/text ark:$dec_data_dir/$x/text | cut -d" " -f2-)
  done
fi
