#!/usr/bin/env bash

stage=-100

stop_stage=1000
.  utils/parse_options.sh
#. path_for_fairseq.sh
#. path_for_fsq_speechtext.sh
#. path_for_fsq_sptt.sh ## hltsz_cluster
. path_for_fairseq_speechtext.sh




if [ ${stage} -le -2 ] && [ ${stop_stage} -ge -2 ];then
   echo "prepared Chinese phoneme lexicon from https://github.com/speechio/BigCiDian.git"
   echo "we use wenetspeech lexicon, it can offer English and Chinese phoneme and it can be used to process aishell-2 english case"
   lang_dir=data/local/wenetspeech_dict
   bash codebase/fairseq_speechtext/examples/wav2vec/unsupervised/Chinese/wenetspeech_dict_prep.sh $lang_dir

fi
if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ];then
   echo "normalize aishell-2 trans.txt "
   echo "reference: https://github.com/wenet-e2e/wenet/blob/main/examples/aishell2/s0/local/prepare_data.sh"
   text_dir=dataset/aishell-2/
   dest_dir=dataset/format/Chinese/aishell-2_norm_phn_seq
   mkdir -p $dest_dir
   awk '{print $1}' $text_dir/trans.txt > $dest_dir/trans_utt.list
   dos2unix < $text_dir/trans.txt | \
    utils/filter_scp.pl -f 1 $dest_dir/trans_utt.list - | \
    sort -k 1 | uniq | tr '[a-z]' '[A-Z]' | \
    sed 's/Ａ/A/g' | sed 's/Ｔ/T/g' | sed 's/Ｍ/M/g' | sed 's/𫚉//g' | sed 's/𫖯/頫/g' | \
    sed 's/[()]//g' | sed "s/\([^A-Z]\)'/\1/g" > $dest_dir/text

   awk '{print $2}' $dest_dir/text > $dest_dir/text_nouttid
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
   echo "get unique lexicon from kaldi format lexicon"
   echo "The way to do this is to keep only the first pronunciation of a word in lexicon.txt."
   lang_dir=data/local/wenetspeech_dict
    python /home/maduo/source_md/wav2vec-u2/text/generate_unique_lexicon.py --lang-dir $lang_dir
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
   echo "add silence into phone sequence"
   sil_prob=0.5 ## the of setting of wav2vec-u is 0.25, however, the of setting of wav2vec-u2.0 is 05
   dest_dir=dataset/format/Chinese/aishell-2_norm_phn_seq
   lang_dir=data/local/wenetspeech_dict
   python  source_md/wav2vec-u2/text/phonemize_with_sil.py\
       -s $sil_prob --surround \
       --language_id "Chinese" \
       --lexicon $lang_dir/uniq_lexicon.txt\
       < $dest_dir/text_nouttid\
       >$dest_dir/lm.phones.filtered.txt
  echo "finish add silence into phone sequence !!!!"
fi



if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
   echo "prepared dict.txt, its first column is phoneme , second column is id"
   lang_dir=data/local/wenetspeech_dict
   dest_dir=dataset/format/Chinese/aishell-2_norm_phn_seq
   #cat $lang_dir/uniq_lexicon.txt | awk '{ for(n=2;n<=NF;n++){ phones[$n] = 1; }} END{for (p in phones) print p;}'| \
   # perl -e 'while(<>){ chomp($_); $phone = $_; next if ($phone eq "sil");
   # m:^([^\d]+)(\d*)$: || die "Bad phone $_"; $q{$1} .= "$phone "; }
   # foreach $l (values %q) {print "$l\n";}
   #' | sort -k1 > $lang_dir/uniq_nonsilence_phones.txt  || exit 1;
   python source_md/wav2vec-u2/lexicon2phn.py < $lang_dir/uniq_lexicon.txt > $lang_dir/uniq_nonsilence_phones.txt 
   awk '{print $1 " " NR+1}' $lang_dir/uniq_nonsilence_phones.txt >$dest_dir/dict.txt


fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
   echo "prepare text trainset"
   echo "get final phone dictionary and compress train.txt into train.bin and train.idx for model training"
   fairseq_dir=/home/maduo/codebase/fairseq_speechtext
   dest_dir=dataset/format/Chinese/aishell-2_norm_phn_seq
   output_dir=$dest_dir/unpair_text
   lang_dir=data/local/wenetspeech_dict
   mkdir -p $output_dir
   python $fairseq_dir/fairseq_cli/preprocess.py\
           --dataset-impl mmap\
           --trainpref $dest_dir/lm.phones.filtered.txt\
           --workers 70\
           --only-source \
           --destdir $output_dir\
           --srcdict $dest_dir/dict.txt

   echo "<SIL> 0" >>  $dest_dir/dict.txt
   mv $dest_dir/dict.txt $output_dir
   mv $output_dir/dict.txt $output_dir/dict.phn.txt ## for wav2vec-u2
  
   ## logging is as follows:
   #   2024-01-08 16:32:32 | INFO | fairseq_cli.preprocess | Namespace(no_progress_bar=False, log_interval=100, log_format=None, log_file=None, aim_repo=None, aim_run_hash=None, tensorboard_logdir=None, wandb_project=None, azureml_logging=False, seed=1, cpu=False, tpu=False, bf16=False, memory_efficient_bf16=False, fp16=False, memory_efficient_fp16=False, fp16_no_flatten_grads=False, fp16_init_scale=128, fp16_scale_window=None, fp16_scale_tolerance=0.0, on_cpu_convert_precision=False, min_loss_scale=0.0001, threshold_loss_scale=None, amp=False, amp_batch_retries=2, amp_init_scale=128, amp_scale_window=None, user_dir=None, empty_cache_freq=0, all_gather_list_size=16384, model_parallel_size=1, quantization_config_path=None, profile=False, reset_logging=False, suppress_crashes=False, use_plasma_view=False, plasma_path='/tmp/plasma', criterion='cross_entropy', tokenizer=None, bpe=None, optimizer=None, lr_scheduler='fixed', scoring='bleu', task='translation', source_lang=None, target_lang=None, trainpref='dataset/format/Chinese/aishell-2_norm_phn_seq/lm.phones.filtered.txt', validpref=None, testpref=None, align_suffix=None, destdir='dataset/format/Chinese/aishell-2_norm_phn_seq/unpair_text', thresholdtgt=0, thresholdsrc=0, tgtdict=None, srcdict='dataset/format/Chinese/aishell-2_norm_phn_seq/dict.txt', nwordstgt=-1, nwordssrc=-1, alignfile=None, dataset_impl='mmap', joined_dictionary=False, only_source=True, padding_factor=8, workers=70, dict_only=False)
#2024-01-08 16:32:32 | INFO | fairseq_cli.preprocess | [None] Dictionary: 163 types
#2024-01-08 16:32:44 | INFO | fairseq_cli.preprocess | [None] dataset/format/Chinese/aishell-2_norm_phn_seq/lm.phones.filtered.txt: 1009194 sents, 31325767 tokens, 22.3% replaced (by <unk>)
#2024-01-08 16:32:44 | INFO | fairseq_cli.preprocess | Wrote preprocessed data to dataset/format/Chinese/aishell-2_norm_phn_seq/unpair_text
   echo "finish get final phone dictionary and compress train.txt into train.bin and train.idx for model training"
fi

### training phone level 4-gram for wav2vec-u2 gan training stage.
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
   echo "prepared Chinese phone 4-gram"
   kenlm_root=codebase/kenlm/build/bin
   dest_dir=dataset/format/Chinese/aishell-2_norm_phn_seq
   $kenlm_root/lmplz -o 4 < $dest_dir/lm.phones.filtered.txt\
        --discount_fallback > $dest_dir/lm.phones.filtered.arpa
   $kenlm_root/build_binary $dest_dir/lm.phones.filtered.arpa $dest_dir/lm.phones.filtered.bin

fi

## 2024-1-30 I reorganized the transcript of aishell-2
if  [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then
    dest_dir=dataset/format/Chinese/aishell-2_norm_phn_seq_again
    input_dir=dataset/format/Chinese/aishell-2_norm_phn_seq
    mkdir -p $dest_dir
    cp -r $input_dir/text $dest_dir/train.text
    datasets=train
    lexicon_dir=data/local/wenetspeech_dict
    for name in $datasets;do
      # splits the Chinese words into character and keep the English words
      python codebase/fairseq_speechtext/examples/wav2vec/unsupervised/Chinese/character_tokenizer.py \
          $dest_dir/${name}.text $dest_dir/${name}.text_split

      python codebase/fairseq_speechtext/examples/wav2vec/unsupervised/Chinese/character2phones.py\
           $lexicon_dir/uniq_lexicon.txt\
           $dest_dir/$name.text_split  \
           $dest_dir/$name.pre_phn	
   done
fi 
if  [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ];then

   echo "add silence into phone sequence"

   sil_prob=0.5 ## the of setting of wav2vec-u is 0.25, however, the of setting of wav2vec-u2.0 is 05
   dest_dir=dataset/format/Chinese/aishell-2_norm_phn_seq_again
   lang_dir=data/local/wenetspeech_dict
   datasets=train
   for name in $datasets;do
     ## remove uttid
     awk '{for(i=2;i<=NF;i=i+1) printf " "$i;print ""}' $dest_dir/$name.text_split > $dest_dir/$name.text_split_nouttid
     head $dest_dir/$name.text_split_nouttid
   
   python  codebase/fairseq_speechtext/examples/wav2vec/unsupervised/wav2vec-u2/text/phonemize_with_sil_mixture.py\
       -s $sil_prob --surround \
       --lexicon $lang_dir/uniq_lexicon.txt\
       < $dest_dir/${name}.text_split_nouttid\
       >$dest_dir/lm.phones.filtered.txt
  echo "finish add silence into phone sequence !!!!"
  done
fi



if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ];then
   echo "prepared dict.txt, its first column is phoneme , second column is id"
   lang_dir=data/local/wenetspeech_dict
   dest_dir=dataset/format/Chinese/aishell-2_norm_phn_seq_again

   python source_md/wav2vec-u2/lexicon2phn.py < $lang_dir/uniq_lexicon.txt > $lang_dir/uniq_nonsilence_phones.txt
   awk '{print $1 " " NR+1}' $lang_dir/uniq_nonsilence_phones.txt >$dest_dir/dict.txt


fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ];then
   echo "prepare text trainset"
   echo "get final phone dictionary and compress train.txt into train.bin and train.idx for model training"
   fairseq_dir=/home/maduo/codebase/fairseq_speechtext
   dest_dir=dataset/format/Chinese/aishell-2_norm_phn_seq_again
   output_dir=$dest_dir/unpair_text
   lang_dir=data/local/wenetspeech_dict
   mkdir -p $output_dir
   python $fairseq_dir/fairseq_cli/preprocess.py\
           --dataset-impl mmap\
           --trainpref $dest_dir/lm.phones.filtered.txt\
           --workers 70\
           --only-source \
           --destdir $output_dir\
           --srcdict $dest_dir/dict.txt

   echo "<SIL> 0" >>  $dest_dir/dict.txt
   mv $dest_dir/dict.txt $output_dir
   mv $output_dir/dict.txt $output_dir/dict.phn.txt ## for wav2vec-u2
fi


### training phone level 4-gram for wav2vec-u2 gan training stage.
if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ];then
   echo "prepared Chinese phone 4-gram"
   kenlm_root=codebase/kenlm/build/bin
   dest_dir=dataset/format/Chinese/aishell-2_norm_phn_seq_again
   $kenlm_root/lmplz -o 4 < $dest_dir/lm.phones.filtered.txt\
        --discount_fallback > $dest_dir/lm.phones.filtered.04.arpa
   $kenlm_root/build_binary $dest_dir/lm.phones.filtered.04.arpa $dest_dir/lm.phones.filtered.04.bin

fi


## 2024-1-31 I reorganized the transcript of aishell-2 with no tone phone of wenetspeech dict
if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ];then
   echo "get unique lexicon from kaldi format lexicon"
   echo "The way to do this is to keep only the first pronunciation of a word in lexicon.txt."
   lang_dir=data/local/wenetspeech_dict
    python /home/maduo/source_md/wav2vec-u2/text/generate_unique_lexicon_remove_tone.py --lang-dir $lang_dir
fi

if  [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ];then
    dest_dir=dataset/format/Chinese/aishell-2_norm_phn_seq_remove_tone
    input_dir=dataset/format/Chinese/aishell-2_norm_phn_seq
    mkdir -p $dest_dir
    cp -r $input_dir/text $dest_dir/train.text
    datasets=train
    lexicon_dir=data/local/wenetspeech_dict
    for name in $datasets;do
      # splits the Chinese words into character and keep the English words
      python codebase/fairseq_speechtext/examples/wav2vec/unsupervised/Chinese/character_tokenizer.py \
          $dest_dir/${name}.text $dest_dir/${name}.text_split

      python codebase/fairseq_speechtext/examples/wav2vec/unsupervised/Chinese/character2phones.py\
           $lexicon_dir/uniq_lexicon_remove_tone.txt\
           $dest_dir/$name.text_split  \
           $dest_dir/$name.pre_phn
   done
fi
if  [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ];then

   echo "add silence into phone sequence"

   sil_prob=0.5 ## the of setting of wav2vec-u is 0.25, however, the of setting of wav2vec-u2.0 is 05
   dest_dir=dataset/format/Chinese/aishell-2_norm_phn_seq_remove_tone
   lang_dir=data/local/wenetspeech_dict
   datasets=train
   for name in $datasets;do
     ## remove uttid
     awk '{for(i=2;i<=NF;i=i+1) printf " "$i;print ""}' $dest_dir/$name.text_split > $dest_dir/$name.text_split_nouttid
     head $dest_dir/$name.text_split_nouttid

   python  codebase/fairseq_speechtext/examples/wav2vec/unsupervised/wav2vec-u2/text/phonemize_with_sil_mixture.py\
       -s $sil_prob --surround \
       --lexicon $lang_dir/uniq_lexicon_remove_tone.txt\
       < $dest_dir/${name}.text_split_nouttid\
       >$dest_dir/lm.phones.filtered.txt
  echo "finish add silence into phone sequence !!!!"
  done
fi

if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ];then
   echo "prepared dict.txt, its first column is phoneme , second column is id"
   lang_dir=data/local/wenetspeech_dict
   dest_dir=dataset/format/Chinese/aishell-2_norm_phn_seq_remove_tone

   python source_md/wav2vec-u2/lexicon2phn.py < $lang_dir/uniq_lexicon_remove_tone.txt > $lang_dir/uniq_nonsilence_phones_remove_tone.txt
   awk '{print $1 " " NR+1}' $lang_dir/uniq_nonsilence_phones_remove_tone.txt >$dest_dir/dict.txt


fi

if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ];then
   echo "prepare text trainset"
   echo "get final phone dictionary and compress train.txt into train.bin and train.idx for model training"
   fairseq_dir=/home/maduo/codebase/fairseq_speechtext
   dest_dir=dataset/format/Chinese/aishell-2_norm_phn_seq_remove_tone
   output_dir=$dest_dir/unpair_text
   lang_dir=data/local/wenetspeech_dict
   mkdir -p $output_dir
   python $fairseq_dir/fairseq_cli/preprocess.py\
           --dataset-impl mmap\
           --trainpref $dest_dir/lm.phones.filtered.txt\
           --workers 70\
           --only-source \
           --destdir $output_dir\
           --srcdict $dest_dir/dict.txt

   echo "<SIL> 0" >>  $dest_dir/dict.txt
   mv $dest_dir/dict.txt $output_dir
   mv $output_dir/dict.txt $output_dir/dict.phn.txt ## for wav2vec-u2
fi


### training phone level 4-gram for wav2vec-u2 gan training stage.
if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ];then
   echo "prepared Chinese phone 4-gram"
   kenlm_root=codebase/kenlm/build/bin
   dest_dir=dataset/format/Chinese/aishell-2_norm_phn_seq_remove_tone
   $kenlm_root/lmplz -o 4 < $dest_dir/lm.phones.filtered.txt\
        --discount_fallback > $dest_dir/lm.phones.filtered.04.arpa
   $kenlm_root/build_binary $dest_dir/lm.phones.filtered.04.arpa $dest_dir/lm.phones.filtered.04.bin

fi


## 2024-2-19 I want to reduce add silence probablity. because Chinese  is very few pauses between words.

if  [ ${stage} -le 16 ] && [ ${stop_stage} -ge 16 ];then
    dest_dir=dataset/format/Chinese/aishell-2_norm_phn_seq_remove_tone_sil_0.25
    input_dir=dataset/format/Chinese/aishell-2_norm_phn_seq
    mkdir -p $dest_dir
    cp -r $input_dir/text $dest_dir/train.text
    datasets=train
    lexicon_dir=data/local/wenetspeech_dict
    for name in $datasets;do
      # splits the Chinese words into character and keep the English words
      python codebase/fairseq_speechtext/examples/wav2vec/unsupervised/Chinese/character_tokenizer.py \
          $dest_dir/${name}.text $dest_dir/${name}.text_split

      python codebase/fairseq_speechtext/examples/wav2vec/unsupervised/Chinese/character2phones.py\
           $lexicon_dir/uniq_lexicon_remove_tone.txt\
           $dest_dir/$name.text_split  \
           $dest_dir/$name.pre_phn
   done
fi

if  [ ${stage} -le 17 ] && [ ${stop_stage} -ge 17 ];then

   echo "add silence into phone sequence"

   sil_prob=0.25 ## the of setting of wav2vec-u is 0.25, however, the of setting of wav2vec-u2.0 is 05
   dest_dir=dataset/format/Chinese/aishell-2_norm_phn_seq_remove_tone_sil_0.25
   lang_dir=data/local/wenetspeech_dict
   datasets=train
   for name in $datasets;do
     ## remove uttid
     awk '{for(i=2;i<=NF;i=i+1) printf " "$i;print ""}' $dest_dir/$name.text_split > $dest_dir/$name.text_split_nouttid
     head $dest_dir/$name.text_split_nouttid

   python  codebase/fairseq_speechtext/examples/wav2vec/unsupervised/wav2vec-u2/text/phonemize_with_sil_mixture.py\
       -s $sil_prob --surround \
       --lexicon $lang_dir/uniq_lexicon_remove_tone.txt\
       < $dest_dir/${name}.text_split_nouttid\
       >$dest_dir/lm.phones.filtered.txt
  echo "finish add silence into phone sequence !!!!"
  done
fi

if [ ${stage} -le 18 ] && [ ${stop_stage} -ge 18 ];then
   echo "prepared dict.txt, its first column is phoneme , second column is id"
   lang_dir=data/local/wenetspeech_dict
   dest_dir=dataset/format/Chinese/aishell-2_norm_phn_seq_remove_tone_sil_0.25

   python source_md/wav2vec-u2/lexicon2phn.py < $lang_dir/uniq_lexicon_remove_tone.txt > $lang_dir/uniq_nonsilence_phones_remove_tone.txt
   awk '{print $1 " " NR+1}' $lang_dir/uniq_nonsilence_phones_remove_tone.txt >$dest_dir/dict.txt


fi

if [ ${stage} -le 19 ] && [ ${stop_stage} -ge 19 ];then
   echo "prepare text trainset"
   echo "get final phone dictionary and compress train.txt into train.bin and train.idx for model training"
   fairseq_dir=/home/maduo/codebase/fairseq_speechtext
   dest_dir=dataset/format/Chinese/aishell-2_norm_phn_seq_remove_tone_sil_0.25
   output_dir=$dest_dir/unpair_text
   lang_dir=data/local/wenetspeech_dict
   mkdir -p $output_dir
   python $fairseq_dir/fairseq_cli/preprocess.py\
           --dataset-impl mmap\
           --trainpref $dest_dir/lm.phones.filtered.txt\
           --workers 70\
           --only-source \
           --destdir $output_dir\
           --srcdict $dest_dir/dict.txt

   echo "<SIL> 0" >>  $dest_dir/dict.txt
   mv $dest_dir/dict.txt $output_dir
   mv $output_dir/dict.txt $output_dir/dict.phn.txt ## for wav2vec-u2
fi


### training phone level 4-gram for wav2vec-u2 gan training stage.
if [ ${stage} -le 20 ] && [ ${stop_stage} -ge 20 ];then
   echo "prepared Chinese phone 4-gram"
   kenlm_root=codebase/kenlm/build/bin
   dest_dir=dataset/format/Chinese/aishell-2_norm_phn_seq_remove_tone_sil_0.25
   $kenlm_root/lmplz -o 4 < $dest_dir/lm.phones.filtered.txt\
        --discount_fallback > $dest_dir/lm.phones.filtered.04.arpa
   $kenlm_root/build_binary $dest_dir/lm.phones.filtered.04.arpa $dest_dir/lm.phones.filtered.04.bin

fi


## 2024-2-20 I want to increase add silence probablity

if  [ ${stage} -le 21 ] && [ ${stop_stage} -ge 21 ];then
    dest_dir=dataset/format/Chinese/aishell-2_norm_phn_seq_remove_tone_sil_0.6
    input_dir=dataset/format/Chinese/aishell-2_norm_phn_seq
    mkdir -p $dest_dir
    cp -r $input_dir/text $dest_dir/train.text
    datasets=train
    lexicon_dir=data/local/wenetspeech_dict
    for name in $datasets;do
      # splits the Chinese words into character and keep the English words
      python codebase/fairseq_speechtext/examples/wav2vec/unsupervised/Chinese/character_tokenizer.py \
          $dest_dir/${name}.text $dest_dir/${name}.text_split

      python codebase/fairseq_speechtext/examples/wav2vec/unsupervised/Chinese/character2phones.py\
           $lexicon_dir/uniq_lexicon_remove_tone.txt\
           $dest_dir/$name.text_split  \
           $dest_dir/$name.pre_phn
   done
fi

if  [ ${stage} -le 22 ] && [ ${stop_stage} -ge 22 ];then

   echo "add silence into phone sequence"

   sil_prob=0.6 ## the of setting of wav2vec-u is 0.25, however, the of setting of wav2vec-u2.0 is 05
   dest_dir=dataset/format/Chinese/aishell-2_norm_phn_seq_remove_tone_sil_0.6
   lang_dir=data/local/wenetspeech_dict
   datasets=train
   for name in $datasets;do
     ## remove uttid
     awk '{for(i=2;i<=NF;i=i+1) printf " "$i;print ""}' $dest_dir/$name.text_split > $dest_dir/$name.text_split_nouttid
     head $dest_dir/$name.text_split_nouttid

   python  codebase/fairseq_speechtext/examples/wav2vec/unsupervised/wav2vec-u2/text/phonemize_with_sil_mixture.py\
       -s $sil_prob --surround \
       --lexicon $lang_dir/uniq_lexicon_remove_tone.txt\
       < $dest_dir/${name}.text_split_nouttid\
       >$dest_dir/lm.phones.filtered.txt
  echo "finish add silence into phone sequence !!!!"
  done
fi

if [ ${stage} -le 23 ] && [ ${stop_stage} -ge 23 ];then
   echo "prepared dict.txt, its first column is phoneme , second column is id"
   lang_dir=data/local/wenetspeech_dict
   dest_dir=dataset/format/Chinese/aishell-2_norm_phn_seq_remove_tone_sil_0.6

   python source_md/wav2vec-u2/lexicon2phn.py < $lang_dir/uniq_lexicon_remove_tone.txt > $lang_dir/uniq_nonsilence_phones_remove_tone.txt
   awk '{print $1 " " NR+1}' $lang_dir/uniq_nonsilence_phones_remove_tone.txt >$dest_dir/dict.txt


fi

if [ ${stage} -le 24 ] && [ ${stop_stage} -ge 24 ];then
   echo "prepare text trainset"
   echo "get final phone dictionary and compress train.txt into train.bin and train.idx for model training"
   fairseq_dir=/home/maduo/codebase/fairseq_speechtext
   dest_dir=dataset/format/Chinese/aishell-2_norm_phn_seq_remove_tone_sil_0.6
   output_dir=$dest_dir/unpair_text
   lang_dir=data/local/wenetspeech_dict
   mkdir -p $output_dir
   python $fairseq_dir/fairseq_cli/preprocess.py\
           --dataset-impl mmap\
           --trainpref $dest_dir/lm.phones.filtered.txt\
           --workers 70\
           --only-source \
           --destdir $output_dir\
           --srcdict $dest_dir/dict.txt

   echo "<SIL> 0" >>  $dest_dir/dict.txt
   mv $dest_dir/dict.txt $output_dir
   mv $output_dir/dict.txt $output_dir/dict.phn.txt ## for wav2vec-u2
fi


### training phone level 4-gram for wav2vec-u2 gan training stage.
if [ ${stage} -le 25 ] && [ ${stop_stage} -ge 25 ];then
   echo "prepared Chinese phone 4-gram"
   kenlm_root=codebase/kenlm/build/bin
   dest_dir=dataset/format/Chinese/aishell-2_norm_phn_seq_remove_tone_sil_0.6
   $kenlm_root/lmplz -o 4 < $dest_dir/lm.phones.filtered.txt\
        --discount_fallback > $dest_dir/lm.phones.filtered.04.arpa
   $kenlm_root/build_binary $dest_dir/lm.phones.filtered.04.arpa $dest_dir/lm.phones.filtered.04.bin

fi



## 2024-2-28 I want to add dev phone sequence, 
if  [ ${stage} -le 30 ] && [ ${stop_stage} -ge 30 ];then
   echo "add silence into phone sequence"

   sil_prob=0.5 ## the of setting of wav2vec-u is 0.25, however, the of setting of wav2vec-u2.0 is 05
   dest_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone
   lang_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone/wenetspeech_dict
   dev_text=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir/dev.wrd
   #datasets=train

   python  codebase/fairseq_speechtext/examples/wav2vec/unsupervised/wav2vec-u2/text/phonemize_with_sil_mixture.py\
       -s $sil_prob --surround \
       --lexicon $lang_dir/uniq_lexicon_remove_tone.txt\
       < $dev_text\
       >$dest_dir/dev.phones.txt
  echo "finish add silence into phone sequence !!!!"
fi

if [ ${stage} -le 31 ] && [ ${stop_stage} -ge 31 ];then
   echo "prepare text devset"
   echo "get final phone dictionary and compress train.txt into train.bin and train.idx for model training"
   lang_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone/wenetspeech_dict
   
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   dest_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone
   output_dir=$dest_dir/unpair_text_with_dev
   awk '{print $1 " " NR+1}' $lang_dir/uniq_nonsilence_phones_remove_tone.txt >$dest_dir/dict.txt
   mkdir -p $output_dir
   python $fairseq_dir/fairseq_cli/preprocess.py\
           --dataset-impl mmap\
           --trainpref $dest_dir/dev.phones.txt\
           --workers 70\
           --only-source \
           --destdir $output_dir\
           --srcdict $dest_dir/dict.txt

   echo "<SIL> 0" >>  $dest_dir/dict.txt
   mv $dest_dir/dict.txt $output_dir
   mv $output_dir/dict.txt $output_dir/dict.phn.txt ## for wav2vec-u2
fi

if [ ${stage} -le 32 ] && [ ${stop_stage} -ge 32 ];then
   echo "prepare GAN phone code dictionary"
   lab_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_middle_speech_aishell-2_unpair_text
   n_cluster=59 ## wc -l  /mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone/unpair_text/dict.phn.txt 
   for x in $(seq 0 $((n_cluster - 1 )));do
     echo "$x 1"
   done>>$lab_dir/dict.unsupphncode1.txt

fi



## 2024-2-28 I will filter out phones with lower frequency,Hope it converges faster
# step1: get uniq lexicon  and remove tone from raw lexicon

# step2: remove lower frequecy phone from lexicon and get new lexicon
# step3 : add silence into unpair text

if [ ${stage} -le 35 ] && [ ${stop_stage} -ge 35 ];then
   echo "get unique lexicon from kaldi format lexicon"
   echo "The way to do this is to keep only the first pronunciation of a word in lexicon.txt."
   lang_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone/wenetspeech_dict
   codebase/fairseq_speechtext/examples/wav2vec/unsupervised/wav2vec-u2/text/generate_unique_lexicon_remove_tone.py --lang-dir $lang_dir
fi

if [ ${stage} -le 36 ] && [ ${stop_stage} -ge 36 ];then
   echo "get word phone sequence from unique_lexicon_remove_tone"
   echo "then get high frequecy phone"
   lang_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone/wenetspeech_dict
   cut -f2- -d' ' $lang_dir/uniq_lexicon_remove_tone.txt > $lang_dir/uniq_lexicon_remove_tone_phones.txt
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   min_phones=1000 # wav2vec-u2 is setting to 1000
   target_dir=$lang_dir
   rm /mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone/wenetspeech_dict/phones/dict.txt
   python $fairseq_dir/fairseq_cli/preprocess.py\
       --dataset-impl mmap\
       --trainpref $lang_dir/uniq_lexicon_remove_tone_phones.txt\
       --only-source \
       --destdir $target_dir/phones \
       --thresholdsrc $min_phones --padding-factor 1 --dict-only

fi

if [ ${stage} -le 37 ] && [ ${stop_stage} -ge 37 ];then
    echo "remove lower frequecy phone from lexicon and get new lexicon"
    fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
    target_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone/wenetspeech_dict
    python $fairseq_dir/examples/wav2vec/unsupervised/scripts/filter_lexicon.py \
        -d $target_dir/phones/dict.txt < $target_dir/uniq_lexicon_remove_tone.txt > $target_dir/uniq_lexicon_remove_tone_filtered.lst
fi

if  [ ${stage} -le 38 ] && [ ${stop_stage} -ge 38 ];then
    echo "splits the Chinese words into character and keep the English words"
   
    dest_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone_remove_lower_frequecy_phones
    mkdir -p $dest_dir
    datasets=train
    sil_prob=0.5 ## the of setting of wav2vec-u is 0.25, however, the of setting of wav2vec-u2.0 is 05
    input=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone/train.text
    cat  $input >$dest_dir/train.text
    lang_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone/wenetspeech_dict
    for name in $datasets;do
      # splits the Chinese words into character and keep the English words
      python codebase/fairseq_speechtext/examples/wav2vec/unsupervised/Chinese/character_tokenizer.py \
          $dest_dir/${name}.text $dest_dir/${name}.text_split
     awk '{for(i=2;i<=NF;i=i+1) printf " "$i;print ""}' $dest_dir/$name.text_split > $dest_dir/$name.text_split_nouttid
     head $dest_dir/$name.text_split_nouttid
   done
fi
if  [ ${stage} -le 39 ] && [ ${stop_stage} -ge 39 ];then
    
    echo "add silence into phone sequence"
    dest_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone_remove_lower_frequecy_phones
    mkdir -p $dest_dir
    datasets=train
    sil_prob=0.5 ## the of setting of wav2vec-u is 0.25, however, the of setting of wav2vec-u2.0 is 05
 
    lang_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone/wenetspeech_dict
    for name in $datasets;do
     python  codebase/fairseq_speechtext/examples/wav2vec/unsupervised/wav2vec-u2/text/phonemize_with_sil_mixture.py\
       -s $sil_prob --surround \
       --lexicon $lang_dir/uniq_lexicon_remove_tone_filtered.lst\
       < $dest_dir/${name}.text_split_nouttid\
       >$dest_dir/lm.phones.filtered.txt
     echo "finish add silence into phone sequence !!!!"
    done
fi


if  [ ${stage} -le 40 ] && [ ${stop_stage} -ge 40 ];then
   echo "compress  phone sequence with silence into binary file"
   target_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone/wenetspeech_dict
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   cp $target_dir/phones/dict.txt $target_dir/phones/dict.phn.txt
   echo "<SIL> 0" >> $target_dir/phones/dict.phn.txt

   dest_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone_remove_lower_frequecy_phones
   output_dir=$dest_dir/unpair_text
   mkdir -p $output_dir
   python $fairseq_dir/fairseq_cli/preprocess.py\
       --dataset-impl mmap\
       --trainpref $dest_dir/lm.phones.filtered.txt\
       --workers 70 \
       --only-source \
       --destdir $output_dir \
       --srcdict $target_dir/phones/dict.phn.txt
  echo "finish add silence into phone sequence !!!!"
## logger:
# 2024-02-28 14:49:32 | INFO | fairseq_cli.preprocess | [None] Dictionary: 60 types
# 2024-02-28 14:49:50 | INFO | fairseq_cli.preprocess | [None] /mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone_remove_lower_frequecy_phones/lm.phones.filtered.txt: 991483 sents, 30767779 tokens, 0.0% replaced (by <unk>)
# 2024-02-28 14:49:50 | INFO | fairseq_cli.preprocess | Wrote preprocessed data to /mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone_remove_lower_frequecy_phones/unpair_text
fi

### training phone level 4-gram for wav2vec-u2 gan training stage.
if [ ${stage} -le 41 ] && [ ${stop_stage} -ge 41 ];then
   echo "prepared Chinese phone 4-gram"
   kenlm_root=/mntnfs/lee_data1/maduo/codebase/kenlm/build/bin
   dest_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone_remove_lower_frequecy_phones
   $kenlm_root/lmplz -o 4 < $dest_dir/lm.phones.filtered.txt\
        --discount_fallback > $dest_dir/lm.phones.filtered.04.arpa
   $kenlm_root/build_binary $dest_dir/lm.phones.filtered.04.arpa $dest_dir/lm.phones.filtered.04.bin

fi

## 2024-2-28 I want to add dev phone sequence,
if  [ ${stage} -le 42 ] && [ ${stop_stage} -ge 42 ];then
   echo "add silence into phone sequence"

   sil_prob=0.5 ## the of setting of wav2vec-u is 0.25, however, the of setting of wav2vec-u2.0 is 05
   dest_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone_remove_lower_frequecy_phones
   lang_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone/wenetspeech_dict
   dev_text=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir/dev.wrd
   #datasets=train

   python  codebase/fairseq_speechtext/examples/wav2vec/unsupervised/wav2vec-u2/text/phonemize_with_sil_mixture.py\
       -s $sil_prob --surround \
       --lexicon $lang_dir/uniq_lexicon_remove_tone.txt\
       < $dev_text\
       >$dest_dir/dev.phones.txt
  echo "finish add silence into phone sequence !!!!"
fi

if [ ${stage} -le 43 ] && [ ${stop_stage} -ge 43 ];then
   echo "prepare text devset"
   echo "get final phone dictionary and compress train.txt into train.bin and train.idx for model training"
   lang_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone/wenetspeech_dict

   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   dest_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone_remove_lower_frequecy_phones
   output_dir=$dest_dir/unpair_text_with_dev
   mkdir -p $output_dir


   python $fairseq_dir/fairseq_cli/preprocess.py\
           --dataset-impl mmap\
           --trainpref $dest_dir/dev.phones.txt\
           --workers 70\
           --only-source \
           --destdir $output_dir\
           --srcdict $lang_dir/phones/dict.phn.txt
   mv $output_dir/train.idx $output_dir/dev.idx
   mv $output_dir/train.bin $output_dir/dev.bin
   cp -r $dest_dir/unpair_text/train.* $output_dir/
   
   echo "finish add silence into phone sequence !!!!" 
fi


