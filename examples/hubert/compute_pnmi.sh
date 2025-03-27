#!/usr/bin/env bash

stage=0
stop_stage=100
. utils/parse_options.sh
. path_for_fairseq_speechtext.sh 

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ];then
  align_textgrid_dir=dataset/librispeech/librispeech_alignments/dev-other
  align_phn_dir=dataset/librispeech/librispeech_alignments/dev-other_aliphn
  python source_md/prepared_libri_align.py \
          $align_textgrid_dir $align_phn_dir

  ## meger one file
  cat $align_phn_dir/*/*.aliphn > $align_phn_dir/dev-other.aliphn  
  head $align_phn_dir/dev-other.aliphn
fi
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
   #align_phn_dir=dataset/librispeech/librispeech_alignments/dev-other_aliphn/
   #align_phn_dir=/workspace2/maduo/recipe/kaldi_egs/librispeech/exp/tri5b/decode_tgsmall_dev_other  ## frame level force align phone sequence
   align_phn_dir=/workspace2/maduo/recipe/kaldi_egs/librispeech/exp/tri5b/decode_tgsmall_valid ## frane level force align phone sequence, valid= dev_clean + dev_other 
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   lab_dir=/workspace2/maduo/dataset/format/librispeech/mfcc/mfcc_lab/
   lab_suffix="km"
   devset_name="valid"
   #lab_sets=["dev-other"]
   #phn_sets=["dev-other"]
   python /workspace2/maduo/fairseq_speechtext/examples/hubert/measure_teacher_quality_md.py \
          $tsv_dir\
          $lab_dir\
          $lab_suffix\
          --phn_dir $align_phn_dir\
          --upsample 1\
          --lab_devsets $devset_name\
          --phn_devsets $devset_name

  ## 
  ## mfcc feat, kmeans cluster is 100, as label
  ## how to align frame level phone , its steps are as follows:
  ## first, I use train-clean-100 and train-clean-360 to train gmm-hmm triphoneme system
  ## then, dev-other is decoded by the above gmm-hmm triphoneme system,
  ## then lattice is align to frame-level(via source_md/get_frame_level_ali_phone.sh), and get frame-level phone force alignment as align phone here
  ## dev-other           
  #   ref pur    hyp pur    H(ref)    H(hyp)      MI    MI/H(ref)    ref segL    hyp segL  p_xy shape      frm tot    frm diff    utt tot    utt miss
  #---------  ---------  --------  --------  ------  -----------  ----------  ----------  ------------  ---------  ----------  ---------  ----------
  #   0.2743     0.1044    3.9689    4.5065  0.9023       0.2273      9.5732      2.2895  (132, 100)      1838524           0       2864           0             
  ## valid(dev-clean + dev-other)
  ##  ref pur(phone purity)    hyp pur(cluster purity)    H(ref)    H(hyp)      MI    MI/H(ref)(PNMI)    ref segL    hyp segL  p_xy shape      frm tot    frm diff    utt tot    utt miss
  ##-----------------------  -------------------------  --------  --------  ------  -----------------  ----------  ----------  ------------  ---------  ----------  ---------  ----------
  ##               0.2715                     0.1045    4.0022    4.5319  0.9561             0.2389      9.4792      2.1977  (135, 100)      3773309           0       5567           0
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
   #align_phn_dir=dataset/librispeech/librispeech_alignments/dev-other_aliphn/
   align_phn_dir=/workspace2/maduo/recipe/kaldi_egs/librispeech/exp/tri5b/decode_tgsmall_dev_other  ## frame level force align phone sequence
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   lab_dir=/workspace2/maduo/dataset/format/librispeech/feat_dir/9layer_feat_from_offical_hubert_base_model/label_dir/
   #lab_dir=/workspace2/maduo/dataset/format/librispeech/feat_dir/huber_6layer_feat_of_0.1_960h_librispeech_250k_update_from_mfcc_offical_setting/label_dir
   #lab_dir=/workspace2/maduo/dataset/format/librispeech/feat_dir/huber_9layer_feat_of_0.1_960h_librispeech_250k_update_from_mfcc_offical_setting/label_dir/
   #lab_dir=/workspace2/maduo/dataset/format/librispeech/mfcc/mfcc_lab/
   lab_suffix="km"
   devset_name="dev-other"
   #lab_sets=["dev-other"]
   #phn_sets=["dev-other"]
   python /workspace2/maduo/fairseq_speechtext/examples/hubert/measure_teacher_quality_md.py \
          $tsv_dir\
          $lab_dir\
          $lab_suffix\
          --phn_dir $align_phn_dir\
          --upsample 2\
          --lab_devsets $devset_name\
          --phn_devsets $devset_name
   ## HuBERT iter2-9th layer feat, kmeans cluster is 500, as label
   ## how to align frame level phone , its steps are as follows:
   ## first, I use train-clean-100 and train-clean-360 to train gmm-hmm triphoneme system
   ## then, dev-other is decoded by the above gmm-hmm triphoneme system,
   ## then lattice is align to frame-level(via source_md/get_frame_level_ali_phone.sh), and get frame-level phone force alignment as align phone here

   # ref pur    hyp pur    H(ref)    H(hyp)      MI    MI/H(ref)    ref segL    hyp segL  p_xy shape      frm tot    frm diff    utt tot    utt miss
   #---------  ---------  --------  --------  ------  -----------  ----------  ----------  ------------  ---------  ----------  ---------  ----------
   #  0.5403     0.1198    3.9689    6.0745  2.2492       0.5667      9.5732      3.4024  (132, 500)      1838524        1436       2864           0
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
   #align_phn_dir=dataset/librispeech/librispeech_alignments/dev-other_aliphn/
   align_phn_dir=/workspace2/maduo/recipe/kaldi_egs/librispeech/exp/tri5b/decode_tgsmall_dev_other  ## frame level force align phone sequence
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   #lab_dir=/workspace2/maduo/dataset/format/librispeech/feat_dir/9layer_feat_from_offical_hubert_base_model/label_dir/
   lab_dir=/workspace2/maduo/dataset/format/librispeech/feat_dir/huber_6layer_feat_of_0.1_960h_librispeech_250k_update_from_mfcc_offical_setting/label_dir
   #lab_dir=/workspace2/maduo/dataset/format/librispeech/feat_dir/huber_9layer_feat_of_0.1_960h_librispeech_250k_update_from_mfcc_offical_setting/label_dir/
   #lab_dir=/workspace2/maduo/dataset/format/librispeech/mfcc/mfcc_lab/
   lab_suffix="km"
   devset_name="dev-other"
   #lab_sets=["dev-other"]
   #phn_sets=["dev-other"]
   python /workspace2/maduo/fairseq_speechtext/examples/hubert/measure_teacher_quality_md.py \
          $tsv_dir\
          $lab_dir\
          $lab_suffix\
          --phn_dir $align_phn_dir\
          --upsample 2\
          --lab_devsets $devset_name\
          --phn_devsets $devset_name
 ## HuBERT iter1-6th layer feat, kmeans cluster is 500, as label 
 ## how to align frame level phone, its steps are as follows:
 ## first, I use train-clean-100 and train-clean-360 to train gmm-hmm triphoneme system
 ## then, dev-other is decoded by the above gmm-hmm triphoneme system, 
 ## then lattice is align to frame-level(via source_md/get_frame_level_ali_phone.sh), and get frame-level phone force alignment as align phone here

 #  ref pur    hyp pur    H(ref)    H(hyp)      MI    MI/H(ref)    ref segL    hyp segL  p_xy shape      frm tot    frm diff    utt tot    utt miss
 # ---------  ---------  --------  --------  ------  -----------  ----------  ----------  ------------  ---------  ----------  ---------  ----------
 #  0.5270     0.1425    3.9689    5.9996  2.1983       0.5539      9.5732      3.1755  (132, 500)      1838524        1436       2864           0   
fi

###
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
   #align_phn_dir=dataset/librispeech/librispeech_alignments/dev-other_aliphn/
   align_phn_dir=/workspace2/maduo/recipe/kaldi_egs/librispeech/exp/tri5b/decode_tgsmall_dev_other  ## frame level force align phone sequence
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   lab_dir=/workspace2/maduo/dataset/format/librispeech/librispeech_frame_monophncode_using_wav2vec-u2_model_newer/
   lab_suffix="unsupphncode"
   devset_name="dev-other"
   python /workspace2/maduo/fairseq_speechtext/examples/hubert/measure_teacher_quality_md.py \
          $tsv_dir\
          $lab_dir\
          $lab_suffix\
          --phn_dir $align_phn_dir\
          --upsample 2\
          --lab_devsets $devset_name\
          --phn_devsets $devset_name
 #  ref pur    hyp pur    H(ref)    H(hyp)      MI    MI/H(ref)    ref segL    hyp segL  p_xy shape      frm tot    frm diff    utt tot    utt miss
 # ---------  ---------  --------  --------  ------  -----------  ----------  ----------  ------------  ---------  ----------  ---------  ----------
 #     0.4507     0.6394    3.9689    2.9621  1.5413       0.3883      9.5732      8.4001  (132, 36)       1838524        1436       2864           0
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
   #align_phn_dir=dataset/librispeech/librispeech_alignments/dev-other_aliphn/
   align_phn_dir=/workspace2/maduo/recipe/kaldi_egs/librispeech/exp/tri5b/decode_tgsmall_dev_other  ## frame level force align phone sequence
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   #lab_dir=/workspace2/maduo/dataset/format/librispeech/feat_dir/9layer_feat_from_offical_hubert_base_model/label_dir/
   lab_dir=/workspace2/maduo/dataset/format/librispeech/feat_dir/voicelm_feat_of_0.1_960h_librispeech_400k_update/voicelm_7layer_label_dir
   #lab_dir=/workspace2/maduo/dataset/format/librispeech/feat_dir/huber_9layer_feat_of_0.1_960h_librispeech_250k_update_from_mfcc_offical_setting/label_dir/
   #lab_dir=/workspace2/maduo/dataset/format/librispeech/mfcc/mfcc_lab/
   lab_suffix="km"
   devset_name="dev-other"
   #lab_sets=["dev-other"]
   #phn_sets=["dev-other"]
   python /workspace2/maduo/fairseq_speechtext/examples/hubert/measure_teacher_quality_md.py \
          $tsv_dir\
          $lab_dir\
          $lab_suffix\
          --phn_dir $align_phn_dir\
          --upsample 2\
          --lab_devsets $devset_name\
          --phn_devsets $devset_name
 ## Voicelm iter1-7th layer feat, kmeans cluster is 500, as label
 ## how to align frame level phone, its steps are as follows:
 ## first, I use train-clean-100 and train-clean-360 to train gmm-hmm triphoneme system
 ## then, dev-other is decoded by the above gmm-hmm triphoneme system,
 ## then lattice is align to frame-level(via source_md/get_frame_level_ali_phone.sh), and get frame-level phone force alignment as align phone here

 ##  ref pur    hyp pur    H(ref)    H(hyp)      MI    MI/H(ref)    ref segL    hyp segL  p_xy shape      frm tot    frm diff    utt tot    utt miss
 ##---------  ---------  --------  --------  ------  -----------  ----------  ----------  ------------  ---------  ----------  ---------  ----------
 ##  0.5398     0.1151    3.9689    6.1159  2.2749       0.5732      9.5732      3.2971  (132, 500)      1838524        1436       2864           0
fi



if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then
   #align_phn_dir=dataset/librispeech/librispeech_alignments/dev-other_aliphn/
   align_phn_dir=/workspace2/maduo/recipe/kaldi_egs/librispeech/exp/tri5b/decode_tgsmall_dev_other  ## frame level force align phone sequence
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   #lab_dir=/workspace2/maduo/dataset/format/librispeech/feat_dir/9layer_feat_from_offical_hubert_base_model/label_dir/
   lab_dir=/workspace2/maduo/dataset/format/librispeech/feat_dir/voicelm_feat_of_0.1_960h_librispeech_400k_update/voicelm_8layer_label_dir
   #lab_dir=/workspace2/maduo/dataset/format/librispeech/feat_dir/huber_9layer_feat_of_0.1_960h_librispeech_250k_update_from_mfcc_offical_setting/label_dir/
   #lab_dir=/workspace2/maduo/dataset/format/librispeech/mfcc/mfcc_lab/
   lab_suffix="km"
   devset_name="dev-other"
   #lab_sets=["dev-other"]
   #phn_sets=["dev-other"]
   python /workspace2/maduo/fairseq_speechtext/examples/hubert/measure_teacher_quality_md.py \
          $tsv_dir\
          $lab_dir\
          $lab_suffix\
          --phn_dir $align_phn_dir\
          --upsample 2\
          --lab_devsets $devset_name\
          --phn_devsets $devset_name
 ## Voicelm iter1-8th layer feat, kmeans cluster is 500, as label
 ## how to align frame level phone, its steps are as follows:
 ## first, I use train-clean-100 and train-clean-360 to train gmm-hmm triphoneme system
 ## then, dev-other is decoded by the above gmm-hmm triphoneme system,
 ## then lattice is align to frame-level(via source_md/get_frame_level_ali_phone.sh), and get frame-level phone force alignment as align phone here


 #   ref pur    hyp pur    H(ref)    H(hyp)      MI    MI/H(ref)    ref segL    hyp segL  p_xy shape      frm tot    frm diff    utt tot    utt miss
 # ---------  ---------  --------  --------  ------  -----------  ----------  ----------  ------------  ---------  ----------  ---------  ----------
 #  0.5407     0.1163    3.9689    6.1326  2.2766       0.5736      9.5732      3.2987  (132, 500)      1838524        1436       2864           0
fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ];then
   #align_phn_dir=dataset/librispeech/librispeech_alignments/dev-other_aliphn/
   align_phn_dir=/workspace2/maduo/recipe/kaldi_egs/librispeech/exp/tri5b/decode_tgsmall_dev_other  ## frame level force align phone sequence
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   #lab_dir=/workspace2/maduo/dataset/format/librispeech/feat_dir/9layer_feat_from_offical_hubert_base_model/label_dir/
   lab_dir=/workspace2/maduo/dataset/format/librispeech/feat_dir/voicelm_feat_of_0.1_960h_librispeech_400k_update/voicelm_9layer_label_dir
   #lab_dir=/workspace2/maduo/dataset/format/librispeech/feat_dir/huber_9layer_feat_of_0.1_960h_librispeech_250k_update_from_mfcc_offical_setting/label_dir/
   #lab_dir=/workspace2/maduo/dataset/format/librispeech/mfcc/mfcc_lab/
   lab_suffix="km"
   devset_name="dev-other"
   #lab_sets=["dev-other"]
   #phn_sets=["dev-other"]
   python /workspace2/maduo/fairseq_speechtext/examples/hubert/measure_teacher_quality_md.py \
          $tsv_dir\
          $lab_dir\
          $lab_suffix\
          --phn_dir $align_phn_dir\
          --upsample 2\
          --lab_devsets $devset_name\
          --phn_devsets $devset_name
 ## Voicelm iter1-9th layer feat, kmeans cluster is 500, as label
 ## how to align frame level phone, its steps are as follows:
 ## first, I use train-clean-100 and train-clean-360 to train gmm-hmm triphoneme system
 ## then, dev-other is decoded by the above gmm-hmm triphoneme system,
 ## then lattice is align to frame-level(via source_md/get_frame_level_ali_phone.sh), and get frame-level phone force alignment as align phone here



 #  ref pur    hyp pur    H(ref)    H(hyp)      MI    MI/H(ref)    ref segL    hyp segL  p_xy shape      frm tot    frm diff    utt tot    utt miss
 #---------  ---------  --------  --------  ------  -----------  ----------  ----------  ------------  ---------  ----------  ---------  ----------
 #  0.5393     0.1157    3.9689    6.1218  2.2706       0.5721      9.5732      3.2852  (132, 500)      1838524        1436       2864           0
fi


if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ];then
   #align_phn_dir=dataset/librispeech/librispeech_alignments/dev-other_aliphn/
   align_phn_dir=/workspace2/maduo/recipe/kaldi_egs/librispeech/exp/tri5b/decode_tgsmall_dev_other  ## frame level force align phone sequence
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   #lab_dir=/workspace2/maduo/dataset/format/librispeech/feat_dir/9layer_feat_from_offical_hubert_base_model/label_dir/
   lab_dir=/workspace2/maduo/dataset/format/librispeech/feat_dir/voicelm_feat_of_0.1_960h_librispeech_400k_update/voicelm_10layer_label_dir
   #lab_dir=/workspace2/maduo/dataset/format/librispeech/feat_dir/huber_9layer_feat_of_0.1_960h_librispeech_250k_update_from_mfcc_offical_setting/label_dir/
   #lab_dir=/workspace2/maduo/dataset/format/librispeech/mfcc/mfcc_lab/
   lab_suffix="km"
   devset_name="dev-other"
   #lab_sets=["dev-other"]
   #phn_sets=["dev-other"]
   python /workspace2/maduo/fairseq_speechtext/examples/hubert/measure_teacher_quality_md.py \
          $tsv_dir\
          $lab_dir\
          $lab_suffix\
          --phn_dir $align_phn_dir\
          --upsample 2\
          --lab_devsets $devset_name\
          --phn_devsets $devset_name
 ## Voicelm iter1-10th layer feat, kmeans cluster is 500, as label
 ## how to align frame level phone, its steps are as follows:
 ## first, I use train-clean-100 and train-clean-360 to train gmm-hmm triphoneme system
 ## then, dev-other is decoded by the above gmm-hmm triphoneme system,
 ## then lattice is align to frame-level(via source_md/get_frame_level_ali_phone.sh), and get frame-level phone force alignment as align phone here

#  ref pur    hyp pur    H(ref)    H(hyp)      MI    MI/H(ref)    ref segL    hyp segL  p_xy shape      frm tot    frm diff    utt tot    utt miss
#---------  ---------  --------  --------  ------  -----------  ----------  ----------  ------------  ---------  ----------  ---------  ----------
#   0.5381     0.1154    3.9689    6.1234  2.2714       0.5723      9.5732      3.2976  (132, 500)      1838524        1436       2864           0

fi



## check which layer feature as second iter label for voicelm2
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ];then
   #align_phn_dir=dataset/librispeech/librispeech_alignments/dev-other_aliphn/
   align_phn_dir=/workspace2/maduo/recipe/kaldi_egs/librispeech/exp/tri5b/decode_tgsmall_dev_other  ## frame level force align phone sequence
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   #lab_dir=/workspace2/maduo/dataset/format/librispeech/feat_dir/9layer_feat_from_offical_hubert_base_model/label_dir/
   lab_dir=/workspace2/maduo/dataset/format/librispeech/feat_dir/voicelm2_feat_of_0.1_960h_librispeech/voicelm2_7layer_100_clusters_label_dir_for_phmi/
   #lab_dir=/workspace2/maduo/dataset/format/librispeech/feat_dir/huber_9layer_feat_of_0.1_960h_librispeech_250k_update_from_mfcc_offical_setting/label_dir/
   #lab_dir=/workspace2/maduo/dataset/format/librispeech/mfcc/mfcc_lab/
   lab_suffix="km"
   devset_phnname="dev_other"
   devset_labname="dev-other"
   #lab_sets=["dev-other"]
   #phn_sets=["dev-other"]
   python /workspace2/maduo/fairseq_speechtext/examples/hubert/measure_teacher_quality_md.py \
          $tsv_dir\
          $lab_dir\
          $lab_suffix\
          --phn_dir $align_phn_dir\
          --upsample 2\
          --lab_devsets $devset_labname\
          --phn_devsets $devset_phnname
 ## Voicelm2 iter1-7th layer feat, kmeans cluster is 100, as label
 ## how to align frame level phone, its steps are as follows:
 ## first, I use train-clean-100 and train-clean-360 to train gmm-hmm triphoneme system
 ## then, dev-other is decoded by the above gmm-hmm triphoneme system,
 ## then lattice is align to frame-level(via source_md/get_frame_level_ali_phone.sh), and get frame-level phone force alignment as align phone here
 ##   ref pur(phone purity)    hyp pur(cluster purity)    H(ref)    H(hyp)      MI    MI/H(ref)(PNMI)    ref segL    hyp segL  p_xy shape      frm tot    frm diff    utt tot    utt miss
 ## -----------------------  -------------------------  --------  --------  ------  -----------------  ----------  ----------  ------------  ---------  ----------  ---------  ----------
 ##                0.4928                     0.4229    3.9689    4.1959  1.8535             0.4670      9.5732      4.8785  (132, 100)      1838524        1436       2864           0
 
fi


## check which layer feature as second iter label for voicelm2
if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ];then
   #align_phn_dir=dataset/librispeech/librispeech_alignments/dev-other_aliphn/
   align_phn_dir=/workspace2/maduo/recipe/kaldi_egs/librispeech/exp/tri5b/decode_tgsmall_dev_other  ## frame level force align phone sequence
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   #lab_dir=/workspace2/maduo/dataset/format/librispeech/feat_dir/9layer_feat_from_offical_hubert_base_model/label_dir/
   lab_dir=/workspace2/maduo/dataset/format/librispeech/feat_dir/voicelm2_feat_of_0.1_960h_librispeech/voicelm2_7layer_500_clusters_label_dir_for_phmi/
   #lab_dir=/workspace2/maduo/dataset/format/librispeech/feat_dir/huber_9layer_feat_of_0.1_960h_librispeech_250k_update_from_mfcc_offical_setting/label_dir/
   #lab_dir=/workspace2/maduo/dataset/format/librispeech/mfcc/mfcc_lab/
   lab_suffix="km"
   devset_phnname="dev_other"
   devset_labname="dev-other"
   #lab_sets=["dev-other"]
   #phn_sets=["dev-other"]
   python /workspace2/maduo/fairseq_speechtext/examples/hubert/measure_teacher_quality_md.py \
          $tsv_dir\
          $lab_dir\
          $lab_suffix\
          --phn_dir $align_phn_dir\
          --upsample 2\
          --lab_devsets $devset_labname\
          --phn_devsets $devset_phnname
 ## Voicelm2 iter1-7th layer feat, kmeans cluster is 500, as label
 ## how to align frame level phone, its steps are as follows:
 ## first, I use train-clean-100 and train-clean-360 to train gmm-hmm triphoneme system
 ## then, dev-other is decoded by the above gmm-hmm triphoneme system,
 ## then lattice is align to frame-level(via source_md/get_frame_level_ali_phone.sh), and get frame-level phone force alignment as align phone here
 ##  ref pur(phone purity)    hyp pur(cluster purity)    H(ref)    H(hyp)      MI    MI/H(ref)(PNMI)    ref segL    hyp segL  p_xy shape      frm tot    frm diff    utt tot    utt miss
 ##-----------------------  -------------------------  --------  --------  ------  -----------------  ----------  ----------  ------------  ---------  ----------  ---------  ----------
 ##                0.5511                     0.1805    3.9689    5.8556  2.1782             0.5488      9.5732      3.0921  (132, 498)      1838524        1436       2864           0
fi


## check which layer feature as second iter label for voicelm2
if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ];then
   #align_phn_dir=dataset/librispeech/librispeech_alignments/dev-other_aliphn/
   align_phn_dir=/workspace2/maduo/recipe/kaldi_egs/librispeech/exp/tri5b/decode_tgsmall_dev_other  ## frame level force align phone sequence
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   #lab_dir=/workspace2/maduo/dataset/format/librispeech/feat_dir/9layer_feat_from_offical_hubert_base_model/label_dir/
   lab_dir=/workspace2/maduo/dataset/format/librispeech/feat_dir/voicelm2_feat_of_0.1_960h_librispeech/voicelm2_12layer_100_clusters_label_dir_for_phmi/
   #lab_dir=/workspace2/maduo/dataset/format/librispeech/feat_dir/huber_9layer_feat_of_0.1_960h_librispeech_250k_update_from_mfcc_offical_setting/label_dir/
   #lab_dir=/workspace2/maduo/dataset/format/librispeech/mfcc/mfcc_lab/
   lab_suffix="km"
   devset_phnname="dev_other"
   devset_labname="dev-other"
   #lab_sets=["dev-other"]
   #phn_sets=["dev-other"]
   python /workspace2/maduo/fairseq_speechtext/examples/hubert/measure_teacher_quality_md.py \
          $tsv_dir\
          $lab_dir\
          $lab_suffix\
          --phn_dir $align_phn_dir\
          --upsample 2\
          --lab_devsets $devset_labname\
          --phn_devsets $devset_phnname
 ## Voicelm2 iter1-12th layer feat, kmeans cluster is 100, as label
 ## how to align frame level phone, its steps are as follows:
 ## first, I use train-clean-100 and train-clean-360 to train gmm-hmm triphoneme system
 ## then, dev-other is decoded by the above gmm-hmm triphoneme system,
 ## then lattice is align to frame-level(via source_md/get_frame_level_ali_phone.sh), and get frame-level phone force alignment as align phone here
 ##  ref pur(phone purity)    hyp pur(cluster purity)    H(ref)    H(hyp)      MI    MI/H(ref)(PNMI)    ref segL    hyp segL  p_xy shape      frm tot    frm diff    utt tot    utt miss
 ## -----------------------  -------------------------  --------  --------  ------  -----------------  ----------  ----------  ------------  ---------  ----------  ---------  ----------
 ##                0.4858                     0.4077    3.9689    4.2389  1.8527             0.4668      9.5732      4.7341  (132, 100)      1838524        1436       2864           0
fi

## check which layer feature as second iter label for voicelm2
if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ];then
   #align_phn_dir=dataset/librispeech/librispeech_alignments/dev-other_aliphn/
   align_phn_dir=/workspace2/maduo/recipe/kaldi_egs/librispeech/exp/tri5b/decode_tgsmall_dev_other  ## frame level force align phone sequence
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   #lab_dir=/workspace2/maduo/dataset/format/librispeech/feat_dir/9layer_feat_from_offical_hubert_base_model/label_dir/
   lab_dir=/workspace2/maduo/dataset/format/librispeech/feat_dir/voicelm2_feat_of_0.1_960h_librispeech/voicelm2_12layer_500_clusters_label_dir_for_phmi/
   #lab_dir=/workspace2/maduo/dataset/format/librispeech/feat_dir/huber_9layer_feat_of_0.1_960h_librispeech_250k_update_from_mfcc_offical_setting/label_dir/
   #lab_dir=/workspace2/maduo/dataset/format/librispeech/mfcc/mfcc_lab/
   lab_suffix="km"
   devset_phnname="dev_other"
   devset_labname="dev-other"
   #lab_sets=["dev-other"]
   #phn_sets=["dev-other"]
   python /workspace2/maduo/fairseq_speechtext/examples/hubert/measure_teacher_quality_md.py \
          $tsv_dir\
          $lab_dir\
          $lab_suffix\
          --phn_dir $align_phn_dir\
          --upsample 2\
          --lab_devsets $devset_labname\
          --phn_devsets $devset_phnname
 ## Voicelm2 iter1-7th layer feat, kmeans cluster is 500, as label
 ## how to align frame level phone, its steps are as follows:
 ## first, I use train-clean-100 and train-clean-360 to train gmm-hmm triphoneme system
 ## then, dev-other is decoded by the above gmm-hmm triphoneme system,
 ## then lattice is align to frame-level(via source_md/get_frame_level_ali_phone.sh), and get frame-level phone force alignment as align phone here
 ##   ref pur(phone purity)    hyp pur(cluster purity)    H(ref)    H(hyp)      MI    MI/H(ref)(PNMI)    ref segL    hyp segL  p_xy shape      frm tot    frm diff    utt tot    utt miss
 ##-----------------------  -------------------------  --------  --------  ------  -----------------  ----------  ----------  ------------  ---------  ----------  ---------  ----------
 ##               0.5577                     0.1802    3.9689    5.8611  2.1830             0.5500      9.5732      3.0960  (132, 500)      1838524        1436       2864           0

fi

## check which layer feature as second iter label for voicelm2
if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ];then
   #align_phn_dir=dataset/librispeech/librispeech_alignments/dev-other_aliphn/
   align_phn_dir=/workspace2/maduo/recipe/kaldi_egs/librispeech/exp/tri5b/decode_tgsmall_dev_other  ## frame level force align phone sequence
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   #lab_dir=/workspace2/maduo/dataset/format/librispeech/feat_dir/9layer_feat_from_offical_hubert_base_model/label_dir/
   lab_dir=/workspace2/maduo/dataset/format/librispeech/feat_dir/voicelm2_feat_of_0.1_960h_librispeech/voicelm2_9layer_500_clusters_label_dir_for_phmi/
   #lab_dir=/workspace2/maduo/dataset/format/librispeech/feat_dir/huber_9layer_feat_of_0.1_960h_librispeech_250k_update_from_mfcc_offical_setting/label_dir/
   #lab_dir=/workspace2/maduo/dataset/format/librispeech/mfcc/mfcc_lab/
   lab_suffix="km"
   devset_phnname="dev_other"
   devset_labname="dev-other"
   #lab_sets=["dev-other"]
   #phn_sets=["dev-other"]
   python /workspace2/maduo/fairseq_speechtext/examples/hubert/measure_teacher_quality_md.py \
          $tsv_dir\
          $lab_dir\
          $lab_suffix\
          --phn_dir $align_phn_dir\
          --upsample 2\
          --lab_devsets $devset_labname\
          --phn_devsets $devset_phnname
 ## Voicelm2 iter1-7th layer feat, kmeans cluster is 500, as label
 ## how to align frame level phone, its steps are as follows:
 ## first, I use train-clean-100 and train-clean-360 to train gmm-hmm triphoneme system
 ## then, dev-other is decoded by the above gmm-hmm triphoneme system,
 ## then lattice is align to frame-level(via source_md/get_frame_level_ali_phone.sh), and get frame-level phone force alignment as align phone here
 #   ref pur(phone purity)    hyp pur(cluster purity)    H(ref)    H(hyp)      MI    MI/H(ref)(PNMI)    ref segL    hyp segL  p_xy shape      frm tot    frm diff    utt tot    utt miss
 ##-----------------------  -------------------------  --------  --------  ------  -----------------  ----------  ----------  ------------  ---------  ----------  ---------  ----------
 #                 0.5539                     0.1794    3.9689    5.8546  2.1797             0.5492      9.5732      3.1017  (132, 498)      1838524        1436       2864           0

fi

## check which layer feature as second iter label for voicelm2
if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ];then
   #align_phn_dir=dataset/librispeech/librispeech_alignments/dev-other_aliphn/
   align_phn_dir=/workspace2/maduo/recipe/kaldi_egs/librispeech/exp/tri5b/decode_tgsmall_dev_other  ## frame level force align phone sequence
   tsv_dir=/workspace2/maduo/dataset/format/librispeech
   #lab_dir=/workspace2/maduo/dataset/format/librispeech/feat_dir/9layer_feat_from_offical_hubert_base_model/label_dir/
   lab_dir=/workspace2/maduo/dataset/format/librispeech/feat_dir/voicelm2_feat_of_0.1_960h_librispeech/voicelm2_9layer_100_clusters_label_dir_for_phmi/
   #lab_dir=/workspace2/maduo/dataset/format/librispeech/feat_dir/huber_9layer_feat_of_0.1_960h_librispeech_250k_update_from_mfcc_offical_setting/label_dir/
   #lab_dir=/workspace2/maduo/dataset/format/librispeech/mfcc/mfcc_lab/
   lab_suffix="km"
   devset_phnname="dev_other"
   devset_labname="dev-other"
   #lab_sets=["dev-other"]
   #phn_sets=["dev-other"]
   python /workspace2/maduo/fairseq_speechtext/examples/hubert/measure_teacher_quality_md.py \
          $tsv_dir\
          $lab_dir\
          $lab_suffix\
          --phn_dir $align_phn_dir\
          --upsample 2\
          --lab_devsets $devset_labname\
          --phn_devsets $devset_phnname
 ## Voicelm2 iter1-7th layer feat, kmeans cluster is 100, as label
 ## how to align frame level phone, its steps are as follows:
 ## first, I use train-clean-100 and train-clean-360 to train gmm-hmm triphoneme system
 ## then, dev-other is decoded by the above gmm-hmm triphoneme system,
 ## then lattice is align to frame-level(via source_md/get_frame_level_ali_phone.sh), and get frame-level phone force alignment as align phone here
 #  ref pur(phone purity)    hyp pur(cluster purity)    H(ref)    H(hyp)      MI    MI/H(ref)(PNMI)    ref segL    hyp segL  p_xy shape      frm tot    frm diff    utt tot    utt miss
 ## -----------------------  -------------------------  --------  --------  ------  -----------------  ----------  ----------  ------------  ---------  ----------  ---------  ----------
 #                0.4866                     0.4055    3.9689    4.2312  1.8629             0.4694      9.5732      4.7427  (132, 100)      1838524        1436       2864           0

fi
