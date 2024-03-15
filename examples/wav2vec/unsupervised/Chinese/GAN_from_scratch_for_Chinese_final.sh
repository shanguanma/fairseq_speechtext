#!/bin/bash



stage=0
stop_stage=1000
. utils/parse_options.sh
. path_for_fairseq_speechtext.sh # python=3.9 pytorch=2.0.1
#. path_for_fsq_speechtext.sh # python=3.11, pytorch=2.1.1, but low version of hydra is not working.
#. path_for_fsq.sh # python=3.9 pytorch=2.1.2
#. activate_cuda11.6.sh
#. path_for_fsq1131.sh # python=3.8 pytorch=1.13.1 cuda11.6 it doesn't install lighning timm , because I don't know how to specify pytorch version when install them(i.e.lighning, timm)

export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_USE_CUDA_DSA=1
#pip uninstall torch -y
#pip3 --timeout=1000 install torch==2.1.2  torchaudio --index-url https://download.pytorch.org/whl/cu118
#pip install torch==1.13.1+cu116  torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
## it must be compile in compute node
#cd codebase/fairseq_speechtext/
#TORCH_USE_CUDA_DSA=1 pip install  --force-reinstall --no-cache-dir -e ./
#pip install flash-attn --no-build-isolation --force-reinstall
echo "collect_env: ~~~~~~"
python -m torch.utils.collect_env


## util now (2024-2-18), it is best setting , and get dev_uer:80.0933, best checkpoint: checkpoint_11_100000.pt
##  i have remove tone from phone sequence, final the number of phone set is 59 include <SIL>
if [ ${stage} -le 33 ] && [ ${stop_stage} -ge 33 ];then
  fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
  des_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence

  ## it stores hubert large model 15 layer representation of raw librispeech speech,
  ## it  removes  silence. it offers feature of speech.
  TASK_DATA=$des_dir/hubert_large_feat_dir_no_silence
   ## offer mfcc pesudo label from without silence unpair audio
   cp -r /mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir/mfcc_no_silence/mfcc_lab//* $TASK_DATA
   cp -r $TASK_DATA/remove_phone_tone/* $TASK_DATA
   head $TASK_DATA/*.phn

  # Unpaired text input
  #text_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_again/
  text_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone
  cp -r $text_dir/unpair_text/dict.phn.txt  $TASK_DATA
  TEXT_DATA=$text_dir/unpair_text
  ## in order to match unpair speech train set name
  #mv $TEXT_DATA/train.bin  $TEXT_DATA/train_m.bin
  #mv $TEXT_DATA/train.idx $TEXT_DATA/train_m.idx
  #KENLM_PATH=$text_dir/lm.phones.filtered.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)
  KENLM_PATH=$text_dir/lm.phones.filtered.04.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)
  config_dir=$fairseq_dir/examples/wav2vec/unsupervised

  dir=/mntcephfs/lab_data/maduo/exp/
  model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_1.75_mmi_weight0.5_code_penalty1_gradient_pennalty1.5_seed2_300k_steps
  exp_dir=$dir/wav2vec-u2_gan_from_scratch/${model_name}
  mkdir -p $exp_dir
  export PYTHONPATH=$fairseq_dir:$PYTHONPATH
  world_size=1
  update_freq=1
  python $fairseq_dir/fairseq_cli/hydra_train.py \
       --config-dir $config_dir/config/gan \
       --config-name w2vu2_local2_md \
       task.data=${TASK_DATA} \
       task.text_data=${TEXT_DATA} \
       task.kenlm_path=${KENLM_PATH} \
       dataset.train_subset=train_m\
       dataset.valid_subset=dev\
       dataset.batch_size=160\
        dataset.num_workers=6\
       optimization.max_update=300000\
       common.user_dir=${fairseq_dir}/examples/wav2vec/unsupervised \
       model.code_penalty=1\
       model.gradient_penalty=1.5\
       model.smoothness_weight=1.75\
       model.mmi_weight=0.5\
       common.seed=2\
       distributed_training.distributed_world_size=${world_size}\
       distributed_training.distributed_port=-1\
       distributed_training.ddp_backend=legacy_ddp\
       optimization.update_freq=[${update_freq}]\
       common.tensorboard_logdir=$exp_dir\
       hydra.run.dir=$exp_dir\
       hydra.job.name=$exp_dir/w2v_unsup_gan_xp

# grep -rn 'dev_uer' logs/gan_training_for_Chinese_on_A800_paper_setting_remove_tone_stage33.log | awk -F ":" '{print $12 $13}' | awk -F "," '{print $2}' | sort
# grep -rn '80.0933' logs/gan_training_for_Chinese_on_A800_paper_setting_remove_tone_stage33.log
#2143:[2024-02-16 01:18:34,062][dev][INFO] - {"epoch": 11, "dev_loss": "0.801", "dev_ntokens": "7930.71", "dev_nsentences": "158.897", "dev_sample_size": "7930.71", "dev_lm_score_sum": "-1.2296e+06", "dev_num_pred_chars": "573033", "dev_vocab_seen_pct": "0.644068", "dev_uer": "80.0933", "dev_weighted_lm_ppl": "300.172", "dev_lm_ppl": "124.519", "dev_wps": "34710.6", "dev_wpb": "7930.7", "dev_bsz": "158.9", "dev_num_updates": "100000", "dev_best_weighted_lm_ppl": "140.882"}
# best checkpoint_11_100000.pt
fi

## 2024-2-27 I will eval GAN performance of stage 33 , because it is best model(dev_uer: 80%) util now
if [ ${stage} -le 133 ]&& [ ${stop_stage} -ge 133 ];then
   echo "decode dev test_meeting test_net using wav2vec-u2.0 model and compute PER "
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   root_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence
   tsv_dir=$root_dir/tsv_dir ### it contains *.wrd and *.phn, because it doesn't matter whether the wav contains silence or not
   feat_dir=$root_dir/hubert_large_feat_dir_no_silence
   ## dict.phn.txt contain two columns ,first column is monophone(total is 41 mono phones, last monophone is <SIL>), second column is index(0-base),
   #cp -r $root_dir/librispeech_lm_norm_phn_seq/phoness/dict.phn.txt $feat_dir
   config_dir=$fairseq_dir/examples/wav2vec/
   dir=/mntcephfs/lab_data/maduo/exp
   model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_1.75_mmi_weight0.5_code_penalty1_gradient_pennalty1.5_seed2_300k_steps
   wav2vec_u2_dir=$dir/wav2vec-u2_gan_from_scratch/${model_name}
   #wav2vec_u2_dir=$dir/wav2vec-u2_gan_from_scratch/w2v_unsup_gan_xp_4gpu_8update_with_unpair_text_0.3M/
   dest_dir=$wav2vec_u2_dir/hyps_debug_for_apply_vad

   mkdir -p $dest_dir
   #testsets="dev-clean dev-other test-clean test-other train"
   testsets="dev test_meeting test_net"
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
               fairseq.common_eval.path=$wav2vec_u2_dir/checkpoint_11_100000.pt \
               fairseq.common_eval.quiet=false \
               fairseq.dataset.gen_subset=$name\
               results_path=$dest_dir\
               decode_stride=2\
               beam_threshold=100.0\
               beam=1500\
               word_score=1.0\
               sil_weight=0.0

   done
   #  grep -rn 'WER' logs/gan_infer_for_Chinese_on_A800_paper_setting_remove_tone_stage133.log
   # 55372:[2024-02-27 11:16:08,718][__main__][INFO] - WER: 84.85953053167376
   # 55374:[2024-02-27 11:16:08,719][__main__][INFO] - | Generate dev with beam=1500, lm_weight=2.0, word_score=1.0, sil_weight=0.0, blank_weight=0.0, WER: 84.85953053167376, LM_PPL: inf, num feats: 3033747, length: 691749, UER to viterbi: 0, score: 84.85953053167376
   # 85979:[2024-02-27 11:17:10,779][__main__][INFO] - WER: 86.73497299427433
   # 85981:[2024-02-27 11:17:10,779][__main__][INFO] - | Generate test_meeting with beam=1500, lm_weight=2.0, word_score=1.0, sil_weight=0.0, blank_weight=0.0, WER: 86.73497299427433, LM_PPL: inf, num feats: 1729066, length: 350793, UER to viterbi: 0, score: 86.73497299427433
   # 183178:[2024-02-27 11:19:26,689][__main__][INFO] - WER: 85.18424425655675
   # 183180:[2024-02-27 11:19:26,689][__main__][INFO] - | Generate test_net with beam=1500, lm_weight=2.0, word_score=1.0, sil_weight=0.0, blank_weight=0.0, WER: 85.18424425655675, LM_PPL: inf, num feats: 3600547, length: 859901, UER to viterbi: 0, score: 85.18424425655675
fi
if [ ${stage} -le 134 ]&& [ ${stop_stage} -ge 134 ];then
   echo "decode dev test_meeting test_net using wav2vec-u2.0 model and compute PER "
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   root_dir=/mntcephfs/lab_data/maduo/datasets
   tsv_dir=$root_dir/wenetspeech/m/no_segements_tsv_dir ### it contains *.wrd and *.phn, because it doesn't matter whether the wav contains silence or not
   feat_dir=$root_dir/wenetspeech/m/hubert_large_feat_dir_with_silence
   ## dict.phn.txt contain two columns ,first column is monophone(total is 60 mono phones, last monophone is <SIL>), second column is index(0-base),
   cp -r $root_dir/wenetspeech_no_silence/hubert_large_feat_dir_no_silence/remove_phone_tone/dict.phn.txt  $feat_dir
   config_dir=$fairseq_dir/examples/wav2vec/
   dir=/mntcephfs/lab_data/maduo/exp
   model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_1.75_mmi_weight0.5_code_penalty1_gradient_pennalty1.5_seed2_300k_steps
   wav2vec_u2_dir=$dir/wav2vec-u2_gan_from_scratch/${model_name}
   #wav2vec_u2_dir=$dir/wav2vec-u2_gan_from_scratch/w2v_unsup_gan_xp_4gpu_8update_with_unpair_text_0.3M/
   dest_dir=$wav2vec_u2_dir/hyps_debug_for_without_vad
   mkdir -p $dest_dir

   #testsets="dev test_meeting test_net"
   #testsets="dev test_meeting"
   testsets="test_net"
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   for name in $testsets;do
       cp -r $tsv_dir/${name}.wrd $feat_dir
       cp -r $tsv_dir/${name}.phn $feat_dir
       python $fairseq_dir/examples/wav2vec/unsupervised/w2vu_generate.py\
               --config-dir $config_dir/unsupervised/config/generate \
               --config-name viterbi_md \
               fairseq.common.user_dir=${fairseq_dir}/examples/wav2vec/unsupervised \
               fairseq.task.data=$feat_dir \
               fairseq.common_eval.path=$wav2vec_u2_dir/checkpoint_11_100000.pt \
               fairseq.common_eval.quiet=false \
               fairseq.dataset.gen_subset=$name\
               results_path=$dest_dir\
               decode_stride=2\
               beam_threshold=100.0\
               beam=1500\
               word_score=1.0\
               sil_weight=0.0

   done
   # grep -rn 'WER:' logs/gan_infer_for_Chinese_on_A800_paper_setting_remove_tone_stage134.log
   # 55377:[2024-02-27 15:39:08,690][__main__][INFO] - WER: 93.73398532156895
   # 55379:[2024-02-27 15:39:08,691][__main__][INFO] - | Generate dev with beam=1500, lm_weight=2.0, word_score=1.0, sil_weight=0.0, blank_weight=0.0, WER: 93.73398532156895, LM_PPL: inf, num feats: 3592801, length: 775016, UER to viterbi: 0, score: 93.73398532156895
   # 88873:[2024-02-27 15:40:19,174][__main__][INFO] - WER: 94.97108105347125
   # 88875:[2024-02-27 15:40:19,175][__main__][INFO] - | Generate test_meeting with beam=1500, lm_weight=2.0, word_score=1.0, sil_weight=0.0, blank_weight=0.0, WER: 94.97108105347125, LM_PPL: inf, num feats: 2726460, length: 493729, UER to viterbi: 0, score: 94.97108105347125

   # cat logs/gan_infer_for_Chinese_on_A800_paper_setting_remove_tone_stage134_test_net.log
   #  WER: 92.17114746596499
   # [2024-02-27 16:35:57,581][__main__][INFO] - | Processed 24774 sentences (956816 tokens) in 118.6s (208.95 sentences/s, 8070.21 tokens/s)
   # [2024-02-27 16:35:57,581][__main__][INFO] - | Generate test_net with beam=1500, lm_weight=2.0, word_score=1.0, sil_weight=0.0, blank_weight=0.0, WER: 92.17114746596499, LM_PPL: inf, num feats: 4130558, length: 956816, UER to viterbi: 0, score: 92.17114746596499
fi

if [ ${stage} -le 135 ]&& [ ${stop_stage} -ge 135 ];then
   echo "decode dev train_m using wav2vec-u2.0 model to get frame level phonemem id sequence "
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   root_dir=/mntcephfs/lab_data/maduo/datasets
   tsv_dir=$root_dir/wenetspeech/m/no_segements_tsv_dir ### it contains *.wrd and *.phn, because it doesn't matter whether the wav contains silence or not
   feat_dir=$root_dir/wenetspeech/m/hubert_large_feat_dir_with_silence
   ## dict.phn.txt contain two columns ,first column is monophone(total is 60 mono phones, last monophone is <SIL>),
   cp -r $root_dir/wenetspeech_no_silence/hubert_large_feat_dir_no_silence/remove_phone_tone/dict.phn.txt  $feat_dir
   config_dir=$fairseq_dir/examples/wav2vec/
   dir=/mntcephfs/lab_data/maduo/exp
   model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_1.75_mmi_weight0.5_code_penalty1_gradient_pennalty1.5_seed2_300k_steps
   wav2vec_u2_dir=$dir/wav2vec-u2_gan_from_scratch/${model_name}
   #wav2vec_u2_dir=/workspace2/maduo/model_hub/librispeech/wav2vec-u2.0-trained_model_using_librispeech_libirspeech_lm_text
   #dest_dir=$dir/wav2vec-u2/frame_phonecode_hyps_newer_final
   dest_dir=$wav2vec_u2_dir/frame_phonecode_hyps_newer_final
   mkdir -p $dest_dir
   testsets="dev train_m"
   #testsets="dev-clean"
   #testsets="test-other"
   export PYTHONPATH=$fairseq_dir:$PYTHONPATH
   for name in $testsets;do
       #cp -r $tsv_dir/${name}.wrd $feat_dir
       #cp -r $tsv_dir/${name}.phn $feat_dir
       python $fairseq_dir/examples/wav2vec/unsupervised/w2vu_generate_frame_phncode_final.py\
               --config-dir $config_dir/unsupervised/config/generate \
               --config-name viterbi_frame \
               fairseq.common.user_dir=${fairseq_dir}/examples/wav2vec/unsupervised \
               fairseq.task.data=$feat_dir \
               fairseq.common_eval.path=$wav2vec_u2_dir/checkpoint_11_100000.pt \
               fairseq.common_eval.quiet=false \
               fairseq.dataset.gen_subset=$name\
               results_path=$dest_dir\
               decode_stride=1
   done
fi
