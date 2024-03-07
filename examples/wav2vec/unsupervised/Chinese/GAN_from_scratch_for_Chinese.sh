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

## this script is display how to train Chinese GAN in SPL paper 
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
  fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
  des_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence

  ## it stores hubert large model 15 layer representation of raw librispeech speech,
  ## it  removes  silence. it offers feature of speech.
  TASK_DATA=$des_dir/hubert_large_feat_dir_no_silence
   ## offer mfcc pesudo label from without silence unpair audio
   cp -r /mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir/mfcc_no_silence/mfcc_lab//* $TASK_DATA 
     
   
  # Unpaired text input
  #text_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_again/
  text_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq
  cp -r $text_dir/unpair_text/dict.phn.txt  $TASK_DATA
  TEXT_DATA=$text_dir/unpair_text 
  ## in order to match unpair speech train set name
  #mv $TEXT_DATA/train.bin  $TEXT_DATA/train_m.bin
  #mv $TEXT_DATA/train.idx $TEXT_DATA/train_m.idx
  KENLM_PATH=$text_dir/lm.phones.filtered.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)
  #KENLM_PATH=$text_dir/lm.phones.filtered.04.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)
  config_dir=$fairseq_dir/examples/wav2vec/unsupervised

  dir=/mntcephfs/lab_data/maduo/exp
  model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_paper_setting
  exp_dir=$dir/wav2vec-u2_gan_from_scratch/${model_name}
  mkdir -p $exp_dir
  export PYTHONPATH=$fairseq_dir:$PYTHONPATH
  world_size=1
  update_freq=1
  #CUDA_VISIBLE_DEVICES=7  python $fairseq_dir/fairseq_cli/hydra_train.py  --multirun \
  #export TORCH_CUDNN_V8_API_DISABLED=1 
  #export TORCH_CUDNN_V8_API_DISABLED=1
  #python $fairseq_dir/fairseq_cli/hydra_train.py\
  ## parameter is from wav2vec-u2 paper, is not from https://github.com/shanguanma/fairseq_speechtext/tree/main/examples/wav2vec/unsupervised
  python $fairseq_dir/fairseq_cli/hydra_train.py  --multirun \
       --config-dir $config_dir/config/gan \
       --config-name w2vu2_local2_md \
       task.data=${TASK_DATA} \
       task.text_data=${TEXT_DATA} \
       task.kenlm_path=${KENLM_PATH} \
       dataset.train_subset=train_m\
       dataset.valid_subset=dev\
       dataset.batch_size=160\
       dataset.num_workers=6\
       common.user_dir=${fairseq_dir}/examples/wav2vec/unsupervised \
       optimization.max_update=150000\
       model.code_penalty=0,3 \
       model.gradient_penalty=1.5,2.0 \
       model.smoothness_weight=1.5,2.5\
       model.mmi_weight=0.3,0.5\
       'common.seed=range(1,3)'\
       distributed_training.distributed_world_size=${world_size}\
       distributed_training.distributed_port=-1\
       distributed_training.ddp_backend=legacy_ddp\
       optimization.update_freq=[${update_freq}]\
       common.tensorboard_logdir=$exp_dir\
       hydra.run.dir=$exp_dir\
       hydra.job.name=$exp_dir/w2v_unsup_gan_xp
fi



## this script is display how to train Chinese GAN in SPL paper
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
  fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
  des_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence

  ## it stores hubert large model 15 layer representation of raw librispeech speech,
  ## it  removes  silence. it offers feature of speech.
  TASK_DATA=$des_dir/hubert_large_feat_dir_no_silence
   ## offer mfcc pesudo label from without silence unpair audio
   cp -r /mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir/mfcc_no_silence/mfcc_lab//* $TASK_DATA


  # Unpaired text input
  #text_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_again/
  text_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq
  cp -r $text_dir/unpair_text/dict.phn.txt  $TASK_DATA
  TEXT_DATA=$text_dir/unpair_text
  ## in order to match unpair speech train set name
  #mv $TEXT_DATA/train.bin  $TEXT_DATA/train_m.bin
  #mv $TEXT_DATA/train.idx $TEXT_DATA/train_m.idx
  KENLM_PATH=$text_dir/lm.phones.filtered.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)
  #KENLM_PATH=$text_dir/lm.phones.filtered.04.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)
  config_dir=$fairseq_dir/examples/wav2vec/unsupervised

  dir=/mntcephfs/lab_data/maduo/exp
  model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_md_setting
  exp_dir=$dir/wav2vec-u2_gan_from_scratch/${model_name}
  mkdir -p $exp_dir
  export PYTHONPATH=$fairseq_dir:$PYTHONPATH
  world_size=1
  update_freq=1
  #CUDA_VISIBLE_DEVICES=7  python $fairseq_dir/fairseq_cli/hydra_train.py  --multirun \
  #export TORCH_CUDNN_V8_API_DISABLED=1
  #export TORCH_CUDNN_V8_API_DISABLED=1
  #python $fairseq_dir/fairseq_cli/hydra_train.py\
  #CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=$world_size --master_port=12345  $fairseq_dir/fairseq_cli/hydra_train.py \
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
       optimization.max_update=150000\
       common.user_dir=${fairseq_dir}/examples/wav2vec/unsupervised \
       model.code_penalty=2 \
       model.gradient_penalty=1.5 \
       model.smoothness_weight=0.5\
        distributed_training.distributed_world_size=${world_size}\
       distributed_training.distributed_port=-1\
       distributed_training.ddp_backend=legacy_ddp\
       optimization.update_freq=[${update_freq}]\
       common.tensorboard_logdir=$exp_dir\
       hydra.run.dir=$exp_dir\
       hydra.job.name=$exp_dir/w2v_unsup_gan_xp

  #  grep -rn '"dev_uer":'   /mntcephfs/lab_data/maduo/exp/wav2vec-u2_gan_from_scratch/w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_md_setting/w2v_unsup_gan_xp.log | awk -F ":" '{print $12 $13}' | awk -F "," '{print $2}' | sort
  # grep -rn '93.6615'  /mntcephfs/lab_data/maduo/exp/wav2vec-u2_gan_from_scratch/w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_md_setting/w2v_unsup_gan_xp.log
  # 81:[2024-01-31 09:14:53,262][dev][INFO] - {"epoch": 1, "dev_loss": "0.937", "dev_ntokens": "7930.71", "dev_nsentences": "158.897", "dev_sample_size": "7930.71", "dev_lm_score_sum": "-1.43771e+06", "dev_num_pred_chars": "428612", "dev_vocab_seen_pct": "0.587572", "dev_uer": "93.6615", "dev_weighted_lm_ppl": "5145.24", "dev_lm_ppl": "1776.34", "dev_wps": "20733.8", "dev_wpb": "7930.7", "dev_bsz": "158.9", "dev_num_updates": "3000", "dev_best_weighted_lm_ppl": "3314.74"} 
  # it doesn't convergence.

fi



if [ ${stage} -le 3 ]&& [ ${stop_stage} -ge 3 ];then
   echo "decode  dev-clean dev-other test-clean test-other using wav2vec-u2.0 model and compute PER "
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   root_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence
   tsv_dir=$root_dir/tsv_dir ### it contains *.wrd and *.phn, because it doesn't matter whether the wav contains silence or not
   feat_dir=$root_dir/hubert_large_feat_dir_no_silence
   ## dict.phn.txt contain two columns ,first column is monophone(total is 41 mono phones, last monophone is <SIL>), second column is index(0-base),
   #cp -r $root_dir/librispeech_lm_norm_phn_seq/phoness/dict.phn.txt $feat_dir
   config_dir=$fairseq_dir/examples/wav2vec/
   dir=/mntcephfs/lab_data/maduo/exp
   #model_name=w2v_unsup_gan_xp_4gpu_8update
   #exp_dir=$dir/wav2vec-u2_gan_from_scratch/${model_name}

   model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_md_setting
   exp_dir=$dir/wav2vec-u2_gan_from_scratch/${model_name}
   #wav2vec_u2_dir=$dir/wav2vec-u2_gan_from_scratch/w2v_unsup_gan_xp_4gpu_8update_with_unpair_text_0.3M/
   decode_checkpoint=$exp_dir/ ## (TODO) add
   dest_dir=$exp_dir/hyps_debug_for_apply_vad
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
               fairseq.common_eval.path=$decode_checkpoint \
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



## this script is display how to train Chinese GAN in SPL paper
##  i have remove tone from phone sequence, final the number of phone set is 59 include <SIL>
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ];then
  fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
  des_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence

  ## it stores hubert large model 15 layer representation of raw librispeech speech,
  ## it  removes  silence. it offers feature of speech.
  TASK_DATA=$des_dir/hubert_large_feat_dir_no_silence
   ## offer mfcc pesudo label from without silence unpair audio
   cp -r /mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir/mfcc_no_silence/mfcc_lab//* $TASK_DATA


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
  model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_md_setting_remove_tone
  exp_dir=$dir/wav2vec-u2_gan_from_scratch/${model_name}
  mkdir -p $exp_dir
  export PYTHONPATH=$fairseq_dir:$PYTHONPATH
  world_size=1
  update_freq=1
  #CUDA_VISIBLE_DEVICES=7  python $fairseq_dir/fairseq_cli/hydra_train.py  --multirun \
  #export TORCH_CUDNN_V8_API_DISABLED=1
  #export TORCH_CUDNN_V8_API_DISABLED=1
  #python $fairseq_dir/fairseq_cli/hydra_train.py\
  #CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=$world_size --master_port=12345  $fairseq_dir/fairseq_cli/hydra_train.py \
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
       optimization.max_update=400000\
       common.user_dir=${fairseq_dir}/examples/wav2vec/unsupervised \
       model.code_penalty=2 \
       model.gradient_penalty=1.5 \
       model.smoothness_weight=0.5\
        distributed_training.distributed_world_size=${world_size}\
       distributed_training.distributed_port=-1\
       distributed_training.ddp_backend=legacy_ddp\
       optimization.update_freq=[${update_freq}]\
       common.tensorboard_logdir=$exp_dir\
       hydra.run.dir=$exp_dir\
       hydra.job.name=$exp_dir/w2v_unsup_gan_xp
# grep -rn '"dev_uer":' /mntcephfs/lab_data/maduo/exp/wav2vec-u2_gan_from_scratch/w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_md_setting_remove_tone/w2v_unsup_gan_xp.log  |  awk -F ":" '{print $12 $13}' | awk -F "," '{print $2}' | sort
# grep -rn '89.3177' /mntcephfs/lab_data/maduo/exp/wav2vec-u2_gan_from_scratch/w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_md_setting_remove_tone/w2v_unsup_gan_xp.log
#6867:[2024-02-05 03:35:02,462][dev][INFO] - {"epoch": 32, "dev_loss": "0.893", "dev_ntokens": "7930.71", "dev_nsentences": "158.897", "dev_sample_size": "7930.71", "dev_lm_score_sum": "-851970", "dev_num_pred_chars": "402965", "dev_vocab_seen_pct": "0.711864", "dev_uer": "89.3177", "dev_weighted_lm_ppl": "218.441", "dev_lm_ppl": "110.695", "dev_wps": "34492.1", "dev_wpb": "7930.7", "dev_bsz": "158.9", "dev_num_updates": "293000", "dev_best_weighted_lm_ppl": "108.072"} 
# best uer: "dev_uer": "89.3177", 
# best uer checkpoint: checkpoint_32_293000.pt
fi

if [ ${stage} -le 10 ]&& [ ${stop_stage} -ge 10 ];then
   echo "decode  dev-clean dev-other test-clean test-other using wav2vec-u2.0 model and compute PER "
   fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
   root_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence
   tsv_dir=$root_dir/tsv_dir ### it contains *.wrd and *.phn, because it doesn't matter whether the wav contains silence or not
   feat_dir=$root_dir/hubert_large_feat_dir_no_silence
   ## dict.phn.txt contain two columns ,first column is monophone(total is 41 mono phones, last monophone is <SIL>), second column is index(0-base),
   #cp -r $root_dir/librispeech_lm_norm_phn_seq/phoness/dict.phn.txt $feat_dir
   config_dir=$fairseq_dir/examples/wav2vec/
   dir=/mntnfs/lee_data1/maduo/exp
   #model_name=w2v_unsup_gan_xp_4gpu_8update
   #exp_dir=$dir/wav2vec-u2_gan_from_scratch/${model_name}

   model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M
   exp_dir=$dir/wav2vec-u2_gan_from_scratch/${model_name}
   #wav2vec_u2_dir=$dir/wav2vec-u2_gan_from_scratch/w2v_unsup_gan_xp_4gpu_8update_with_unpair_text_0.3M/
   decode_checkpoint=$exp_dir/ ## (TODO) add
   dest_dir=$exp_dir/hyps_debug_for_apply_vad
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
               fairseq.common_eval.path=$decode_checkpoint \
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

## this script is display how to train Chinese GAN in SPL paper
##  i have remove tone from phone sequence, final the number of phone set is 59 include <SIL>
if [ ${stage} -le 17 ] && [ ${stop_stage} -ge 17 ];then
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
  model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_md_new_setting_remove_tone_again
  exp_dir=$dir/wav2vec-u2_gan_from_scratch/${model_name}
  mkdir -p $exp_dir
  export PYTHONPATH=$fairseq_dir:$PYTHONPATH
  world_size=1
  update_freq=1
  #CUDA_VISIBLE_DEVICES=7  python $fairseq_dir/fairseq_cli/hydra_train.py  --multirun \
  #export TORCH_CUDNN_V8_API_DISABLED=1
  #export TORCH_CUDNN_V8_API_DISABLED=1
  #python $fairseq_dir/fairseq_cli/hydra_train.py\
  #CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=$world_size --master_port=12345  $fairseq_dir/fairseq_cli/hydra_train.py \
  python $fairseq_dir/fairseq_cli/hydra_train.py   --multirun  \
       --config-dir $config_dir/config/gan \
       --config-name w2vu2_local2_md \
       task.data=${TASK_DATA} \
       task.text_data=${TEXT_DATA} \
       task.kenlm_path=${KENLM_PATH} \
       dataset.train_subset=train_m\
       dataset.valid_subset=dev\
       dataset.batch_size=160\
        dataset.num_workers=6\
       optimization.max_update=150000\
       common.user_dir=${fairseq_dir}/examples/wav2vec/unsupervised \
       model.code_penalty=2 \
       model.gradient_penalty=1.5,2.0 \
       model.smoothness_weight=1.0,0.75,1.5\
       model.mmi_weight=0.5\
       'common.seed=range(1,3)'\
        distributed_training.distributed_world_size=${world_size}\
       distributed_training.distributed_port=-1\
       distributed_training.ddp_backend=legacy_ddp\
       optimization.update_freq=[${update_freq}]\
       common.tensorboard_logdir=$exp_dir\
       hydra.run.dir=$exp_dir\
       hydra.job.name=$exp_dir/w2v_unsup_gan_xp
fi

## this script is display how to train Chinese GAN in SPL paper
##  i have remove tone from phone sequence, final the number of phone set is 59 include <SIL>
if [ ${stage} -le 18 ] && [ ${stop_stage} -ge 18 ];then
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
  model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_0.75_mmi_weight0.5
  exp_dir=$dir/wav2vec-u2_gan_from_scratch/${model_name}
  mkdir -p $exp_dir
  export PYTHONPATH=$fairseq_dir:$PYTHONPATH
  world_size=1
  update_freq=1
  #CUDA_VISIBLE_DEVICES=7  python $fairseq_dir/fairseq_cli/hydra_train.py  --multirun \
  #export TORCH_CUDNN_V8_API_DISABLED=1
  #export TORCH_CUDNN_V8_API_DISABLED=1
  #python $fairseq_dir/fairseq_cli/hydra_train.py\
  #CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=$world_size --master_port=12345  $fairseq_dir/fairseq_cli/hydra_train.py \
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
       optimization.max_update=150000\
       common.user_dir=${fairseq_dir}/examples/wav2vec/unsupervised \
       model.code_penalty=2 \
       model.gradient_penalty=1.5\
       model.smoothness_weight=0.75\
       model.mmi_weight=0.5\
       common.seed=1\
        distributed_training.distributed_world_size=${world_size}\
       distributed_training.distributed_port=-1\
       distributed_training.ddp_backend=legacy_ddp\
       optimization.update_freq=[${update_freq}]\
       common.tensorboard_logdir=$exp_dir\
       hydra.run.dir=$exp_dir\
       hydra.job.name=$exp_dir/w2v_unsup_gan_xp
fi

## this script is display how to train Chinese GAN in SPL paper
##  i have remove tone from phone sequence, final the number of phone set is 59 include <SIL>
if [ ${stage} -le 19 ] && [ ${stop_stage} -ge 19 ];then
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
  model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_0.75_mmi_weight0.5_gradient_penalty1.5_code_penalty1
  exp_dir=$dir/wav2vec-u2_gan_from_scratch/${model_name}
  mkdir -p $exp_dir
  export PYTHONPATH=$fairseq_dir:$PYTHONPATH
  world_size=1
  update_freq=1
  #CUDA_VISIBLE_DEVICES=7  python $fairseq_dir/fairseq_cli/hydra_train.py  --multirun \
  #export TORCH_CUDNN_V8_API_DISABLED=1
  #export TORCH_CUDNN_V8_API_DISABLED=1
  #python $fairseq_dir/fairseq_cli/hydra_train.py\
  #CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=$world_size --master_port=12345  $fairseq_dir/fairseq_cli/hydra_train.py \
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
       optimization.max_update=150000\
       common.user_dir=${fairseq_dir}/examples/wav2vec/unsupervised \
       model.code_penalty=1 \
       model.gradient_penalty=1.5\
       model.smoothness_weight=0.75\
       model.mmi_weight=0.5\
       common.seed=1\
        distributed_training.distributed_world_size=${world_size}\
       distributed_training.distributed_port=-1\
       distributed_training.ddp_backend=legacy_ddp\
       optimization.update_freq=[${update_freq}]\
       common.tensorboard_logdir=$exp_dir\
       hydra.run.dir=$exp_dir\
       hydra.job.name=$exp_dir/w2v_unsup_gan_xp
# grep -rn 'dev_uer' logs/gan_training_for_Chinese_on_A800_paper_setting_remove_tone_stage19.log | awk -F ":" '{print $12 $13}' | awk -F "," '{print $2}' | sortgrep -rn 'dev_uer' logs/gan_training_for_Chinese_on_A800_paper_setting_remove_tone_stage19.log | awk -F ":" '{print $12 $13}' | awk -F "," '{print $2}' | sort
# best dev_uer:88.9268
#  grep -rn '88.9268' logs/gan_training_for_Chinese_on_A800_paper_setting_remove_tone_stage19.log
# 2895:[2024-02-09 07:23:16,879][dev][INFO] - {"epoch": 15, "dev_loss": "0.889", "dev_ntokens": "7930.71", "dev_nsentences": "158.897", "dev_sample_size": "7930.71", "dev_lm_score_sum": "-1.07078e+06", "dev_num_pred_chars": "504423", "dev_vocab_seen_pct": "0.677966", "dev_uer": "88.9268", "dev_weighted_lm_ppl": "253.358", "dev_lm_ppl": "116.453", "dev_wps": "33573.5", "dev_wpb": "7930.7", "dev_bsz": "158.9", "dev_num_updates": "139000", "dev_best_weighted_lm_ppl": "111.234"} 
# best checkpoint_15_139000.pt
fi

## this script is display how to train Chinese GAN in SPL paper
##  i have remove tone from phone sequence, final the number of phone set is 59 include <SIL>
if [ ${stage} -le 20 ] && [ ${stop_stage} -ge 20 ];then
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
  model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_1.0_mmi_weight0.5
  exp_dir=$dir/wav2vec-u2_gan_from_scratch/${model_name}
  mkdir -p $exp_dir
  export PYTHONPATH=$fairseq_dir:$PYTHONPATH
  world_size=1
  update_freq=1
  #CUDA_VISIBLE_DEVICES=7  python $fairseq_dir/fairseq_cli/hydra_train.py  --multirun \
  #export TORCH_CUDNN_V8_API_DISABLED=1
  #export TORCH_CUDNN_V8_API_DISABLED=1
  #python $fairseq_dir/fairseq_cli/hydra_train.py\
  #CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=$world_size --master_port=12345  $fairseq_dir/fairseq_cli/hydra_train.py \
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
       optimization.max_update=150000\
       common.user_dir=${fairseq_dir}/examples/wav2vec/unsupervised \
       model.code_penalty=2 \
       model.gradient_penalty=1.5\
       model.smoothness_weight=1.0\
       model.mmi_weight=0.5\
       common.seed=3\
        distributed_training.distributed_world_size=${world_size}\
       distributed_training.distributed_port=-1\
       distributed_training.ddp_backend=legacy_ddp\
       optimization.update_freq=[${update_freq}]\
       common.tensorboard_logdir=$exp_dir\
       hydra.run.dir=$exp_dir\
       hydra.job.name=$exp_dir/w2v_unsup_gan_xp
fi

## this script is display how to train Chinese GAN in SPL paper
##  i have remove tone from phone sequence, final the number of phone set is 59 include <SIL>
if [ ${stage} -le 21 ] && [ ${stop_stage} -ge 21 ];then
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
  model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_1.5_mmi_weight0.3
  exp_dir=$dir/wav2vec-u2_gan_from_scratch/${model_name}
  mkdir -p $exp_dir
  export PYTHONPATH=$fairseq_dir:$PYTHONPATH
  world_size=1
  update_freq=1
  #CUDA_VISIBLE_DEVICES=7  python $fairseq_dir/fairseq_cli/hydra_train.py  --multirun \
  #export TORCH_CUDNN_V8_API_DISABLED=1
  #export TORCH_CUDNN_V8_API_DISABLED=1
  #python $fairseq_dir/fairseq_cli/hydra_train.py\
  #CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=$world_size --master_port=12345  $fairseq_dir/fairseq_cli/hydra_train.py \
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
       optimization.max_update=150000\
       common.user_dir=${fairseq_dir}/examples/wav2vec/unsupervised \
       model.code_penalty=2\
       model.gradient_penalty=1.5\
       model.smoothness_weight=1.5\
       model.mmi_weight=0.3\
       common.seed=2\
        distributed_training.distributed_world_size=${world_size}\
       distributed_training.distributed_port=-1\
       distributed_training.ddp_backend=legacy_ddp\
       optimization.update_freq=[${update_freq}]\
       common.tensorboard_logdir=$exp_dir\
       hydra.run.dir=$exp_dir\
       hydra.job.name=$exp_dir/w2v_unsup_gan_xp
fi
#util now 2024-2-9, it is best setting  for Chinese
## this script is display how to train Chinese GAN in SPL paper
##  i have remove tone from phone sequence, final the number of phone set is 59 include <SIL>
if [ ${stage} -le 22 ] && [ ${stop_stage} -ge 22 ];then
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
  model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_1.0_mmi_weight0.5_code_penalty1_gradient_pennalty1.5
  exp_dir=$dir/wav2vec-u2_gan_from_scratch/${model_name}
  mkdir -p $exp_dir
  export PYTHONPATH=$fairseq_dir:$PYTHONPATH
  world_size=1
  update_freq=1
  #CUDA_VISIBLE_DEVICES=7  python $fairseq_dir/fairseq_cli/hydra_train.py  --multirun \
  #export TORCH_CUDNN_V8_API_DISABLED=1
  #export TORCH_CUDNN_V8_API_DISABLED=1
  #python $fairseq_dir/fairseq_cli/hydra_train.py\
  #CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=$world_size --master_port=12345  $fairseq_dir/fairseq_cli/hydra_train.py \
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
       optimization.max_update=150000\
       common.user_dir=${fairseq_dir}/examples/wav2vec/unsupervised \
       model.code_penalty=1\
       model.gradient_penalty=1.5\
       model.smoothness_weight=1.0\
       model.mmi_weight=0.5\
       common.seed=2\
        distributed_training.distributed_world_size=${world_size}\
       distributed_training.distributed_port=-1\
       distributed_training.ddp_backend=legacy_ddp\
       optimization.update_freq=[${update_freq}]\
       common.tensorboard_logdir=$exp_dir\
       hydra.run.dir=$exp_dir\
       hydra.job.name=$exp_dir/w2v_unsup_gan_xp
#  grep -rn '"dev_uer":'  logs/gan_training_for_Chinese_on_A800_paper_setting_remove_tone_stage22.log | awk -F ":" '{print $12 $13}' | awk -F "," '{print $2}' | sort 
# best dev_uer: 86.2496"
# grep -rn '"86.2496"' logs/gan_training_for_Chinese_on_A800_paper_setting_remove_tone_stage22.log
#2247:[2024-02-07 23:17:26,483][dev][INFO] - {"epoch": 11, "dev_loss": "0.862", "dev_ntokens": "7930.71", "dev_nsentences": "158.897", "dev_sample_size": "7930.71", "dev_lm_score_sum": "-926206", "dev_num_pred_chars": "437897", "dev_vocab_seen_pct": "0.678161", "dev_uer": "86.2496", "dev_weighted_lm_ppl": "244.19", "dev_lm_ppl": "112.304", "dev_wps": "33831", "dev_wpb": "7930.7", "dev_bsz": "158.9", "dev_num_updates": "99000", "dev_best_weighted_lm_ppl": "127.022"} 
# best checkpoint_11_99000.pt
fi


## this script is display how to train Chinese GAN in SPL paper
##  i have remove tone from phone sequence, final the number of phone set is 59 include <SIL>
if [ ${stage} -le 23 ] && [ ${stop_stage} -ge 23 ];then
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
  model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_1.5_mmi_weight0.5_code_penalty1_gradient_pennalty1.5
  exp_dir=$dir/wav2vec-u2_gan_from_scratch/${model_name}
  mkdir -p $exp_dir
  export PYTHONPATH=$fairseq_dir:$PYTHONPATH
  world_size=1
  update_freq=1
  #CUDA_VISIBLE_DEVICES=7  python $fairseq_dir/fairseq_cli/hydra_train.py  --multirun \
  #export TORCH_CUDNN_V8_API_DISABLED=1
  #export TORCH_CUDNN_V8_API_DISABLED=1
  #python $fairseq_dir/fairseq_cli/hydra_train.py\
  #CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=$world_size --master_port=12345  $fairseq_dir/fairseq_cli/hydra_train.py \
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
       optimization.max_update=150000\
       common.user_dir=${fairseq_dir}/examples/wav2vec/unsupervised \
       model.code_penalty=1\
       model.gradient_penalty=1.5\
       model.smoothness_weight=1.5\
       model.mmi_weight=0.5\
       common.seed=2\
        distributed_training.distributed_world_size=${world_size}\
       distributed_training.distributed_port=-1\
       distributed_training.ddp_backend=legacy_ddp\
       optimization.update_freq=[${update_freq}]\
       common.tensorboard_logdir=$exp_dir\
       hydra.run.dir=$exp_dir\
       hydra.job.name=$exp_dir/w2v_unsup_gan_xp
#  grep -rn '"dev_uer":'  logs/gan_training_for_Chinese_on_A800_paper_setting_remove_tone_stage23.log | awk -F ":" '{print $12 $13}' | awk -F "," '{print $2}' | sort 
#  "dev_uer" "90.2664"
# best dev_uer: "90.2664"
# grep -rn '90.2664'  logs/gan_training_for_Chinese_on_A800_paper_setting_remove_tone_stage23.log
# 1167:[2024-02-07 21:03:20,477][dev][INFO] - {"epoch": 8, "dev_loss": "0.903", "dev_ntokens": "7930.71", "dev_nsentences": "158.897", "dev_sample_size": "7930.71", "dev_lm_score_sum": "-1.32969e+06", "dev_num_pred_chars": "644554", "dev_vocab_seen_pct": "0.661017", "dev_uer": "90.2664", "dev_weighted_lm_ppl": "239.455", "dev_lm_ppl": "104.628", "dev_wps": "33411.8", "dev_wpb": "7930.7", "dev_bsz": "158.9", "dev_num_updates": "69000", "dev_best_weighted_lm_ppl": "137.259"}
# best checkpoint_8_69000.pt
fi

## this script is display how to train Chinese GAN in SPL paper
##  i have remove tone from phone sequence, final the number of phone set is 59 include <SIL>
if [ ${stage} -le 24 ] && [ ${stop_stage} -ge 24 ];then
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
  model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_1.0_mmi_weight0.5_code_penalty1_gradient_pennalty2
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
       optimization.max_update=150000\
       common.user_dir=${fairseq_dir}/examples/wav2vec/unsupervised \
       model.code_penalty=1\
       model.gradient_penalty=2.0\
       model.smoothness_weight=1.0\
       model.mmi_weight=0.5\
       common.seed=2\
        distributed_training.distributed_world_size=${world_size}\
       distributed_training.distributed_port=-1\
       distributed_training.ddp_backend=legacy_ddp\
       optimization.update_freq=[${update_freq}]\
       common.tensorboard_logdir=$exp_dir\
       hydra.run.dir=$exp_dir\
       hydra.job.name=$exp_dir/w2v_unsup_gan_xp

# grep -rn 'dev_uer' logs/gan_training_for_Chinese_on_A800_paper_setting_remove_tone_stage24.log | awk -F ":" '{print $12 $13}' | awk -F "," '{print $2}' | sort
# best dev_uer: "88.3059"
# grep -rn '"88.3059"' logs/gan_training_for_Chinese_on_A800_paper_setting_remove_tone_stage24.log
#2595:[2024-02-09 03:35:11,187][dev][INFO] - {"epoch": 13, "dev_loss": "0.883", "dev_ntokens": "7930.71", "dev_nsentences": "158.897", "dev_sample_size": "7930.71", "dev_lm_score_sum": "-1.1876e+06", "dev_num_pred_chars": "575945", "dev_vocab_seen_pct": "0.677966", "dev_uer": "88.3059", "dev_weighted_lm_ppl": "224.52", "dev_lm_ppl": "103.198", "dev_wps": "34276.6", "dev_wpb": "7930.7", "dev_bsz": "158.9", "dev_num_updates": "113000", "dev_best_weighted_lm_ppl": "122.783"}  
# best checkpoint_13_11300.pt

fi


## this script is display how to train Chinese GAN in SPL paper
##  i have remove tone from phone sequence, final the number of phone set is 59 include <SIL>
if [ ${stage} -le 25 ] && [ ${stop_stage} -ge 25 ];then
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
  model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_1.0_mmi_weight0.5_code_penalty1_gradient_pennalty1.5_seed1
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
       optimization.max_update=150000\
       common.user_dir=${fairseq_dir}/examples/wav2vec/unsupervised \
       model.code_penalty=1\
       model.gradient_penalty=1.5\
       model.smoothness_weight=1.0\
       model.mmi_weight=0.5\
       common.seed=1\
        distributed_training.distributed_world_size=${world_size}\
       distributed_training.distributed_port=-1\
       distributed_training.ddp_backend=legacy_ddp\
       optimization.update_freq=[${update_freq}]\
       common.tensorboard_logdir=$exp_dir\
       hydra.run.dir=$exp_dir\
       hydra.job.name=$exp_dir/w2v_unsup_gan_xp
fi

## this script is display how to train Chinese GAN in SPL paper
##  i have remove tone from phone sequence, final the number of phone set is 59 include <SIL>
if [ ${stage} -le 26 ] && [ ${stop_stage} -ge 26 ];then
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
  model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_1.0_mmi_weight0.5_code_penalty1_gradient_pennalty2.5_seed2
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
       optimization.max_update=150000\
       common.user_dir=${fairseq_dir}/examples/wav2vec/unsupervised \
       model.code_penalty=1\
       model.gradient_penalty=2.5\
       model.smoothness_weight=1.0\
       model.mmi_weight=0.5\
       common.seed=2\
        distributed_training.distributed_world_size=${world_size}\
       distributed_training.distributed_port=-1\
       distributed_training.ddp_backend=legacy_ddp\
       optimization.update_freq=[${update_freq}]\
       common.tensorboard_logdir=$exp_dir\
       hydra.run.dir=$exp_dir\
       hydra.job.name=$exp_dir/w2v_unsup_gan_xp

  # grep -rn 'dev_uer' logs/gan_training_for_Chinese_on_A800_paper_setting_remove_tone_stage26.log | awk -F ":" '{print $12 $13}' | awk -F "," '{print $2}' | sort 

  # grep -rn '"88.196"' logs/gan_training_for_Chinese_on_A800_paper_setting_remove_tone_stage26.log
  #2526:[2024-02-10 04:47:45,508][dev][INFO] - {"epoch": 12, "dev_loss": "0.882", "dev_ntokens": "7930.71", "dev_nsentences": "158.897", "dev_sample_size": "7930.71", "dev_lm_score_sum": "-893033", "dev_num_pred_chars": "420413", "dev_vocab_seen_pct": "0.677966", "dev_uer": "88.196", "dev_weighted_lm_ppl": "247.823", "dev_lm_ppl": "113.909", "dev_wps": "34328", "dev_wpb": "7930.7", "dev_bsz": "158.9", "dev_num_updates": "110000", "dev_best_weighted_lm_ppl": "127.586"}
  # best "dev_uer" "88.196""
  # best checkpoint_12_110000.pt 
  
fi

## this script is display how to train Chinese GAN in SPL paper
##  i have remove tone from phone sequence, final the number of phone set is 59 include <SIL>
if [ ${stage} -le 27 ] && [ ${stop_stage} -ge 27 ];then
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
  model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_1.0_mmi_weight0.5_code_penalty1_gradient_pennalty1.0_seed2
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
       optimization.max_update=150000\
       common.user_dir=${fairseq_dir}/examples/wav2vec/unsupervised \
       model.code_penalty=1\
       model.gradient_penalty=1.0\
       model.smoothness_weight=1.0\
       model.mmi_weight=0.5\
       common.seed=2\
        distributed_training.distributed_world_size=${world_size}\
       distributed_training.distributed_port=-1\
       distributed_training.ddp_backend=legacy_ddp\
       optimization.update_freq=[${update_freq}]\
       common.tensorboard_logdir=$exp_dir\
       hydra.run.dir=$exp_dir\
       hydra.job.name=$exp_dir/w2v_unsup_gan_xp

# grep -rn '"dev_uer":' logs/gan_training_for_Chinese_on_A800_paper_setting_remove_tone_stage27_1.log | awk -F ":" '{print $12 $13}' | awk -F "," '{print $2}' | sort
# best dev_uer: 89.179
#  grep -rn '89.179' logs/gan_training_for_Chinese_on_A800_paper_setting_remove_tone_stage27_1.log 
#2338:[2024-02-14 02:09:52,965][dev][INFO] - {"epoch": 12, "dev_loss": "0.892", "dev_ntokens": "7930.71", "dev_nsentences": "158.897", "dev_sample_size": "7930.71", "dev_lm_score_sum": "-1.00038e+06", "dev_num_pred_chars": "487819", "dev_vocab_seen_pct": "0.694915", "dev_uer": "89.179", "dev_weighted_lm_ppl": "204.338", "dev_lm_ppl": "98.6764", "dev_wps": "35311.9", "dev_wpb": "7930.7", "dev_bsz": "158.9", "dev_num_updates": "109000", "dev_best_weighted_lm_ppl": "109.742"}
# best checkpoint_12_109000.pt 

fi



## this script is display how to train Chinese GAN in SPL paper
##  i have remove tone from phone sequence, final the number of phone set is 59 include <SIL>
if [ ${stage} -le 28 ] && [ ${stop_stage} -ge 28 ];then
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
  model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_1.0_mmi_weight0.5_code_penalty1_gradient_pennalty1.25_seed2
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
       optimization.max_update=150000\
       common.user_dir=${fairseq_dir}/examples/wav2vec/unsupervised \
       model.code_penalty=1\
       model.gradient_penalty=1.25\
       model.smoothness_weight=1.0\
       model.mmi_weight=0.5\
       common.seed=2\
        distributed_training.distributed_world_size=${world_size}\
       distributed_training.distributed_port=-1\
       distributed_training.ddp_backend=legacy_ddp\
       optimization.update_freq=[${update_freq}]\
       common.tensorboard_logdir=$exp_dir\
       hydra.run.dir=$exp_dir\
       hydra.job.name=$exp_dir/w2v_unsup_gan_xp

# grep -rn 'dev_uer' /mntcephfs/lab_data/maduo/exp/wav2vec-u2_gan_from_scratch/w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_1.0_mmi_weight0.5_code_penalty1_gradient_pennalty1.25_seed2/w2v_unsup_gan_xp.log | awk -F ":" '{print $12 $13}' | awk -F "," '{print $2}' | sort
# best dev_uer: 87.2917
# grep -rn '87.2917' /mntcephfs/lab_data/maduo/exp/wav2vec-u2_gan_from_scratch/w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_1.0_mmi_weight0.5_code_penalty1_gradient_pennalty1.25_seed2/w2v_unsup_gan_xp.log 
#1474:[2024-02-12 22:27:40,508][dev][INFO] - {"epoch": 7, "dev_loss": "0.873", "dev_ntokens": "7930.71", "dev_nsentences": "158.897", "dev_sample_size": "7930.71", "dev_lm_score_sum": "-1.03947e+06", "dev_num_pred_chars": "499080", "dev_vocab_seen_pct": "0.711864", "dev_uer": "87.2917", "dev_weighted_lm_ppl": "209.818", "dev_lm_ppl": "106.325", "dev_wps": "32654.7", "dev_wpb": "7930.7", "dev_bsz": "158.9", "dev_num_updates": "60000", "dev_best_weighted_lm_ppl": "120.919"}
# best checkpoint_7_60000.pt

fi

##  i have remove tone from phone sequence, final the number of phone set is 59 include <SIL>
if [ ${stage} -le 29 ] && [ ${stop_stage} -ge 29 ];then
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
  model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_1.25_mmi_weight0.5_code_penalty1_gradient_pennalty1.5_seed2
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
       optimization.max_update=150000\
       common.user_dir=${fairseq_dir}/examples/wav2vec/unsupervised \
       model.code_penalty=1\
       model.gradient_penalty=1.5\
       model.smoothness_weight=1.25\
       model.mmi_weight=0.5\
       common.seed=2\
        distributed_training.distributed_world_size=${world_size}\
       distributed_training.distributed_port=-1\
       distributed_training.ddp_backend=legacy_ddp\
       optimization.update_freq=[${update_freq}]\
       common.tensorboard_logdir=$exp_dir\
       hydra.run.dir=$exp_dir\
       hydra.job.name=$exp_dir/w2v_unsup_gan_xp
# grep -rn 'dev_uer' logs/gan_training_for_Chinese_on_A800_paper_setting_remove_tone_stage29.log | awk -F ":" '{print $12 $13}' | awk -F "," '{print $2}' | sort
# best dev_uer: 86.9254
#  grep -rn '86.9254' logs/gan_training_for_Chinese_on_A800_paper_setting_remove_tone_stage29.log
#3418:[2024-02-14 13:01:31,950][dev][INFO] - {"epoch": 17, "dev_loss": "0.869", "dev_ntokens": "7930.71", "dev_nsentences": "158.897", "dev_sample_size": "7930.71", "dev_lm_score_sum": "-915282", "dev_num_pred_chars": "421070", "dev_vocab_seen_pct": "0.610169", "dev_uer": "86.9254", "dev_weighted_lm_ppl": "341.751", "dev_lm_ppl": "127.236", "dev_wps": "35363.3", "dev_wpb": "7930.7", "dev_bsz": "158.9", "dev_num_updates": "150000", "dev_best_weighted_lm_ppl": "125.959"} 
# best  checkpoint_17_150000.pt
  fi




##  i have remove tone from phone sequence, final the number of phone set is 59 include <SIL>
if [ ${stage} -le 30 ] && [ ${stop_stage} -ge 30 ];then
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
  model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_1.0_mmi_weight0.5_code_penalty0.75_gradient_pennalty1.5_seed2
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
       optimization.max_update=150000\
       common.user_dir=${fairseq_dir}/examples/wav2vec/unsupervised \
       model.code_penalty=0.75\
       model.gradient_penalty=1.5\
       model.smoothness_weight=1.0\
       model.mmi_weight=0.5\
       common.seed=2\
        distributed_training.distributed_world_size=${world_size}\
       distributed_training.distributed_port=-1\
       distributed_training.ddp_backend=legacy_ddp\
       optimization.update_freq=[${update_freq}]\
       common.tensorboard_logdir=$exp_dir\
       hydra.run.dir=$exp_dir\
       hydra.job.name=$exp_dir/w2v_unsup_gan_xp

# grep -rn 'dev_uer' logs/gan_training_for_Chinese_on_A800_paper_setting_remove_tone_stage30.log | awk -F ":" '{print $12 $13}' | awk -F "," '{print $2}' | sort
# best dev_uer: 87.0964
# grep -rn '87.0964' logs/gan_training_for_Chinese_on_A800_paper_setting_remove_tone_stage30.log
#2380:[2024-02-14 11:36:31,178][dev][INFO] - {"epoch": 13, "dev_loss": "0.871", "dev_ntokens": "7930.71", "dev_nsentences": "158.897", "dev_sample_size": "7930.71", "dev_lm_score_sum": "-871891", "dev_num_pred_chars": "402667", "dev_vocab_seen_pct": "0.677966", "dev_uer": "87.0964", "dev_weighted_lm_ppl": "269.777", "dev_lm_ppl": "124", "dev_wps": "37422.9", "dev_wpb": "7930.7", "dev_bsz": "158.9", "dev_num_updates": "114000", "dev_best_weighted_lm_ppl": "121.506"}
# best checkpoint_13_114000.pt


  fi

##  i have remove tone from phone sequence, final the number of phone set is 59 include <SIL>
if [ ${stage} -le 31 ] && [ ${stop_stage} -ge 31 ];then
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
  model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_1.25_mmi_weight0.5_code_penalty1_gradient_pennalty1.5_seed2_300k_steps
  exp_dir=$dir/wav2vec-u2_gan_from_scratch/${model_name}
  mkdir -p $exp_dir
  cp -r $dir/wav2vec-u2_gan_from_scratch/w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_1.25_mmi_weight0.5_code_penalty1_gradient_pennalty1.5_seed2/checkpoints/checkpoint_last.pt $exp_dir/checkpoints/
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
       model.smoothness_weight=1.25\
       model.mmi_weight=0.5\
       common.seed=2\
        distributed_training.distributed_world_size=${world_size}\
       distributed_training.distributed_port=-1\
       distributed_training.ddp_backend=legacy_ddp\
       optimization.update_freq=[${update_freq}]\
       common.tensorboard_logdir=$exp_dir\
       hydra.run.dir=$exp_dir\
       hydra.job.name=$exp_dir/w2v_unsup_gan_xp

# grep -rn 'dev_uer' logs/gan_training_for_Chinese_on_A800_paper_setting_remove_tone_stage31.log | awk -F ":" '{print $12 $13}' | awk -F "," '{print $2}' | sort
#best "dev_uer" "84.4818"
#  grep -rn '84.4818' logs/gan_training_for_Chinese_on_A800_paper_setting_remove_tone_stage31.log 
#3316:[2024-02-15 22:33:35,679][dev][INFO] - {"epoch": 32, "dev_loss": "0.845", "dev_ntokens": "7930.71", "dev_nsentences": "158.897", "dev_sample_size": "7930.71", "dev_lm_score_sum": "-1.27631e+06", "dev_num_pred_chars": "648204", "dev_vocab_seen_pct": "0.651861", "dev_uer": "84.4818", "dev_weighted_lm_ppl": "199.328", "dev_lm_ppl": "84.699", "dev_wps": "27396.2", "dev_wpb": "7930.7", "dev_bsz": "158.9", "dev_num_updates": "296000", "dev_best_weighted_lm_ppl": "125.959"} 
# best checkpoint_32_296000.pt
fi

##  i have remove tone from phone sequence, final the number of phone set is 59 include <SIL>
if [ ${stage} -le 32 ] && [ ${stop_stage} -ge 32 ];then
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
  model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_1.5_mmi_weight0.5_code_penalty1_gradient_pennalty1.5_seed2
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
       model.smoothness_weight=1.5\
       model.mmi_weight=0.5\
       common.seed=2\
        distributed_training.distributed_world_size=${world_size}\
       distributed_training.distributed_port=-1\
       distributed_training.ddp_backend=legacy_ddp\
       optimization.update_freq=[${update_freq}]\
       common.tensorboard_logdir=$exp_dir\
       hydra.run.dir=$exp_dir\
       hydra.job.name=$exp_dir/w2v_unsup_gan_xp

# grep -rn 'dev_uer' logs/gan_training_for_Chinese_on_A800_paper_setting_remove_tone_stage32.log | awk -F ":" '{print $12 $13}' | awk -F "," '{print $2}' | sort
# "dev_uer" "85.1917"
# grep -rn '85.1917' logs/gan_training_for_Chinese_on_A800_paper_setting_remove_tone_stage32.log 
#6364:[2024-02-16 16:15:56,815][dev][INFO] - {"epoch": 32, "dev_loss": "0.852", "dev_ntokens": "7930.71", "dev_nsentences": "158.897", "dev_sample_size": "7930.71", "dev_lm_score_sum": "-973901", "dev_num_pred_chars": "443424", "dev_vocab_seen_pct": "0.338983", "dev_uer": "85.1917", "dev_weighted_lm_ppl": "1173.71", "dev_lm_ppl": "134.871", "dev_wps": "35959.4", "dev_wpb": "7930.7", "dev_bsz": "158.9", "dev_num_updates": "292000", "dev_best_weighted_lm_ppl": "127.178"} 
# best checkpoint_32_292000.pt

fi
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

##  i have remove tone from phone sequence, final the number of phone set is 59 include <SIL>
if [ ${stage} -le 34 ] && [ ${stop_stage} -ge 34 ];then
  fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
  des_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence

  ## it stores hubert large model 15 layer representation of raw librispeech speech,
  ## it  removes  silence. it offers feature of speech.
  TASK_DATA=$des_dir/hubert_large_feat_dir_no_silence
   ## offer mfcc pesudo label from without silence unpair audio
   #cp -r /mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir/mfcc_no_silence/mfcc_lab//* $TASK_DATA
   #cp -r $TASK_DATA/remove_phone_tone/* $TASK_DATA
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
  model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_2.0_mmi_weight0.5_code_penalty1_gradient_pennalty1.5_seed2_300k_steps
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
       model.smoothness_weight=2.0\
       model.mmi_weight=0.5\
       common.seed=2\
        distributed_training.distributed_world_size=${world_size}\
       distributed_training.distributed_port=-1\
       distributed_training.ddp_backend=legacy_ddp\
       optimization.update_freq=[${update_freq}]\
       common.tensorboard_logdir=$exp_dir\
       hydra.run.dir=$exp_dir\
       hydra.job.name=$exp_dir/w2v_unsup_gan_xp

# grep -rn 'dev_uer' logs/gan_training_for_Chinese_on_A800_paper_setting_remove_tone_stage34.log | awk -F ":" '{print $12 $13}' | awk -F "," '{print $2}' | sort
# best "dev_uer" "86.5877"
# grep -rn "86.5877" logs/gan_training_for_Chinese_on_A800_paper_setting_remove_tone_stage34.log
#  grep -rn "86.5877" logs/gan_training_for_Chinese_on_A800_paper_setting_remove_tone_stage34.log
# 2257:[2024-02-18 10:38:26,451][dev][INFO] - {"epoch": 12, "dev_loss": "0.866", "dev_ntokens": "7930.71", "dev_nsentences": "158.897", "dev_sample_size": "7930.71", "dev_lm_score_sum": "-1.15255e+06", "dev_num_pred_chars": "509946", "dev_vocab_seen_pct": "0.610169", "dev_uer": "86.5877", "dev_weighted_lm_ppl": "426.181", "dev_lm_ppl": "158.67", "dev_wps": "35042.1", "dev_wpb": "7930.7", "dev_bsz": "158.9", "dev_num_updates": "112000", "dev_best_weighted_lm_ppl": "126.135"}
# best checkpoint_12_112000.pt
fi

##  i have remove tone from phone sequence, final the number of phone set is 59 include <SIL>
if [ ${stage} -le 35 ] && [ ${stop_stage} -ge 35 ];then
  fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
  des_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence

  ## it stores hubert large model 15 layer representation of raw librispeech speech,
  ## it  removes  silence. it offers feature of speech.
  TASK_DATA=$des_dir/hubert_large_feat_dir_no_silence
   ## offer mfcc pesudo label from without silence unpair audio
   #cp -r /mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir/mfcc_no_silence/mfcc_lab//* $TASK_DATA
   #cp -r $TASK_DATA/remove_phone_tone/* $TASK_DATA
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
  model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_2.25_mmi_weight0.5_code_penalty1_gradient_pennalty1.5_seed2_300k_steps
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
       model.smoothness_weight=2.25\
       model.mmi_weight=0.5\
       common.seed=2\
        distributed_training.distributed_world_size=${world_size}\
       distributed_training.distributed_port=-1\
       distributed_training.ddp_backend=legacy_ddp\
       optimization.update_freq=[${update_freq}]\
       common.tensorboard_logdir=$exp_dir\
       hydra.run.dir=$exp_dir\
       hydra.job.name=$exp_dir/w2v_unsup_gan_xp
# grep -rn 'dev_uer' logs/gan_training_for_Chinese_on_A800_paper_setting_remove_tone_stage35.log | awk -F ":" '{print $12 $13}' | awk -F "," '{print $2}' | sort
# best "dev_uer" "84.5773"
# grep -rn '84.5773' logs/gan_training_for_Chinese_on_A800_paper_setting_remove_tone_stage35.log
#486:[2024-02-17 23:03:04,090][dev][INFO] - {"epoch": 4, "dev_loss": "0.846", "dev_ntokens": "7930.71", "dev_nsentences": "158.897", "dev_sample_size": "7930.71", "dev_lm_score_sum": "-1.36271e+06", "dev_num_pred_chars": "711668", "dev_vocab_seen_pct": "0.675628", "dev_uer": "84.5773", "dev_weighted_lm_ppl": "165.543", "dev_lm_ppl": "75.5662", "dev_wps": "28923.2", "dev_wpb": "7930.7", "dev_bsz": "158.9", "dev_num_updates": "31000", "dev_best_weighted_lm_ppl": "144.065"} 
# best checkpoint_4_31000.pt
fi

##  i have remove tone from phone sequence, final the number of phone set is 59 include <SIL>
if [ ${stage} -le 36 ] && [ ${stop_stage} -ge 36 ];then
  fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
  des_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence

  ## it stores hubert large model 15 layer representation of raw librispeech speech,
  ## it  removes  silence. it offers feature of speech.
  TASK_DATA=$des_dir/hubert_large_feat_dir_no_silence
   ## offer mfcc pesudo label from without silence unpair audio
   #cp -r /mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir/mfcc_no_silence/mfcc_lab//* $TASK_DATA
   #cp -r $TASK_DATA/remove_phone_tone/* $TASK_DATA
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
  model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_1.75_mmi_weight0.5_code_penalty1.25_gradient_pennalty1.5_seed2_300k_steps
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
       model.code_penalty=1.25\
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
fi



## 2024-2-19, I will use wav2vec2 large model as feature extractor for GAN model
##  i have remove tone from phone sequence, final the number of phone set is 59 include <SIL>
if [ ${stage} -le 40 ] && [ ${stop_stage} -ge 40 ];then
  fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
  des_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence

  ## it stores wav2vec2 large model 15 layer representation of raw wenetspeech speech,
  ## it  removes  silence. it offers feature of speech.
  TASK_DATA=$des_dir/wav2vec2_large_feat_dir_no_silence
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
  model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_1.75_mmi_weight0.5_code_penalty1_gradient_pennalty1.5_seed2_300k_steps_wav2vec2_large_feat
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
# grep -rn 'dev_uer' /mntcephfs/lab_data/maduo/exp/wav2vec-u2_gan_from_scratch/w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_1.75_mmi_weight0.5_code_penalty1_gradient_pennalty1.5_seed2_300k_steps_wav2vec2_large_feat/w2v_unsup_gan_xp.log | awk -F ":" '{print $12 $13}' | awk -F "," '{print $2}' |sort
# best dev_uer:85.0197
#  grep -rn '85.0197'  /mntcephfs/lab_data/maduo/exp/wav2vec-u2_gan_from_scratch/w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_1.75_mmi_weight0.5_code_penalty1_gradient_pennalty1.5_seed2_300k_steps_wav2vec2_large_feat/w2v_unsup_gan_xp.log
# 3161:[2024-02-21 13:05:39,814][dev][INFO] - {"epoch": 14, "dev_loss": "0.85", "dev_ntokens": "7930.71", "dev_nsentences": "158.897", "dev_sample_size": "7930.71", "dev_lm_score_sum": "-1.31863e+06", "dev_num_pred_chars": "580129", "dev_vocab_seen_pct": "0.608416", "dev_uer": "85.0197", "dev_weighted_lm_ppl": "448.417", "dev_lm_ppl": "165.991", "dev_wps": "35982.4", "dev_wpb": "7930.7", "dev_bsz": "158.9", "dev_num_updates": "128000", "dev_best_weighted_lm_ppl": "204.135"}
# best checkpoint_14_128000.pt


fi


## 2024-2-19, I will use wav2vec2 large model as feature extractor for GAN model
##  i have remove tone from phone sequence, final the number of phone set is 59 include <SIL>
## and I will reduce add silence probablity for unpair text phone sequence.
if [ ${stage} -le 41 ] && [ ${stop_stage} -ge 41 ];then
  fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
  des_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence

  ## it stores wav2vec2 large model 15 layer representation of raw wenetspeech speech,
  ## it  removes  silence. it offers feature of speech.
  TASK_DATA=$des_dir/wav2vec2_large_feat_dir_no_silence
   ## offer mfcc pesudo label from without silence unpair audio
   cp -r /mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir/mfcc_no_silence/mfcc_lab//* $TASK_DATA
   #cp -r $TASK_DATA/remove_phone_tone/* $TASK_DATA
   head $TASK_DATA/*.phn

  # Unpaired text input
  #text_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_again/
  text_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone_sil_0.25
  cp -r $text_dir/unpair_text/dict.phn.txt  $TASK_DATA
  TEXT_DATA=$text_dir/unpair_text
  ## in order to match unpair speech train set name
  mv $TEXT_DATA/train.bin  $TEXT_DATA/train_m.bin
  mv $TEXT_DATA/train.idx $TEXT_DATA/train_m.idx
  #KENLM_PATH=$text_dir/lm.phones.filtered.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)
  KENLM_PATH=$text_dir/lm.phones.filtered.04.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)
  config_dir=$fairseq_dir/examples/wav2vec/unsupervised

  dir=/mntcephfs/lab_data/maduo/exp/
  model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_1.75_mmi_weight0.5_code_penalty1_gradient_pennalty1.5_seed2_300k_steps_wav2vec2_large_feat_phone_sil_0.25
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

#  grep -rn 'dev_uer' /mntcephfs/lab_data/maduo/exp/wav2vec-u2_gan_from_scratch/w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_1.75_mmi_weight0.5_code_penalty1_gradient_pennalty1.5_seed2_300k_steps_wav2vec2_large_feat_phone_sil_0.25/w2v_unsup_gan_xp.log | awk -F ":" '{print $12 $13}' | awk -F "," '{print $2}' |sort
# best dev_uer: 86.5012
# grep -rn '86.5012' /mntcephfs/lab_data/maduo/exp/wav2vec-u2_gan_from_scratch/w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_1.75_mmi_weight0.5_code_penalty1_gradient_pennalty1.5_seed2_300k_steps_wav2vec2_large_feat_phone_sil_0.25/w2v_unsup_gan_xp.log
#6257:[2024-02-22 01:05:32,769][dev][INFO] - {"epoch": 30, "dev_loss": "0.865", "dev_ntokens": "7930.71", "dev_nsentences": "158.897", "dev_sample_size": "7930.71", "dev_lm_score_sum": "-1.20262e+06", "dev_num_pred_chars": "589623", "dev_vocab_seen_pct": "0.49776", "dev_uer": "86.5012", "dev_weighted_lm_ppl": "397.079", "dev_lm_ppl": "98.3821", "dev_wps": "35232.5", "dev_wpb": "7930.7", "dev_bsz": "158.9", "dev_num_updates": "278000", "dev_best_weighted_lm_ppl": "153.187"}
# best checkpoint_30_278000.pt
fi


## 2024-2-19, I will use wav2vec2 large model as feature extractor for GAN model
##  i have remove tone from phone sequence, final the number of phone set is 59 include <SIL>
## and I will increase add silence probablity for unpair text phone sequence.
if [ ${stage} -le 42 ] && [ ${stop_stage} -ge 42 ];then
  fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
  des_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence

  ## it stores wav2vec2 large model 15 layer representation of raw wenetspeech speech,
  ## it  removes  silence. it offers feature of speech.
  TASK_DATA=$des_dir/wav2vec2_large_feat_dir_no_silence
   ## offer mfcc pesudo label from without silence unpair audio
   cp -r /mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir/mfcc_no_silence/mfcc_lab//* $TASK_DATA
   #cp -r $TASK_DATA/remove_phone_tone/* $TASK_DATA
   head $TASK_DATA/*.phn

  # Unpaired text input
  #text_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_again/
  text_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone_sil_0.6
  #cp -r $text_dir/unpair_text/dict.phn.txt  $TASK_DATA
  TEXT_DATA=$text_dir/unpair_text
  ## in order to match unpair speech train set name
  mv $TEXT_DATA/train.bin  $TEXT_DATA/train_m.bin
  mv $TEXT_DATA/train.idx $TEXT_DATA/train_m.idx
  #KENLM_PATH=$text_dir/lm.phones.filtered.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)
  KENLM_PATH=$text_dir/lm.phones.filtered.04.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)
  config_dir=$fairseq_dir/examples/wav2vec/unsupervised

  dir=/mntcephfs/lab_data/maduo/exp/
  model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_1.75_mmi_weight0.5_code_penalty1_gradient_pennalty1.5_seed2_300k_steps_wav2vec2_large_feat_phone_sil_0.6
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
#grep -rn 'dev_uer' logs/gan_training_for_Chinese_on_A800_paper_setting_remove_tone_stage42.log | awk -F ":" '{print $12 $13}' | awk -F "," '{print $2}' | sort
# best "dev_uer" "86.4194"
# grep -rn '86.4194'  logs/gan_training_for_Chinese_on_A800_paper_setting_remove_tone_stage42.log 
#1802:[2024-02-22 20:32:43,264][dev][INFO] - {"epoch": 10, "dev_loss": "0.864", "dev_ntokens": "7930.71", "dev_nsentences": "158.897", "dev_sample_size": "7930.71", "dev_lm_score_sum": "-1.12677e+06", "dev_num_pred_chars": "530788", "dev_vocab_seen_pct": "0.508475", "dev_uer": "86.4194", "dev_weighted_lm_ppl": "453.324", "dev_lm_ppl": "117.205", "dev_wps": "34911.7", "dev_wpb": "7930.7", "dev_bsz": "158.9", "dev_num_updates": "90000", "dev_best_weighted_lm_ppl": "220.619"}
# best checkpoint_10_90000.pt
fi


## 2024-2-19, I will use wav2vec2 large model as feature extractor for GAN model
##  i have remove tone from phone sequence, final the number of phone set is 59 include <SIL>
if [ ${stage} -le 43 ] && [ ${stop_stage} -ge 43 ];then
  fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
  des_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence

  ## it stores wav2vec2 large model 15 layer representation of raw wenetspeech speech,
  ## it  removes  silence. it offers feature of speech.
  TASK_DATA=$des_dir/wav2vec2_large_feat_dir_no_silence
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
  model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_1.75_mmi_weight0.5_code_penalty0_gradient_pennalty1.5_seed2_300k_steps_wav2vec2_large_feat_300k_steps
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
       model.code_penalty=0\
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


  fi


## 2024-2-19, I will use wav2vec2 large model as feature extractor for GAN model
##  i have remove tone from phone sequence, final the number of phone set is 59 include <SIL>
if [ ${stage} -le 44 ] && [ ${stop_stage} -ge 44 ];then
  fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
  des_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence

  ## it stores wav2vec2 large model 15 layer representation of raw wenetspeech speech,
  ## it  removes  silence. it offers feature of speech.
  TASK_DATA=$des_dir/wav2vec2_large_feat_dir_no_silence
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
  model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_1.75_mmi_weight0.5_code_penalty1_gradient_pennalty1.5_seed1_300k_steps_wav2vec2_large_feat_300k_steps
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
       common.seed=1\
        distributed_training.distributed_world_size=${world_size}\
       distributed_training.distributed_port=-1\
       distributed_training.ddp_backend=legacy_ddp\
       optimization.update_freq=[${update_freq}]\
       common.tensorboard_logdir=$exp_dir\
       hydra.run.dir=$exp_dir\
       hydra.job.name=$exp_dir/w2v_unsup_gan_xp


  fi


  ## 2024-2-19, I will use wav2vec2 large model as feature extractor for GAN model
##  i have remove tone from phone sequence, final the number of phone set is 59 include <SIL>
if [ ${stage} -le 45 ] && [ ${stop_stage} -ge 45 ];then
  fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
  des_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence

  ## it stores wav2vec2 large model 15 layer representation of raw wenetspeech speech,
  ## it  removes  silence. it offers feature of speech.
  TASK_DATA=$des_dir/wav2vec2_large_feat_dir_no_silence
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
  model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_1.75_mmi_weight0.5_code_penalty1_gradient_pennalty1.5_seed3_300k_steps_wav2vec2_large_feat_300k_steps  exp_dir=$dir/wav2vec-u2_gan_from_scratch/${model_name}
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
       common.seed=3\
        distributed_training.distributed_world_size=${world_size}\
       distributed_training.distributed_port=-1\
       distributed_training.ddp_backend=legacy_ddp\
       optimization.update_freq=[${update_freq}]\
       common.tensorboard_logdir=$exp_dir\
       hydra.run.dir=$exp_dir\
       hydra.job.name=$exp_dir/w2v_unsup_gan_xp
# grep -rn 'dev_uer' /mntcephfs/lab_data/maduo/exp/wav2vec-u2_gan_from_scratch/w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_1.75_mmi_weight0.5_code_penalty1_gradient_pennalty1.5_seed3_300k_steps_wav2vec2_large_feat_300k_steps/w2v_unsup_gan_xp.log | awk -F ":" '{print $12 $13}' | awk -F "," '{print $2}'|sort
# best "dev_uer" "86.2431"
# grep -rn '86.2431' /mntcephfs/lab_data/maduo/exp/wav2vec-u2_gan_from_scratch/w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_1.75_mmi_weight0.5_code_penalty1_gradient_pennalty1.5_seed3_300k_steps_wav2vec2_large_feat_300k_steps/w2v_unsup_gan_xp.log
# 3628:[2024-02-24 22:16:19,706][dev][INFO] - {"epoch": 18, "dev_loss": "0.862", "dev_ntokens": "7930.71", "dev_nsentences": "158.897", "dev_sample_size": "7930.71", "dev_lm_score_sum": "-1.27836e+06", "dev_num_pred_chars": "527097", "dev_vocab_seen_pct": "0.256965", "dev_uer": "86.2431", "dev_weighted_lm_ppl": "3495.88", "dev_lm_ppl": "230.836", "dev_wps": "37734.2", "dev_wpb": "7930.7", "dev_bsz": "158.9", "dev_num_updates": "162000", "dev_best_weighted_lm_ppl": "185.085"}
# best checkpoint_18_86.2431.pt
  fi


## 2024-2-26 I continue to use hubert large model as feature extractor for training GAN
##  i have remove tone from phone sequence, final the number of phone set is 59 include <SIL>
if [ ${stage} -le 50 ] && [ ${stop_stage} -ge 50 ];then
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
  model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_1.75_mmi_weight0.5_code_penalty1_gradient_pennalty1.5_seed3_300k_steps
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
       common.seed=3\
        distributed_training.distributed_world_size=${world_size}\
       distributed_training.distributed_port=-1\
       distributed_training.ddp_backend=legacy_ddp\
       optimization.update_freq=[${update_freq}]\
       common.tensorboard_logdir=$exp_dir\
       hydra.run.dir=$exp_dir\
       hydra.job.name=$exp_dir/w2v_unsup_gan_xp
fi


## 2024-2-26 I continue to use hubert large model as feature extractor for training GAN
##  i have remove tone from phone sequence, final the number of phone set is 59 include <SIL>
if [ ${stage} -le 51 ] && [ ${stop_stage} -ge 51 ];then
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
  model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_1.75_mmi_weight0.5_code_penalty1_gradient_pennalty1.5_seed4_300k_steps
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
       common.seed=4\
        distributed_training.distributed_world_size=${world_size}\
       distributed_training.distributed_port=-1\
       distributed_training.ddp_backend=legacy_ddp\
       optimization.update_freq=[${update_freq}]\
       common.tensorboard_logdir=$exp_dir\
       hydra.run.dir=$exp_dir\
       hydra.job.name=$exp_dir/w2v_unsup_gan_xp
fi


## 2024-2-26 I continue to use hubert large model as feature extractor for training GAN
##  i have remove tone from phone sequence, final the number of phone set is 59 include <SIL>
if [ ${stage} -le 52 ] && [ ${stop_stage} -ge 52 ];then
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
  model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_1.75_mmi_weight0.5_code_penalty0.75_gradient_pennalty1.5_seed2_300k_steps
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
       model.code_penalty=0.75\
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
fi

## 2024-2-26 I continue to use hubert large model as feature extractor for training GAN
##  i have remove tone from phone sequence, final the number of phone set is 59 include <SIL>
if [ ${stage} -le 53 ] && [ ${stop_stage} -ge 53 ];then
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
  model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_1.75_mmi_weight0.5_code_penalty0.75_gradient_pennalty1.5_seed4_300k_steps
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
       model.code_penalty=0.75\
       model.gradient_penalty=1.5\
       model.smoothness_weight=1.75\
       model.mmi_weight=0.5\
       common.seed=4\
       distributed_training.distributed_world_size=${world_size}\
       distributed_training.distributed_port=-1\
       distributed_training.ddp_backend=legacy_ddp\
       optimization.update_freq=[${update_freq}]\
       common.tensorboard_logdir=$exp_dir\
       hydra.run.dir=$exp_dir\
       hydra.job.name=$exp_dir/w2v_unsup_gan_xp
fi


## remove lower frequency phone, then get new phone dictionary, and remove contain lower frequecy phone utterance (unpair text and dev set (speech and text))
if [ ${stage} -le 54 ] && [ ${stop_stage} -ge 54 ];then
  fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
  des_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence

  ## it stores hubert large model 15 layer representation of raw librispeech speech,
  ## it  removes  silence. it offers feature of speech.
  TASK_DATA=$des_dir/hubert_large_feat_dir_no_silence
   ## offer mfcc pesudo label from without silence unpair audio
   #cp -r /mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir/mfcc_no_silence/mfcc_lab//* $TASK_DATA
   cp -r /mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir_remove_lower_freq_phone/*.km $TASK_DATA
   #cp -r $TASK_DATA/remove_phone_tone/* $TASK_DATA
   cp -r  /mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir_remove_lower_freq_phone/*.phn $TASK_DATA 
   head $TASK_DATA/*.phn

  # Unpaired text input
  #text_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_again/
  #text_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone
  text_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone_remove_lower_frequecy_phones
  cat $text_dir/unpair_text/dict.txt > $TASK_DATA/dict.phn.txt
  TEXT_DATA=$text_dir/unpair_text
  ## in order to match unpair speech train set name
  mv $TEXT_DATA/train.bin  $TEXT_DATA/train_m.bin
  mv $TEXT_DATA/train.idx $TEXT_DATA/train_m.idx
  #KENLM_PATH=$text_dir/lm.phones.filtered.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)
  KENLM_PATH=$text_dir/lm.phones.filtered.04.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)
  config_dir=$fairseq_dir/examples/wav2vec/unsupervised

  dir=/mntcephfs/lab_data/maduo/exp/
  ## 
  model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_1.75_mmi_weight0.5_code_penalty1_gradient_pennalty1.5_seed2_300k_steps_remove_lower_frequency_phones
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

fi


## I remove lower frequency phone, then get new phone dictionary, and remove contain lower frequecy phone utterance (unpair text and dev set (speech and text))
## I use transcript of dev speech as unpair dev text.
if [ ${stage} -le 55 ] && [ ${stop_stage} -ge 55 ];then
  fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
  des_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence

  ## it stores hubert large model 15 layer representation of raw librispeech speech,
  ## it  removes  silence. it offers feature of speech.
  TASK_DATA=$des_dir/hubert_large_feat_dir_no_silence
   ## offer mfcc pesudo label from without silence unpair audio
   #cp -r /mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir/mfcc_no_silence/mfcc_lab//* $TASK_DATA
   #cp -r $TASK_DATA/remove_phone_tone/* $TASK_DATA
  cp -r /mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir_remove_lower_freq_phone/*.km $TASK_DATA
  cp -r  /mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir_remove_lower_freq_phone/*.phn $TASK_DATA
   head $TASK_DATA/*.phn

  # Unpaired text input
  #text_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_again/
  #text_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone
  text_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone_remove_lower_frequecy_phones
  cat $text_dir/unpair_text_with_dev/dict.txt > $TASK_DATA/dict.phn.txt
  TEXT_DATA=$text_dir/unpair_text_with_dev
  ## in order to match unpair speech train set name
  mv $TEXT_DATA/train.bin  $TEXT_DATA/train_m.bin
  mv $TEXT_DATA/train.idx $TEXT_DATA/train_m.idx
  #KENLM_PATH=$text_dir/lm.phones.filtered.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)
  KENLM_PATH=$text_dir/lm.phones.filtered.04.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)
  config_dir=$fairseq_dir/examples/wav2vec/unsupervised

  dir=/mntcephfs/lab_data/maduo/exp/
  model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_1.75_mmi_weight0.5_code_penalty1_gradient_pennalty1.5_seed2_300k_steps_remove_lower_frequency_phones_with_dev
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

fi
## remove lower frequency phone, then get new phone dictionary, and remove contain lower frequecy phone utterance (unpair text and dev set (speech and text))
if [ ${stage} -le 56 ] && [ ${stop_stage} -ge 56 ];then
  fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
  des_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence

  ## it stores hubert large model 15 layer representation of raw librispeech speech,
  ## it  removes  silence. it offers feature of speech.
  TASK_DATA=$des_dir/hubert_large_feat_dir_no_silence
   ## offer mfcc pesudo label from without silence unpair audio
   #cp -r /mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir/mfcc_no_silence/mfcc_lab//* $TASK_DATA
   cp -r /mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir_remove_lower_freq_phone/*.km $TASK_DATA
   #cp -r $TASK_DATA/remove_phone_tone/* $TASK_DATA
   cp -r  /mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir_remove_lower_freq_phone/*.phn $TASK_DATA
   head $TASK_DATA/*.phn

  # Unpaired text input
  #text_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_again/
  #text_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone
  text_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone_remove_lower_frequecy_phones
  cat $text_dir/unpair_text/dict.txt > $TASK_DATA/dict.phn.txt
  TEXT_DATA=$text_dir/unpair_text
  ## in order to match unpair speech train set name
  #mv $TEXT_DATA/train.bin  $TEXT_DATA/train_m.bin
  #mv $TEXT_DATA/train.idx $TEXT_DATA/train_m.idx
  #KENLM_PATH=$text_dir/lm.phones.filtered.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)
  KENLM_PATH=$text_dir/lm.phones.filtered.04.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)
  config_dir=$fairseq_dir/examples/wav2vec/unsupervised

  dir=/mntcephfs/lab_data/maduo/exp/
  ##
  model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_1.5_mmi_weight0.5_code_penalty0_gradient_pennalty1.5_seed1_300k_steps_remove_lower_frequency_phones  exp_dir=$dir/wav2vec-u2_gan_from_scratch/${model_name}
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
       model.code_penalty=0\
       model.gradient_penalty=1.5\
       model.smoothness_weight=1.5\
       model.mmi_weight=0.5\
       common.seed=1\
       distributed_training.distributed_world_size=${world_size}\
       distributed_training.distributed_port=-1\
       distributed_training.ddp_backend=legacy_ddp\
       optimization.update_freq=[${update_freq}]\
       common.tensorboard_logdir=$exp_dir\
       hydra.run.dir=$exp_dir\
       hydra.job.name=$exp_dir/w2v_unsup_gan_xp

fi


## remove lower frequency phone, then get new phone dictionary, and remove contain lower frequecy phone utterance (unpair text and dev set (speech and text))
if [ ${stage} -le 57 ] && [ ${stop_stage} -ge 57 ];then
  fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
  des_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence

  ## it stores hubert large model 15 layer representation of raw librispeech speech,
  ## it  removes  silence. it offers feature of speech.
  TASK_DATA=$des_dir/hubert_large_feat_dir_no_silence
   ## offer mfcc pesudo label from without silence unpair audio
   #cp -r /mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir/mfcc_no_silence/mfcc_lab//* $TASK_DATA
   cp -r /mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir_remove_lower_freq_phone/*.km $TASK_DATA
   #cp -r $TASK_DATA/remove_phone_tone/* $TASK_DATA
   cp -r  /mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir_remove_lower_freq_phone/*.phn $TASK_DATA
   head $TASK_DATA/*.phn

  # Unpaired text input
  #text_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_again/
  #text_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone
  text_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone_remove_lower_frequecy_phones
  cat $text_dir/unpair_text/dict.txt > $TASK_DATA/dict.phn.txt
  TEXT_DATA=$text_dir/unpair_text
  ## in order to match unpair speech train set name
  #mv $TEXT_DATA/train.bin  $TEXT_DATA/train_m.bin
  #mv $TEXT_DATA/train.idx $TEXT_DATA/train_m.idx
  #KENLM_PATH=$text_dir/lm.phones.filtered.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)
  KENLM_PATH=$text_dir/lm.phones.filtered.04.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)
  config_dir=$fairseq_dir/examples/wav2vec/unsupervised

  dir=/mntcephfs/lab_data/maduo/exp/
  ##
  model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_0.5_mmi_weight0.5_code_penalty2_gradient_pennalty1.5_seed0_300k_steps_remove_lower_frequency_phones  exp_dir=$dir/wav2vec-u2_gan_from_scratch/${model_name}
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
       model.code_penalty=2\
       model.gradient_penalty=1.5\
       model.smoothness_weight=0.5\
       model.mmi_weight=0.5\
       common.seed=0\
       distributed_training.distributed_world_size=${world_size}\
       distributed_training.distributed_port=-1\
       distributed_training.ddp_backend=legacy_ddp\
       optimization.update_freq=[${update_freq}]\
       common.tensorboard_logdir=$exp_dir\
       hydra.run.dir=$exp_dir\
       hydra.job.name=$exp_dir/w2v_unsup_gan_xp

fi
## remove lower frequency phone, then get new phone dictionary, and remove contain lower frequecy phone utterance (unpair text and dev set (speech and text))
if [ ${stage} -le 58 ] && [ ${stop_stage} -ge 58 ];then
  fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
  des_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence

  ## it stores hubert large model 15 layer representation of raw librispeech speech,
  ## it  removes  silence. it offers feature of speech.
  TASK_DATA=$des_dir/hubert_large_feat_dir_no_silence
   ## offer mfcc pesudo label from without silence unpair audio
   #cp -r /mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir/mfcc_no_silence/mfcc_lab//* $TASK_DATA
   cp -r /mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir_remove_lower_freq_phone/*.km $TASK_DATA
   #cp -r $TASK_DATA/remove_phone_tone/* $TASK_DATA
   cp -r  /mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir_remove_lower_freq_phone/*.phn $TASK_DATA
   head $TASK_DATA/*.phn

  # Unpaired text input
  #text_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_again/
  #text_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone
  text_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone_remove_lower_frequecy_phones
  cat $text_dir/unpair_text/dict.txt > $TASK_DATA/dict.phn.txt
  TEXT_DATA=$text_dir/unpair_text
  ## in order to match unpair speech train set name
  #mv $TEXT_DATA/train.bin  $TEXT_DATA/train_m.bin
  #mv $TEXT_DATA/train.idx $TEXT_DATA/train_m.idx
  #KENLM_PATH=$text_dir/lm.phones.filtered.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)
  KENLM_PATH=$text_dir/lm.phones.filtered.04.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)
  config_dir=$fairseq_dir/examples/wav2vec/unsupervised

  dir=/mntcephfs/lab_data/maduo/exp/
  ##
  model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_1.25_mmi_weight0.5_code_penalty1_gradient_pennalty1.5_seed2_300k_steps_remove_lower_frequency_phones
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
       model.smoothness_weight=1.25\
       model.mmi_weight=0.5\
       common.seed=2\
       distributed_training.distributed_world_size=${world_size}\
       distributed_training.distributed_port=-1\
       distributed_training.ddp_backend=legacy_ddp\
       optimization.update_freq=[${update_freq}]\
       common.tensorboard_logdir=$exp_dir\
       hydra.run.dir=$exp_dir\
       hydra.job.name=$exp_dir/w2v_unsup_gan_xp
# grep -rn 'dev_lm_ppl' logs/gan_training_for_Chinese_on_A800_paper_setting_remove_tone_remove_lower_freq_stage58.log |awk -F ":" '{print $14 $15}' | awk -F "," '{print $2}' | sort
# best dev_lm_ppl:61.9186
#grep -rn '61.9186' logs/gan_training_for_Chinese_on_A800_paper_setting_remove_tone_remove_lower_freq_stage58.log
#6024:[2024-03-02 02:46:07,006][dev][INFO] - {"epoch": 30, "dev_loss": "0.971", "dev_ntokens": "7924.77", "dev_nsentences": "159.429", "dev_sample_size": "7924.77", "dev_lm_score_sum": "-1.29042e+06", "dev_num_pred_chars": "706781", "dev_vocab_seen_pct": "0.709609", "dev_uer": "97.0935", "dev_weighted_lm_ppl": "122.965", "dev_lm_ppl": "61.9186", "dev_wps": "34200.6", "dev_wpb": "7924.8", "dev_bsz": "159.4", "dev_num_updates": "272000", "dev_best_weighted_lm_ppl": "111.282"} 
#  grep -rn 'dev_uer' logs/gan_training_for_Chinese_on_A800_paper_setting_remove_tone_remove_lower_freq_stage58.log |awk -F ":" '{print $12 $13}' | awk -F "," '{print $2}' | sort
#  best "dev_uer" "87.864"
# grep -rn '87.864' logs/gan_training_for_Chinese_on_A800_paper_setting_remove_tone_remove_lower_freq_stage58.log
#3252:[2024-03-01 07:54:22,681][dev][INFO] - {"epoch": 16, "dev_loss": "0.879", "dev_ntokens": "7924.77", "dev_nsentences": "159.429", "dev_sample_size": "7924.77", "dev_lm_score_sum": "-775193", "dev_num_pred_chars": "346518", "dev_vocab_seen_pct": "0.764031", "dev_uer": "87.864", "dev_weighted_lm_ppl": "244.136", "dev_lm_ppl": "142.513", "dev_wps": "34602.6", "dev_wpb": "7924.8", "dev_bsz": "159.4", "dev_num_updates": "144000", "dev_best_weighted_lm_ppl": "111.282"}


fi

## remove lower frequency phone, then get new phone dictionary, and remove contain lower frequecy phone utterance (unpair text and dev set (speech and text))
if [ ${stage} -le 59 ] && [ ${stop_stage} -ge 59 ];then
  fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
  des_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence

  ## it stores hubert large model 15 layer representation of raw librispeech speech,
  ## it  removes  silence. it offers feature of speech.
  TASK_DATA=$des_dir/hubert_large_feat_dir_no_silence
   ## offer mfcc pesudo label from without silence unpair audio
   #cp -r /mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir/mfcc_no_silence/mfcc_lab//* $TASK_DATA
   cp -r /mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir_remove_lower_freq_phone/*.km $TASK_DATA
   #cp -r $TASK_DATA/remove_phone_tone/* $TASK_DATA
   cp -r  /mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir_remove_lower_freq_phone/*.phn $TASK_DATA
   head $TASK_DATA/*.phn

  # Unpaired text input
  #text_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_again/
  #text_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone
  text_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone_remove_lower_frequecy_phones
  cat $text_dir/unpair_text/dict.txt > $TASK_DATA/dict.phn.txt
  TEXT_DATA=$text_dir/unpair_text
  ## in order to match unpair speech train set name
  #mv $TEXT_DATA/train.bin  $TEXT_DATA/train_m.bin
  #mv $TEXT_DATA/train.idx $TEXT_DATA/train_m.idx
  #KENLM_PATH=$text_dir/lm.phones.filtered.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)
  KENLM_PATH=$text_dir/lm.phones.filtered.04.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)
  config_dir=$fairseq_dir/examples/wav2vec/unsupervised

  dir=/mntcephfs/lab_data/maduo/exp/
  ##
  model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_1.25_mmi_weight0.5_code_penalty1_gradient_pennalty1.6_seed2_300k_steps_remove_lower_frequency_phones
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
       model.gradient_penalty=1.6\
       model.smoothness_weight=1.25\
       model.mmi_weight=0.5\
       common.seed=2\
       distributed_training.distributed_world_size=${world_size}\
       distributed_training.distributed_port=-1\
       distributed_training.ddp_backend=legacy_ddp\
       optimization.update_freq=[${update_freq}]\
       common.tensorboard_logdir=$exp_dir\
       hydra.run.dir=$exp_dir\
       hydra.job.name=$exp_dir/w2v_unsup_gan_xp

fi


## remove lower frequency phone, then get new phone dictionary, and remove contain lower frequecy phone utterance (unpair text and dev set (speech and text))
if [ ${stage} -le 60 ] && [ ${stop_stage} -ge 60 ];then
  fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
  des_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence

  ## it stores hubert large model 15 layer representation of raw librispeech speech,
  ## it  removes  silence. it offers feature of speech.
  TASK_DATA=$des_dir/hubert_large_feat_dir_no_silence
   ## offer mfcc pesudo label from without silence unpair audio
   #cp -r /mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir/mfcc_no_silence/mfcc_lab//* $TASK_DATA
   cp -r /mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir_remove_lower_freq_phone/*.km $TASK_DATA
   #cp -r $TASK_DATA/remove_phone_tone/* $TASK_DATA
   cp -r  /mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir_remove_lower_freq_phone/*.phn $TASK_DATA
   head $TASK_DATA/*.phn

  # Unpaired text input
  #text_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_again/
  #text_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone
  text_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone_remove_lower_frequecy_phones
  cat $text_dir/unpair_text/dict.txt > $TASK_DATA/dict.phn.txt
  TEXT_DATA=$text_dir/unpair_text
  ## in order to match unpair speech train set name
  #mv $TEXT_DATA/train.bin  $TEXT_DATA/train_m.bin
  #mv $TEXT_DATA/train.idx $TEXT_DATA/train_m.idx
  #KENLM_PATH=$text_dir/lm.phones.filtered.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)
  KENLM_PATH=$text_dir/lm.phones.filtered.04.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)
  config_dir=$fairseq_dir/examples/wav2vec/unsupervised

  dir=/mntcephfs/lab_data/maduo/exp/
  ##
  model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_1.25_mmi_weight0.5_code_penalty1_gradient_pennalty1.5_seed2_300k_steps_remove_lower_frequency_phones_batch_size800
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
       dataset.batch_size=800\
        dataset.num_workers=6\
       optimization.max_update=300000\
       common.user_dir=${fairseq_dir}/examples/wav2vec/unsupervised \
       model.code_penalty=1\
       model.gradient_penalty=1.5\
       model.smoothness_weight=1.25\
       model.mmi_weight=0.5\
       common.seed=2\
       distributed_training.distributed_world_size=${world_size}\
       distributed_training.distributed_port=-1\
       distributed_training.ddp_backend=legacy_ddp\
       optimization.update_freq=[${update_freq}]\
       common.tensorboard_logdir=$exp_dir\
       hydra.run.dir=$exp_dir\
       hydra.job.name=$exp_dir/w2v_unsup_gan_xp

fi



## remove lower frequency phone, then get new phone dictionary, and remove contain lower frequecy phone utterance (unpair text and dev set (speech and text))
if [ ${stage} -le 61 ] && [ ${stop_stage} -ge 61 ];then
  fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
  des_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence

  ## it stores hubert large model 15 layer representation of raw librispeech speech,
  ## it  removes  silence. it offers feature of speech.
  TASK_DATA=$des_dir/hubert_large_feat_dir_no_silence
   ## offer mfcc pesudo label from without silence unpair audio
   #cp -r /mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir/mfcc_no_silence/mfcc_lab//* $TASK_DATA
   cp -r /mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir_remove_lower_freq_phone/*.km $TASK_DATA
   #cp -r $TASK_DATA/remove_phone_tone/* $TASK_DATA
   cp -r  /mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir_remove_lower_freq_phone/*.phn $TASK_DATA
   head $TASK_DATA/*.phn

  # Unpaired text input
  #text_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_again/
  #text_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone
  text_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone_remove_lower_frequecy_phones
  cat $text_dir/unpair_text/dict.txt > $TASK_DATA/dict.phn.txt
  TEXT_DATA=$text_dir/unpair_text
  ## in order to match unpair speech train set name
  #mv $TEXT_DATA/train.bin  $TEXT_DATA/train_m.bin
  #mv $TEXT_DATA/train.idx $TEXT_DATA/train_m.idx
  #KENLM_PATH=$text_dir/lm.phones.filtered.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)
  KENLM_PATH=$text_dir/lm.phones.filtered.04.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)
  config_dir=$fairseq_dir/examples/wav2vec/unsupervised

  dir=/mntcephfs/lab_data/maduo/exp/
  ##
  model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_1.5_mmi_weight0.5_code_penalty0_gradient_pennalty1.5_seed1_300k_steps_remove_lower_frequency_phones_batch_size800
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
       dataset.batch_size=800\
        dataset.num_workers=6\
       optimization.max_update=300000\
       common.user_dir=${fairseq_dir}/examples/wav2vec/unsupervised \
       model.code_penalty=0\
       model.gradient_penalty=1.5\
       model.smoothness_weight=1.5\
       model.mmi_weight=0.5\
       common.seed=1\
       distributed_training.distributed_world_size=${world_size}\
       distributed_training.distributed_port=-1\
       distributed_training.ddp_backend=legacy_ddp\
       optimization.update_freq=[${update_freq}]\
       common.tensorboard_logdir=$exp_dir\
       hydra.run.dir=$exp_dir\
       hydra.job.name=$exp_dir/w2v_unsup_gan_xp

fi

## I remove lower frequency phone, then get new phone dictionary, and remove contain lower frequecy phone utterance (unpair text and dev set (speech and text))
## I use transcript of dev speech as unpair dev text.
if [ ${stage} -le 62 ] && [ ${stop_stage} -ge 62 ];then
  fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
  des_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence

  ## it stores hubert large model 15 layer representation of raw librispeech speech,
  ## it  removes  silence. it offers feature of speech.
  TASK_DATA=$des_dir/hubert_large_feat_dir_no_silence
   ## offer mfcc pesudo label from without silence unpair audio
   #cp -r /mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir/mfcc_no_silence/mfcc_lab//* $TASK_DATA
   #cp -r $TASK_DATA/remove_phone_tone/* $TASK_DATA
  cp -r /mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir_remove_lower_freq_phone/*.km $TASK_DATA
  cp -r  /mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir_remove_lower_freq_phone/*.phn $TASK_DATA
   head $TASK_DATA/*.phn

  # Unpaired text input
  #text_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_again/
  #text_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone
  text_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone_remove_lower_frequecy_phones
  cat $text_dir/unpair_text_with_dev/dict.txt > $TASK_DATA/dict.phn.txt
  TEXT_DATA=$text_dir/unpair_text_with_dev
  ## in order to match unpair speech train set name
  #mv $TEXT_DATA/train.bin  $TEXT_DATA/train_m.bin
  #mv $TEXT_DATA/train.idx $TEXT_DATA/train_m.idx
  #KENLM_PATH=$text_dir/lm.phones.filtered.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)
  KENLM_PATH=$text_dir/lm.phones.filtered.04.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)
  config_dir=$fairseq_dir/examples/wav2vec/unsupervised

  dir=/mntcephfs/lab_data/maduo/exp/
  model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_1.25_mmi_weight0.5_code_penalty1_gradient_pennalty1.5_seed2_300k_steps_remove_lower_frequency_phones_with_dev
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
       model.smoothness_weight=1.25\
       model.mmi_weight=0.5\
       common.seed=2\
       distributed_training.distributed_world_size=${world_size}\
       distributed_training.distributed_port=-1\
       distributed_training.ddp_backend=legacy_ddp\
       optimization.update_freq=[${update_freq}]\
       common.tensorboard_logdir=$exp_dir\
       hydra.run.dir=$exp_dir\
       hydra.job.name=$exp_dir/w2v_unsup_gan_xp

fi


## remove lower frequency phone, then get new phone dictionary, and remove contain lower frequecy phone utterance (unpair text and dev set (speech and text))
if [ ${stage} -le 63 ] && [ ${stop_stage} -ge 63 ];then
  fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
  des_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence

  ## it stores hubert large model 15 layer representation of raw librispeech speech,
  ## it  removes  silence. it offers feature of speech.
  TASK_DATA=$des_dir/hubert_large_feat_dir_no_silence
   ## offer mfcc pesudo label from without silence unpair audio
   #cp -r /mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir/mfcc_no_silence/mfcc_lab//* $TASK_DATA
   cp -r /mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir_remove_lower_freq_phone/*.km $TASK_DATA
   #cp -r $TASK_DATA/remove_phone_tone/* $TASK_DATA
   cp -r  /mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir_remove_lower_freq_phone/*.phn $TASK_DATA
   head $TASK_DATA/*.phn

  # Unpaired text input
  #text_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_again/
  #text_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone
  text_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone_remove_lower_frequecy_phones
  cat $text_dir/unpair_text/dict.txt > $TASK_DATA/dict.phn.txt
  TEXT_DATA=$text_dir/unpair_text
  ## in order to match unpair speech train set name
  #mv $TEXT_DATA/train.bin  $TEXT_DATA/train_m.bin
  #mv $TEXT_DATA/train.idx $TEXT_DATA/train_m.idx
  #KENLM_PATH=$text_dir/lm.phones.filtered.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)
  KENLM_PATH=$text_dir/lm.phones.filtered.04.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)
  config_dir=$fairseq_dir/examples/wav2vec/unsupervised

  dir=/mntcephfs/lab_data/maduo/exp/
  ##
  model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_0.5_mmi_weight0.5_code_penalty2_gradient_pennalty1.5_seed2_300k_steps_remove_lower_frequency_phones_batch_size800
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
       dataset.batch_size=800\
        dataset.num_workers=6\
       optimization.max_update=300000\
       common.user_dir=${fairseq_dir}/examples/wav2vec/unsupervised \
       model.code_penalty=2\
       model.gradient_penalty=1.5\
       model.smoothness_weight=0.5\
       model.mmi_weight=0.5\
       common.seed=2\
       distributed_training.distributed_world_size=${world_size}\
       distributed_training.distributed_port=-1\
       distributed_training.ddp_backend=legacy_ddp\
       optimization.update_freq=[${update_freq}]\
       common.tensorboard_logdir=$exp_dir\
       hydra.run.dir=$exp_dir\
       hydra.job.name=$exp_dir/w2v_unsup_gan_xp

fi

## remove lower frequency phone, then get new phone dictionary, and remove contain lower frequecy phone utterance (unpair text and dev set (speech and text))
if [ ${stage} -le 64 ] && [ ${stop_stage} -ge 64 ];then
  fairseq_dir=/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext
  des_dir=/mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence

  ## it stores hubert large model 15 layer representation of raw librispeech speech,
  ## it  removes  silence. it offers feature of speech.
  TASK_DATA=$des_dir/hubert_large_feat_dir_no_silence
   ## offer mfcc pesudo label from without silence unpair audio
   #cp -r /mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir/mfcc_no_silence/mfcc_lab//* $TASK_DATA
   cp -r /mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir_remove_lower_freq_phone/*.km $TASK_DATA
   #cp -r $TASK_DATA/remove_phone_tone/* $TASK_DATA
   cp -r  /mntcephfs/lab_data/maduo/datasets/wenetspeech_no_silence/tsv_dir_remove_lower_freq_phone/*.phn $TASK_DATA
   head $TASK_DATA/*.phn

  # Unpaired text input
  #text_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_again/
  #text_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone
  text_dir=/mntcephfs/lab_data/maduo/datasets/format/Chinese/aishell-2_norm_phn_seq_remove_tone_remove_lower_frequecy_phones
  cat $text_dir/unpair_text/dict.txt > $TASK_DATA/dict.phn.txt
  TEXT_DATA=$text_dir/unpair_text
  ## in order to match unpair speech train set name
  #mv $TEXT_DATA/train.bin  $TEXT_DATA/train_m.bin
  #mv $TEXT_DATA/train.idx $TEXT_DATA/train_m.idx
  #KENLM_PATH=$text_dir/lm.phones.filtered.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)
  KENLM_PATH=$text_dir/lm.phones.filtered.04.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)
  config_dir=$fairseq_dir/examples/wav2vec/unsupervised

  dir=/mntcephfs/lab_data/maduo/exp/
  ##
  model_name=w2v_unsup_gan_xp_1gpu_1update_with_unpair_text_1M_remove_tone_smoothness_weight_0.5_mmi_weight0.5_code_penalty1_gradient_pennalty1.5_seed2_300k_steps_remove_lower_frequency_phones_batch_size800
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
       dataset.batch_size=800\
        dataset.num_workers=6\
       optimization.max_update=300000\
       common.user_dir=${fairseq_dir}/examples/wav2vec/unsupervised \
       model.code_penalty=1\
       model.gradient_penalty=1.5\
       model.smoothness_weight=0.5\
       model.mmi_weight=0.5\
       common.seed=2\
       distributed_training.distributed_world_size=${world_size}\
       distributed_training.distributed_port=-1\
       distributed_training.ddp_backend=legacy_ddp\
       optimization.update_freq=[${update_freq}]\
       common.tensorboard_logdir=$exp_dir\
       hydra.run.dir=$exp_dir\
       hydra.job.name=$exp_dir/w2v_unsup_gan_xp

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
