#!/usr/bin/env bash

stage=0
stop_stage=1000
nj=32
. utils/parse_options.sh
#. path_for_fairseq_speechtext.sh
#. path_for_fsq_speechtext.sh
. path_for_fsq_sptt.sh # with pytorch_lightning=2.1.2
#. path.sh # with kaldi env and fsq_sptt
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1


if [ ${stage} -le 1 ]&&[ ${stop_stage} -ge 1 ];then

    echo "Start infer"
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    train_conf=$root_dir/fs_eend/config/spk_onl_tfm_enc_dec_nonautoreg_simple.yaml
    ## (TODO) modify data dir at config
    train_dir=$root_dir/data/simu/data/train_clean_5_ns2_beta2_500
    dev_dir=$root_dir/data/simu/data/dev_clean_2_ns2_beta2_500
    model_dir=$root_dir/exp/fs_eend_base
    infer_model=$model_dir/ave_model.pt
    #model_dir=$root_dir/exp/fs_eend_base/
    mkdir -p $model_dir
    python fs_eend/train_pl.py \
            --configs $train_conf \
            --gpus 1\
            --exp_dir $model_dir\
            --inference True\
	    --save_avg_path $infer_model


fi
## result:
#Start infer
#[rank: 0] Seed set to 777
#GPU available: True (cuda), used: True
#TPU available: False, using: 0 TPU cores
#IPU available: False, using: 0 IPUs
#HPU available: False, using: 0 HPUs
#[rank: 0] Seed set to 777
#Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/1
#----------------------------------------------------------------------------------------------------
#distributed_backend=nccl
#All distributed processes registered. Starting with 1 processes
#----------------------------------------------------------------------------------------------------
#
#LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
#configs: {'log': {'model_name': None, 'log_dir': './logs/None', 'save_top_k': -1, 'start_epoch': 90, 'end_epoch': 99}, 'training': {'batch_size': 32, 'n_workers': 8, 'shuffle': True, 'lr': 1, 'opt': 'noam', 'max_epochs': 100, 'grad_clip': 5, 'grad_accm': 1, 'scheduler': 'noam', 'schedule_scale': 1.0, 'warm_steps': 100000, 'early_stop_epoch': 100, 'init_ckpt': None, 'val_interval': 1, 'seed': 777}, 'model': {'arch': 'onlineTransformerDA_emb1dcnn_linear_nonautoreg_l2norm', 'params': {'n_units': 256, 'n_heads': 4, 'enc_n_layers': 4, 'dec_dim_feedforward': 2048, 'dropout': 0.1, 'has_mask': True, 'max_seqlen': 500, 'mask_delay': 0, 'dec_n_layers': 2}}, 'data': {'num_speakers': 2, 'max_speakers': 2, 'context_recp': 7, 'label_delay': 0, 'feat_type': 'logmel23', 'chunk_size': 500, 'subsampling': 10, 'use_last_samples': True, 'shuffle': False, 'augment': None, 'feat': {'sample_rate': 8000, 'win_length': 200, 'n_fft': 1024, 'hop_length': 80, 'n_mels': 23, 'f_max': 4000, 'power': 1}, 'scaler': {'statistic': 'instance', 'normtype': 'minmax', 'dims': [1, 2]}, 'train_data_dir': 'data/simu/data/train_clean_5_ns2_beta2_500', 'val_data_dir': 'data/simu/data/dev_clean_2_ns2_beta2_500', 'test_data_dir': 'data/simu/data/dev_clean_2_ns2_beta2_500'}, 'task': {'max_speakers': None, 'spk_attractor': {'enable': True, 'shuffle': True, 'enc_dropout': 0.5, 'dec_dropout': 0.5, 'consis_weight': 1}}, 'debug': {'num_sanity_val_steps': 3, 'log_every_n_steps': 100}}
#2730  chunks
#1863  chunks
#1863  chunks
#Using noam scheduler
#Noam initializing...
#Experiment dir: /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/fs_eend_base
#Testing DataLoader 0: 100%|██████████| 1863/1863 [00:29<00:00, 62.66it/s]{'test/diarization_error': 154.02683843263554,
# 'test/frames': 431.9151905528717,
# 'test/preliminary_DER': 0.272574771122379,
# 'test/speaker_error': 20.673107890499196,
# 'test/speaker_error_rate': 0.03658432328405902,
# 'test/speaker_falarm': 103.1653247450349,
# 'test/speaker_falarm_rate': 0.18256730493395368,
# 'test/speaker_miss': 30.18840579710145,
# 'test/speaker_miss_rate': 0.0534231429043663,
# 'test/speaker_scored': 565.0810520665593,
# 'test/speech_falarm': 3.3311862587224907,
# 'test/speech_miss': 13.050993022007514,
# 'test/speech_scored': 389.8679549114332}
#Testing DataLoader 0: 100%|██████████| 1863/1863 [00:29<00:00, 62.51it/s]



