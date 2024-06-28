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


if [ ${stage} -le 0 ]&&[ ${stop_stage} -ge 0 ];then
 
    echo "Start training"
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    train_conf=$root_dir/fs_eend/config/spk_onl_tfm_enc_dec_nonautoreg.yaml
    ## (TODO) modify data dir at config
    train_dir=$root_dir/data/simu/data/train_clean_5_ns2_beta2_500
    dev_dir=$root_dir/data/simu/data/dev_clean_2_ns2_beta2_500
    model_dir=$root_dir/exp/fs_eend_base
    #model_dir=$root_dir/exp/fs_eend_base/
    mkdir -p $model_dir
    python fs_eend/train.py --configs $train_conf  --gpus 1
    ## train logging: 
#    Start training
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
#You are using a CUDA device ('NVIDIA GeForce RTX 3090') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
#LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
#
#  | Name  | Type                           | Params
#---------------------------------------------------------
#0 | model | OnlineTransformerDADiarization | 8.4 M
#---------------------------------------------------------
#8.4 M     Trainable params
#0         Non-trainable params
#8.4 M     Total params
#33.581    Total estimated model params size (MB)
#configs: {'log': {'model_name': 'spk_onl_tfm_enc_dec_10w', 'log_dir': './logs/spk_onl_tfm_enc_dec_10w', 'save_top_k': -1, 'start_epoch': 90, 'end_epoch': 99, 'save_avg_path': None}, 'training': {'batch_size': 32, 'n_workers': 8, 'shuffle': True, 'lr': 1, 'opt': 'noam', 'max_epochs': 100, 'grad_clip': 5, 'grad_accm': 1, 'scheduler': 'noam', 'schedule_scale': 1.0, 'warm_steps': 100000, 'early_stop_epoch': 100, 'init_ckpt': None, 'dist_strategy': 'ddp', 'val_interval': 1, 'seed': 777}, 'model': {'arch': 'onlineTransformerDA_emb1dcnn_linear_nonautoreg_l2norm', 'params': {'n_units': 256, 'n_heads': 4, 'enc_n_layers': 4, 'dec_dim_feedforward': 2048, 'dropout': 0.1, 'has_mask': True, 'max_seqlen': 500, 'mask_delay': 0, 'dec_n_layers': 2}}, 'data': {'num_speakers': None, 'max_speakers': 5, 'context_recp': 7, 'label_delay': 0, 'feat_type': 'logmel23', 'chunk_size': 500, 'subsampling': 10, 'use_last_samples': True, 'shuffle': False, 'augment': None, 'feat': {'sample_rate': 8000, 'win_length': 200, 'n_fft': 1024, 'hop_length': 80, 'n_mels': 23, 'f_max': 4000, 'power': 1}, 'scaler': {'statistic': 'instance', 'normtype': 'minmax', 'dims': [1, 2]}, 'train_data_dir': 'data/simu/data/train_clean_5_ns2_beta2_500', 'val_data_dir': 'data/simu/data/dev_clean_2_ns2_beta2_500'}, 'task': {'max_speakers': None, 'spk_attractor': {'enable': True, 'shuffle': True, 'enc_dropout': 0.5, 'dec_dropout': 0.5, 'consis_weight': 1}}, 'debug': {'num_sanity_val_steps': 3, 'log_every_n_steps': 100}}
#2730  chunks
#1863  chunks
#Using noam scheduler
#Noam initializing...
#Experiment dir: ./logs/spk_onl_tfm_enc_dec_10w/version_8
#Epoch 0: 100%|██████████| 86/86 [01:04<00:00,  1.33it/s, v_num=8, train/lr=1.68e-7]Metric val/obj_metric improved. New best score: 0.789
#Epoch 8: 100%|██████████| 86/86 [00:51<00:00,  1.68it/s, v_num=8, train/lr=1.53e-6]Metric val/obj_metric improved by 0.265 >= min_delta = 0.0. New best score: 0.524
#Epoch 9: 100%|██████████| 86/86 [00:52<00:00,  1.65it/s, v_num=8, train/lr=1.7e-6]Metric val/obj_metric improved by 0.148 >= min_delta = 0.0. New best score: 0.376
#Epoch 15: 100%|██████████| 86/86 [00:51<00:00,  1.66it/s, v_num=8, train/lr=2.72e-6]Metric val/obj_metric improved by 0.022 >= min_delta = 0.0. New best score: 0.354
#Epoch 16: 100%|██████████| 86/86 [00:50<00:00,  1.71it/s, v_num=8, train/lr=2.89e-6]Metric val/obj_metric improved by 0.006 >= min_delta = 0.0. New best score: 0.348
#Epoch 17: 100%|██████████| 86/86 [00:51<00:00,  1.67it/s, v_num=8, train/lr=3.06e-6]Metric val/obj_metric improved by 0.015 >= min_delta = 0.0. New best score: 0.333
#Epoch 18: 100%|██████████| 86/86 [00:51<00:00,  1.67it/s, v_num=8, train/lr=3.23e-6]Metric val/obj_metric improved by 0.001 >= min_delta = 0.0. New best score: 0.332
#Epoch 19: 100%|██████████| 86/86 [00:48<00:00,  1.76it/s, v_num=8, train/lr=3.4e-6]Metric val/obj_metric improved by 0.017 >= min_delta = 0.0. New best score: 0.314
#Epoch 20: 100%|██████████| 86/86 [00:50<00:00,  1.69it/s, v_num=8, train/lr=3.57e-6]Metric val/obj_metric improved by 0.002 >= min_delta = 0.0. New best score: 0.312
#Epoch 21: 100%|██████████| 86/86 [00:50<00:00,  1.69it/s, v_num=8, train/lr=3.74e-6]Metric val/obj_metric improved by 0.010 >= min_delta = 0.0. New best score: 0.302
#Epoch 23: 100%|██████████| 86/86 [00:50<00:00,  1.69it/s, v_num=8, train/lr=4.08e-6]Metric val/obj_metric improved by 0.004 >= min_delta = 0.0. New best score: 0.298
#Epoch 24: 100%|██████████| 86/86 [01:25<00:00,  1.01it/s, v_num=8, train/lr=4.25e-6]Metric val/obj_metric improved by 0.003 >= min_delta = 0.0. New best score: 0.295
#Epoch 27: 100%|██████████| 86/86 [00:49<00:00,  1.74it/s, v_num=8, train/lr=4.76e-6]Metric val/obj_metric improved by 0.002 >= min_delta = 0.0. New best score: 0.293
#Epoch 28: 100%|██████████| 86/86 [00:51<00:00,  1.68it/s, v_num=8, train/lr=4.93e-6]Metric val/obj_metric improved by 0.001 >= min_delta = 0.0. New best score: 0.292
#Epoch 29: 100%|██████████| 86/86 [00:51<00:00,  1.68it/s, v_num=8, train/lr=5.1e-6]Metric val/obj_metric improved by 0.010 >= min_delta = 0.0. New best score: 0.282
#Epoch 32: 100%|██████████| 86/86 [00:50<00:00,  1.71it/s, v_num=8, train/lr=5.61e-6]Metric val/obj_metric improved by 0.004 >= min_delta = 0.0. New best score: 0.278
#Epoch 33: 100%|██████████| 86/86 [00:49<00:00,  1.73it/s, v_num=8, train/lr=5.78e-6]Metric val/obj_metric improved by 0.001 >= min_delta = 0.0. New best score: 0.277
#Epoch 70: 100%|██████████| 86/86 [00:51<00:00,  1.67it/s, v_num=8, train/lr=1.21e-5]Metric val/obj_metric improved by 0.003 >= min_delta = 0.0. New best score: 0.274
#Epoch 73: 100%|██████████| 86/86 [00:51<00:00,  1.66it/s, v_num=8, train/lr=1.26e-5]Metric val/obj_metric improved by 0.001 >= min_delta = 0.0. New best score: 0.274
#Epoch 83: 100%|██████████| 86/86 [00:52<00:00,  1.65it/s, v_num=8, train/lr=1.43e-5]Metric val/obj_metric improved by 0.002 >= min_delta = 0.0. New best score: 0.272
#Epoch 96: 100%|██████████| 86/86 [01:38<00:00,  0.88it/s, v_num=8, train/lr=1.65e-5]Metric val/obj_metric improved by 0.001 >= min_delta = 0.0. New best score: 0.271
#Epoch 99: 100%|██████████| 86/86 [00:50<00:00,  1.71it/s, v_num=8, train/lr=1.7e-5]`Trainer.fit` stopped: `max_epochs=100` reached.
#Epoch 99: 100%|██████████| 86/86 [00:54<00:00,  1.57it/s, v_num=8, train/lr=1.7e-5][rank: 0] Seed set to 777
#LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
#
#Best model path: ./logs/spk_onl_tfm_enc_dec_10w/version_8/epoch=96-step=8342.ckpt
#Test using ckpts:
#./logs/spk_onl_tfm_enc_dec_10w/version_8/epoch=90-step=7826.ckpt
#./logs/spk_onl_tfm_enc_dec_10w/version_8/epoch=91-step=7912.ckpt
#./logs/spk_onl_tfm_enc_dec_10w/version_8/epoch=92-step=7998.ckpt
#./logs/spk_onl_tfm_enc_dec_10w/version_8/epoch=93-step=8084.ckpt
#./logs/spk_onl_tfm_enc_dec_10w/version_8/epoch=94-step=8170.ckpt
#./logs/spk_onl_tfm_enc_dec_10w/version_8/epoch=95-step=8256.ckpt
#./logs/spk_onl_tfm_enc_dec_10w/version_8/epoch=96-step=8342.ckpt
#./logs/spk_onl_tfm_enc_dec_10w/version_8/epoch=97-step=8428.ckpt
#./logs/spk_onl_tfm_enc_dec_10w/version_8/epoch=98-step=8514.ckpt
#./logs/spk_onl_tfm_enc_dec_10w/version_8/epoch=99-step=8600.ckpt
#Testing DataLoader 0: 100%|██████████| 59/59 [00:10<00:00,  5.49it/s]{'test/diarization_error': 158.3166935050993,
# 'test/frames': 431.9151905528717,
# 'test/preliminary_DER': 0.28016634591819867,
# 'test/speaker_error': 16.831454643048847,
# 'test/speaker_error_rate': 0.029785912271336108,
# 'test/speaker_falarm': 120.1905528717123,
# 'test/speaker_falarm_rate': 0.21269612993067655,
# 'test/speaker_miss': 21.294685990338163,
# 'test/speaker_miss_rate': 0.03768430371618605,
# 'test/speaker_scored': 565.0810520665593,
# 'test/speech_falarm': 4.760601180891036,
# 'test/speech_miss': 8.31508319914117,
# 'test/speech_scored': 389.8679549114332}
#Testing DataLoader 0: 100%|██████████| 59/59 [00:11<00:00,  5.34it/s]
fi


if [ ${stage} -le 1 ]&&[ ${stop_stage} -ge 1 ];then

    echo "Start training"
    root_dir=/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization
    train_conf=$root_dir/fs_eend/config/spk_onl_tfm_enc_dec_nonautoreg_simple.yaml
    ## (TODO) modify data dir at config
    train_dir=$root_dir/data/simu/data/train_clean_5_ns2_beta2_500
    dev_dir=$root_dir/data/simu/data/dev_clean_2_ns2_beta2_500
    model_dir=$root_dir/exp/fs_eend_base
    resume_pt=""
    #model_dir=$root_dir/exp/fs_eend_base/
    mkdir -p $model_dir
    python fs_eend/train_pl.py \
	    --configs $train_conf \
	    --gpus 1\
	    --exp_dir $model_dir

fi
    # Start training
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
#
#  | Name  | Type                           | Params
#---------------------------------------------------------
#0 | model | OnlineTransformerDADiarization | 8.4 M
#---------------------------------------------------------
#8.4 M     Trainable params
#0         Non-trainable params
#8.4 M     Total params
#33.581    Total estimated model params size (MB)
#configs: {'log': {'model_name': None, 'log_dir': './logs/None', 'save_top_k': -1, 'start_epoch': 90, 'end_epoch': 99}, 'training': {'batch_size': 32, 'n_workers': 8, 'shuffle': True, 'lr': 1, 'opt': 'noam', 'max_epochs': 100, 'grad_clip': 5, 'grad_accm': 1, 'scheduler': 'noam', 'schedule_scale': 1.0, 'warm_steps': 100000, 'early_stop_epoch': 100, 'init_ckpt': None, 'val_interval': 1, 'seed': 777}, 'model': {'arch': 'onlineTransformerDA_emb1dcnn_linear_nonautoreg_l2norm', 'params': {'n_units': 256, 'n_heads': 4, 'enc_n_layers': 4, 'dec_dim_feedforward': 2048, 'dropout': 0.1, 'has_mask': True, 'max_seqlen': 500, 'mask_delay': 0, 'dec_n_layers': 2}}, 'data': {'num_speakers': 2, 'max_speakers': 2, 'context_recp': 7, 'label_delay': 0, 'feat_type': 'logmel23', 'chunk_size': 500, 'subsampling': 10, 'use_last_samples': True, 'shuffle': False, 'augment': None, 'feat': {'sample_rate': 8000, 'win_length': 200, 'n_fft': 1024, 'hop_length': 80, 'n_mels': 23, 'f_max': 4000, 'power': 1}, 'scaler': {'statistic': 'instance', 'normtype': 'minmax', 'dims': [1, 2]}, 'train_data_dir': 'data/simu/data/train_clean_5_ns2_beta2_500', 'val_data_dir': 'data/simu/data/dev_clean_2_ns2_beta2_500', 'test_data_dir': 'data/simu/data/dev_clean_2_ns2_beta2_500'}, 'task': {'max_speakers': None, 'spk_attractor': {'enable': True, 'shuffle': True, 'enc_dropout': 0.5, 'dec_dropout': 0.5, 'consis_weight': 1}}, 'debug': {'num_sanity_val_steps': 3, 'log_every_n_steps': 100}}
#2730  chunks
#1863  chunks
#1863  chunks
#Using noam scheduler
#Noam initializing...
#Experiment dir: /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/fs_eend_base
#Epoch 0: 100%|██████████| 86/86 [00:44<00:00,  1.92it/s, v_num=0, train/lr=1.68e-7]Metric val/obj_metric improved. New best score: 0.789
#Epoch 8: 100%|██████████| 86/86 [00:45<00:00,  1.87it/s, v_num=0, train/lr=1.53e-6]Metric val/obj_metric improved by 0.265 >= min_delta = 0.0. New best score: 0.524
#Epoch 9: 100%|██████████| 86/86 [00:51<00:00,  1.66it/s, v_num=0, train/lr=1.7e-6]Metric val/obj_metric improved by 0.148 >= min_delta = 0.0. New best score: 0.376
#Epoch 15: 100%|██████████| 86/86 [00:45<00:00,  1.89it/s, v_num=0, train/lr=2.72e-6]Metric val/obj_metric improved by 0.022 >= min_delta = 0.0. New best score: 0.354
#Epoch 16: 100%|██████████| 86/86 [00:46<00:00,  1.84it/s, v_num=0, train/lr=2.89e-6]Metric val/obj_metric improved by 0.006 >= min_delta = 0.0. New best score: 0.348
#Epoch 17: 100%|██████████| 86/86 [00:45<00:00,  1.89it/s, v_num=0, train/lr=3.06e-6]Metric val/obj_metric improved by 0.015 >= min_delta = 0.0. New best score: 0.333
#Epoch 18: 100%|██████████| 86/86 [00:45<00:00,  1.88it/s, v_num=0, train/lr=3.23e-6]Metric val/obj_metric improved by 0.001 >= min_delta = 0.0. New best score: 0.332
#Epoch 19: 100%|██████████| 86/86 [00:46<00:00,  1.87it/s, v_num=0, train/lr=3.4e-6]Metric val/obj_metric improved by 0.017 >= min_delta = 0.0. New best score: 0.314
#Epoch 20: 100%|██████████| 86/86 [00:46<00:00,  1.86it/s, v_num=0, train/lr=3.57e-6]Metric val/obj_metric improved by 0.002 >= min_delta = 0.0. New best score: 0.312
#Epoch 21: 100%|██████████| 86/86 [00:45<00:00,  1.87it/s, v_num=0, train/lr=3.74e-6]Metric val/obj_metric improved by 0.010 >= min_delta = 0.0. New best score: 0.302
#Epoch 23: 100%|██████████| 86/86 [00:46<00:00,  1.87it/s, v_num=0, train/lr=4.08e-6]Metric val/obj_metric improved by 0.004 >= min_delta = 0.0. New best score: 0.298
#Epoch 24: 100%|██████████| 86/86 [00:45<00:00,  1.88it/s, v_num=0, train/lr=4.25e-6]Metric val/obj_metric improved by 0.003 >= min_delta = 0.0. New best score: 0.295
#Epoch 27: 100%|██████████| 86/86 [00:45<00:00,  1.88it/s, v_num=0, train/lr=4.76e-6]Metric val/obj_metric improved by 0.002 >= min_delta = 0.0. New best score: 0.293
#Epoch 28: 100%|██████████| 86/86 [00:46<00:00,  1.84it/s, v_num=0, train/lr=4.93e-6]Metric val/obj_metric improved by 0.001 >= min_delta = 0.0. New best score: 0.292
#Epoch 29: 100%|██████████| 86/86 [00:45<00:00,  1.87it/s, v_num=0, train/lr=5.1e-6]Metric val/obj_metric improved by 0.010 >= min_delta = 0.0. New best score: 0.282
#Epoch 32: 100%|██████████| 86/86 [00:46<00:00,  1.86it/s, v_num=0, train/lr=5.61e-6]Metric val/obj_metric improved by 0.004 >= min_delta = 0.0. New best score: 0.278
#Epoch 33: 100%|██████████| 86/86 [00:45<00:00,  1.90it/s, v_num=0, train/lr=5.78e-6]Metric val/obj_metric improved by 0.001 >= min_delta = 0.0. New best score: 0.277
#Epoch 70: 100%|██████████| 86/86 [00:47<00:00,  1.82it/s, v_num=0, train/lr=1.21e-5]Metric val/obj_metric improved by 0.003 >= min_delta = 0.0. New best score: 0.274
#Epoch 73: 100%|██████████| 86/86 [00:47<00:00,  1.82it/s, v_num=0, train/lr=1.26e-5]Metric val/obj_metric improved by 0.001 >= min_delta = 0.0. New best score: 0.274
#Epoch 83: 100%|██████████| 86/86 [00:47<00:00,  1.81it/s, v_num=0, train/lr=1.43e-5]Metric val/obj_metric improved by 0.002 >= min_delta = 0.0. New best score: 0.271
#Epoch 99: 100%|██████████| 86/86 [00:47<00:00,  1.82it/s, v_num=0, train/lr=1.7e-5]`Trainer.fit` stopped: `max_epochs=100` reached.
#Epoch 99: 100%|██████████| 86/86 [00:49<00:00,  1.73it/s, v_num=0, train/lr=1.7e-5][rank: 0] Seed set to 777
#LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
#
#Best model path: /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/fs_eend_base/version_0/epoch=83-step=7224.ckpt
#dirctory of all ckpt : /home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/fs_eend_base/version_0
#ckpts: ['epoch=90-step=7826.ckpt', 'epoch=91-step=7912.ckpt', 'epoch=92-step=7998.ckpt', 'epoch=93-step=8084.ckpt', 'epoch=94-step=8170.ckpt', 'epoch=95-step=8256.ckpt', 'epoch=96-step=8342.ckpt', 'epoch=97-step=8428.ckpt', 'epoch=98-step=8514.ckpt', 'epoch=99-step=8600.ckpt']
#Test using ckpts:
#/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/fs_eend_base/version_0/epoch=90-step=7826.ckpt
#/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/fs_eend_base/version_0/epoch=91-step=7912.ckpt
#/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/fs_eend_base/version_0/epoch=92-step=7998.ckpt
#/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/fs_eend_base/version_0/epoch=93-step=8084.ckpt
#/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/fs_eend_base/version_0/epoch=94-step=8170.ckpt
#/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/fs_eend_base/version_0/epoch=95-step=8256.ckpt
#/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/fs_eend_base/version_0/epoch=96-step=8342.ckpt
#/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/fs_eend_base/version_0/epoch=97-step=8428.ckpt
#/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/fs_eend_base/version_0/epoch=98-step=8514.ckpt
#/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/exp/fs_eend_base/version_0/epoch=99-step=8600.ckpt
####test_state: dict_keys(['model.enc.bn.weight', 'model.enc.bn.bias', 'model.enc.bn.running_mean', 'model.enc.bn.running_var', 'model.enc.bn.num_batches_tracked', 'model.enc.encoder.weight', 'model.enc.encoder.bias', 'model.enc.encoder_norm.weight', 'model.enc.encoder_norm.bias', 'model.enc.transformer_encoder.layers.0.self_attn.in_proj_weight', 'model.enc.transformer_encoder.layers.0.self_attn.in_proj_bias', 'model.enc.transformer_encoder.layers.0.self_attn.out_proj.weight', 'model.enc.transformer_encoder.layers.0.self_attn.out_proj.bias', 'model.enc.transformer_encoder.layers.0.linear1.weight', 'model.enc.transformer_encoder.layers.0.linear1.bias', 'model.enc.transformer_encoder.layers.0.linear2.weight', 'model.enc.transformer_encoder.layers.0.linear2.bias', 'model.enc.transformer_encoder.layers.0.norm1.weight', 'model.enc.transformer_encoder.layers.0.norm1.bias', 'model.enc.transformer_encoder.layers.0.norm2.weight', 'model.enc.transformer_encoder.layers.0.norm2.bias', 'model.enc.transformer_encoder.layers.1.self_attn.in_proj_weight', 'model.enc.transformer_encoder.layers.1.self_attn.in_proj_bias', 'model.enc.transformer_encoder.layers.1.self_attn.out_proj.weight', 'model.enc.transformer_encoder.layers.1.self_attn.out_proj.bias', 'model.enc.transformer_encoder.layers.1.linear1.weight', 'model.enc.transformer_encoder.layers.1.linear1.bias', 'model.enc.transformer_encoder.layers.1.linear2.weight', 'model.enc.transformer_encoder.layers.1.linear2.bias', 'model.enc.transformer_encoder.layers.1.norm1.weight', 'model.enc.transformer_encoder.layers.1.norm1.bias', 'model.enc.transformer_encoder.layers.1.norm2.weight', 'model.enc.transformer_encoder.layers.1.norm2.bias', 'model.enc.transformer_encoder.layers.2.self_attn.in_proj_weight', 'model.enc.transformer_encoder.layers.2.self_attn.in_proj_bias', 'model.enc.transformer_encoder.layers.2.self_attn.out_proj.weight', 'model.enc.transformer_encoder.layers.2.self_attn.out_proj.bias', 'model.enc.transformer_encoder.layers.2.linear1.weight', 'model.enc.transformer_encoder.layers.2.linear1.bias', 'model.enc.transformer_encoder.layers.2.linear2.weight', 'model.enc.transformer_encoder.layers.2.linear2.bias', 'model.enc.transformer_encoder.layers.2.norm1.weight', 'model.enc.transformer_encoder.layers.2.norm1.bias', 'model.enc.transformer_encoder.layers.2.norm2.weight', 'model.enc.transformer_encoder.layers.2.norm2.bias', 'model.enc.transformer_encoder.layers.3.self_attn.in_proj_weight', 'model.enc.transformer_encoder.layers.3.self_attn.in_proj_bias', 'model.enc.transformer_encoder.layers.3.self_attn.out_proj.weight', 'model.enc.transformer_encoder.layers.3.self_attn.out_proj.bias', 'model.enc.transformer_encoder.layers.3.linear1.weight', 'model.enc.transformer_encoder.layers.3.linear1.bias', 'model.enc.transformer_encoder.layers.3.linear2.weight', 'model.enc.transformer_encoder.layers.3.linear2.bias', 'model.enc.transformer_encoder.layers.3.norm1.weight', 'model.enc.transformer_encoder.layers.3.norm1.bias', 'model.enc.transformer_encoder.layers.3.norm2.weight', 'model.enc.transformer_encoder.layers.3.norm2.bias', 'model.dec.encoder.weight', 'model.dec.encoder.bias', 'model.dec.encoder_norm.weight', 'model.dec.encoder_norm.bias', 'model.dec.pos_enc.pe', 'model.dec.convert.weight', 'model.dec.convert.bias', 'model.dec.attractor_decoder.0.self_attn1.in_proj_weight', 'model.dec.attractor_decoder.0.self_attn1.in_proj_bias', 'model.dec.attractor_decoder.0.self_attn1.out_proj.weight', 'model.dec.attractor_decoder.0.self_attn1.out_proj.bias', 'model.dec.attractor_decoder.0.self_attn2.in_proj_weight', 'model.dec.attractor_decoder.0.self_attn2.in_proj_bias', 'model.dec.attractor_decoder.0.self_attn2.out_proj.weight', 'model.dec.attractor_decoder.0.self_attn2.out_proj.bias', 'model.dec.attractor_decoder.0.linear1.weight', 'model.dec.attractor_decoder.0.linear1.bias', 'model.dec.attractor_decoder.0.linear2.weight', 'model.dec.attractor_decoder.0.linear2.bias', 'model.dec.attractor_decoder.0.norm11.weight', 'model.dec.attractor_decoder.0.norm11.bias', 'model.dec.attractor_decoder.0.norm12.weight', 'model.dec.attractor_decoder.0.norm12.bias', 'model.dec.attractor_decoder.0.norm21.weight', 'model.dec.attractor_decoder.0.norm21.bias', 'model.dec.attractor_decoder.0.norm22.weight', 'model.dec.attractor_decoder.0.norm22.bias', 'model.dec.attractor_decoder.1.self_attn1.in_proj_weight', 'model.dec.attractor_decoder.1.self_attn1.in_proj_bias', 'model.dec.attractor_decoder.1.self_attn1.out_proj.weight', 'model.dec.attractor_decoder.1.self_attn1.out_proj.bias', 'model.dec.attractor_decoder.1.self_attn2.in_proj_weight', 'model.dec.attractor_decoder.1.self_attn2.in_proj_bias', 'model.dec.attractor_decoder.1.self_attn2.out_proj.weight', 'model.dec.attractor_decoder.1.self_attn2.out_proj.bias', 'model.dec.attractor_decoder.1.linear1.weight', 'model.dec.attractor_decoder.1.linear1.bias', 'model.dec.attractor_decoder.1.linear2.weight', 'model.dec.attractor_decoder.1.linear2.bias', 'model.dec.attractor_decoder.1.norm11.weight', 'model.dec.attractor_decoder.1.norm11.bias', 'model.dec.attractor_decoder.1.norm12.weight', 'model.dec.attractor_decoder.1.norm12.bias', 'model.dec.attractor_decoder.1.norm21.weight', 'model.dec.attractor_decoder.1.norm21.bias', 'model.dec.attractor_decoder.1.norm22.weight', 'model.dec.attractor_decoder.1.norm22.bias', 'model.cnn.weight', 'model.cnn.bias'])

#Testing DataLoader 0: 100%|██████████| 1863/1863 [00:28<00:00, 65.49it/s]{'test/diarization_error': 154.02683843263554,
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
#Testing DataLoader 0: 100%|██████████| 1863/1863 [00:28<00:00, 65.27it/s]

