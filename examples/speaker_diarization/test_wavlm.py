#!/usr/bin/env python3
# Author: Duo MA
# Email: maduo@cuhk.edu.cn

import torch
from examples.speaker_diarization.ts_vad.models.modules.WavLM import WavLM, WavLMConfig

# load the pre-trained checkpoints
#checkpoint = torch.load('/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Base+.pt')
checkpoint = torch.load('/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Large.pt')
cfg = WavLMConfig(checkpoint['cfg'])
model = WavLM(cfg)
model.load_state_dict(checkpoint['model'])
model.eval()

# extract the the representation of last layer
wav_input_16khz = torch.randn(1,10000)
rep = model.extract_features(wav_input_16khz)[0]
print(f"rep shape: {rep.shape}")
# extract the the representation of each layer
wav_input_16khz = torch.randn(1,10000)
rep, layer_results = model.extract_features(wav_input_16khz, output_layer=model.cfg.encoder_layers, ret_layer_results=True)[0]
layer_reps = [x.transpose(0, 1) for x, _ in layer_results]
print(f"model.cfg.encoder_layers: {model.cfg.encoder_layers}")
print(f"layer_reps len: {len(layer_reps)}, its shape: {layer_reps[0].shape}") #(b,t,f)
