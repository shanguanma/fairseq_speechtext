#!/usr/bin/env python3
# it is from https://github.com/microsoft/UniSpeech/blob/main/downstreams/speaker_verification/models/utils.py
import torch
import fairseq
from packaging import version
import torch.nn.functional as F
from fairseq import tasks
from fairseq.checkpoint_utils import load_checkpoint_to_cpu
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from omegaconf import OmegaConf
from s3prl.upstream.interfaces import UpstreamBase
from torch.nn.utils.rnn import pad_sequence
from examples.speaker_diarization.ts_vad.models.modules.WavLM import WavLM, WavLMConfig

def load_model(filepath):
    state = torch.load(filepath, map_location=lambda storage, loc: storage)
    # state = load_checkpoint_to_cpu(filepath)
    state["cfg"] = OmegaConf.create(state["cfg"])

    if "args" in state and state["args"] is not None:
        cfg = convert_namespace_to_omegaconf(state["args"])
    elif "cfg" in state and state["cfg"] is not None:
        cfg = state["cfg"]
    else:
        raise RuntimeError(
            f"Neither args nor cfg exist in state keys = {state.keys()}"
            )

    task = tasks.setup_task(cfg.task)
    if "task_state" in state:
        task.load_state_dict(state["task_state"])

    model = task.build_model(cfg.model)

    return model, cfg, task


###################
# UPSTREAM EXPERT #
###################
class UpstreamExpert(UpstreamBase):
    def __init__(self, ckpt, **kwargs):
        super().__init__(**kwargs)
        assert version.parse(fairseq.__version__) > version.parse(
            "0.10.2"
        ), "Please install the fairseq master branch."
        checkpoint = torch.load(ckpt, map_location=torch.device("cpu"))
        cfg = WavLMConfig(checkpoint['cfg'])
        feature_extract = WavLM(cfg) # instance WavLM model
        self.wavlm_encoder_layers = checkpoint['cfg']["encoder_layers"]
        self.model = feature_extract
        self.cfg_normalize=checkpoint['cfg']["normalize"]
        self.wavlm_encoder_layers = checkpoint['cfg']["encoder_layers"]
        #model, cfg, task = load_model(ckpt)

#        self.model = model
#        self.task = task
#
#        if len(self.hooks) == 0:
#            module_name = "self.model.encoder.layers"
#            for module_id in range(len(eval(module_name))):
#                self.add_hook(
#                    f"{module_name}[{module_id}]",
#                    lambda input, output: input[0].transpose(0, 1),
#                )
#            self.add_hook("self.model.encoder", lambda input, output: output[0])

    def forward(self, wavs):
        if self.cfg_normalize:
            wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]

        device = wavs[0].device
        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1),
        )
        padded_wav = pad_sequence(wavs, batch_first=True)

        features, feat_padding_mask = self.model.extract_features(
            padded_wav,
            padding_mask=wav_padding_mask,
            mask=None,
            output_layer=self.wavlm_encoder_layers,
            ret_layer_results=True
        )
        return{"default": features}


