#!/usr/bin/env python3
# Author: Duo MA
# Email: maduo@cuhk.edu.cn

class Voicelm2FeatureReader(object):
    def __init__(self, sample_rate, ckpt_path, layer, max_chunk=1600000):
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model[0].eval().to(self.device)
        self.task = task
        self.layer = layer
        self.max_chunk = max_chunk
        self.sample_rate = sample_rate
        logger.info(f"Task config: \n {self.task.cfg}")
        logger.info(f"max_chunk = {self.max_chunk}")
    def read_audio(self, path):
        wav, sr = sf.read(path)
        assert sr == self.sample_rate,sr
        if wav.ndim == 2:
            wav = wav.mean(-1)
        return wav

    def get_feats(self, path):
        x = self.read_audio(path)
        with torch.no_grad():
            x = torch.from_numpy(x).float().to(self.device)
            x = x.view(1,-1)

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
               x_chunk  = x[:, start: start+self.max_chunk]
               feat_chunk, _ = self.model.extract_features(
                   source=x_chunk,
                   padding_mask=None,
                   mask=False,
                   output_layer=self.layer,
               )
               feat.append(feat_chunk)
        return  torch.cat(feat, 1).squeeze(0).cpu()



