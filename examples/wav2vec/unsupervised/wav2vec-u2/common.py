import argparse
import logging
import math
import os
import sys
import warnings
from  random  import sample


import fairseq
import joblib
import numpy as np

import soundfile as sf
import torch
import torchaudio
import tqdm
from sklearn.cluster import MiniBatchKMeans

logging.basicConfig(
     level=logging.DEBUG,
     format='%(asctime)s  (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
)
logger = logging.getLogger('common')


class MfccFeatureReader(object):
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
    def load_audio(self, path, nsample):
        wav, sr = sf.read(path)
        if nsample is not None and abs(abs(nsample) - len(wav)) >160:
            logging.warning(f"ref nsample length {nsample} != read {len(wav)} ({path})")
        assert sr == self.sample_rate, sr
        if wav.ndim == 2:
            wav = wav.mean(-1)
        return wav

    def get_feats(self, path, nsample=None):
        x = self.load_audio(path, nsample)
        with torch.no_grad():
            x = torch.from_numpy(x).view(1,-1).float()

            mfcc = torchaudio.compliance.kaldi.mfcc(
                waveform=x,
                sample_frequency=self.sample_rate,
                use_energy=False,
            ).transpose(0,1) ## (time, freq)

            delta = torchaudio.functional.compute_deltas(mfcc)
            ddelta = torchaudio.functional.compute_deltas(delta)
            concat  = torch.cat([mfcc,delta,ddelta], dim=0).transpose(0,1).contiguous()
            return  concat

class HubertFeatureReader(object):
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




class VoicelmFeatureReader(object):
    def __init__(self, sample_rate, ckpt_path, layer, max_chunk=1600000):
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
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



class ApplyKmeans(object):
    def __init__(self, km_path):
        self.km_model = joblib.load(km_path)
        self.nc = self.km_model.cluster_centers_.transpose()  ## (freq, n_cluster)
        self.nc_norm  = (self.nc**2).sum(0, keepdims=True) #(1, n_cluster)
    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()  #(time, freq)
        probs = (x**2).sum(1,keepdims=True) - 2 * np.matmul(x, self.nc) + self.nc_norm ## (time,n_cluster)
        return np.argmin(probs,axis=1) #(time,)



def train_km_model(
    n_clusters,
    init,
    max_iter,
    batch_size,
    tol,
    max_no_improvement,
    n_init,
    reassignment_ratio,
):
    return MiniBatchKMeans(
        n_clusters=n_clusters,
        init=init,
        max_iter=max_iter,
        batch_size=batch_size,
        verbose=1,
        compute_labels=False,
        tol=tol,
        max_no_improvement=max_no_improvement,
        init_size=None,
        n_init=n_init,
        reassignment_ratio=reassignment_ratio,

    )


def learn_kmeans(
    feats,
    km_path,
    n_clusters,
    init,
    max_iter,
    batch_size,
    tol,
    max_no_improvement,
    n_init,
    reassignment_ratio,

):
    km_model = train_km_model(
        n_clusters,
        init,
        max_iter,
        batch_size,
        tol,
        max_no_improvement,
        n_init,
        reassignment_ratio,

    )
    km_model.fit(feats)
    joblib.dump(km_model,f"{km_path}")
    inertia = -km_model.score(feats)/len(feats)
    logger.info("total intertia:%.5f",inertia)
    logger.info("K-means training successfully")



def get_path_iterator(tsvfile, portion=0.1):
    with open(tsvfile,'r')as f:
        root = f.readline().rstrip()
        lines = [line.rstrip() for line in f]
        nums = int(portion * len(lines))
        logger.info(f"numbers: {nums}")
        lines = sample(lines,nums) ## it will random element in lines
        logger.info(f"selected utterances: {len(lines)}")
        def iterate():
            for line in lines:
                subpath, nsample = line.split("\t") 
                yield f"{root}/{subpath}",int(nsample)

        return iterate, len(lines)


def get_path_iterator_for_dump(tsvfile):
    with open(tsvfile,'r')as f:
        root = f.readline().rstrip()
        lines = [line.rstrip() for line in f]
        def iterate():
            for line in lines:              
                subpath, nsample = line.split("\t") 
                yield f"{root}/{subpath}",int(nsample)
                          
        return iterate, len(lines)

