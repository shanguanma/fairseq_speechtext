import os 
import sys
import joblib
import argparse
import logging
import math

import numpy as np
import torch
import tqdm
from common import MfccFeatureReader, get_path_iterator, ApplyKmeans, get_path_iterator_for_dump

logging.basicConfig(
     level=logging.DEBUG,
     format='%(asctime)s  (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
)
logger = logging.getLogger('dump_pseudo_label_on_mfcc')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsvfile",type=str, help="for example, train-960.tsv for training in fariseq data format")
    parser.add_argument("--nj", default=1, type=int, help="only support mfcc, it will  use multi thread for computing")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--label-path", type=str)
    parser.add_argument("--km-path", type=str)

    return parser


def dump_pseudo_label_mfcc(tsvfile, km_path, sample_rate, nj):
    reader = MfccFeatureReader(sample_rate)
    apply_kmeans = ApplyKmeans(km_path)
    generator, num = get_path_iterator_for_dump(tsvfile)
    iterator = generator()
    if nj>1:
        feats = joblib.Parallel(n_jobs=nj)(joblib.delayed(reader.get_feats)(path) for path, nsample in tqdm.tqdm(iterator, total=num))
        p_labs  = joblib.Parallel(n_jobs=nj)(joblib.delayed(apply_kmeans)(feat) for feat in tqdm.tqdm(feats, total=num))
        iterator = generator()
        nsamples = [nsample for _, nsample in iterator]
    else:
        nsamples, p_labs = [], []
        for path, nsample in tqdm.tqdm(iterator, total=num):
            feat = reader.get_feats(path,nsample)
            p_lab =apply_kmeans(feat).tolist()
            p_labs.append(p_lab)
            nsamples.append(nsample)
    return p_labs, nsamples

def dump_label(km_path, label_path, tsvfile, nj, sample_rate):
    logger.info(f"Dump pseudo labeling  for mfcc feature")
    p_labs, nsamples = dump_pseudo_label_mfcc(tsvfile, km_path, sample_rate, nj)
    with open(label_path, 'w')as f:
        for p_lab, nsample in zip(p_labs, nsamples):
            f.write(" ".join(map(str, p_lab))+"\n")
    logger.info("dumpping label finished ")



if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logger.info(str(args))
    dump_label(**vars(args))
