import os 
import sys
import joblib
import argparse
import logging
import math

import numpy as np
import torch
import tqdm
from common import HubertFeatureReader,ApplyKmeans,get_path_iterator_for_dump

logging.basicConfig(
     level=logging.DEBUG,
     format='%(asctime)s  (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
)
logger = logging.getLogger('dump_pseudo_label_on_hubert')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--km-path", type=str, help="path for  k-means model.")
    parser.add_argument("--label-path", type=str, help="")
    parser.add_argument("--tsvfile",type=str, help="for example, train-960.tsv for training in fariseq data format")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--ckpt-path", type=str, help="trained iter1 hubert model (e.g.using mfcc feature and its kmeans label to train a hubert model) ")
    parser.add_argument("--layer", type=int, default=6, help="select a number of layer of trained hubert(this model is trained on mfcc and  its kmeans label) to as hidden representation feature, it is used to replace mfcc feature to train hubert model from scratch")
  
    return parser




def dump_pseudo_label_hubert(tsvfile,km_path,sample_rate, ckpt_path,layer):
    apply_kmeans = ApplyKmeans(km_path)
    reader = HubertFeatureReader(sample_rate,ckpt_path, layer)
    generator, num = get_path_iterator_for_dump(tsvfile)
    iterator = generator()
    
    p_labs=[]
    for path, nsample in tqdm.tqdm(iterator, total=num):
        feat = reader.get_feats(path)
        p_lab = apply_kmeans(feat).tolist()
        p_labs.append(p_lab)
    return p_labs


def dump_label(tsvfile,km_path,sample_rate, ckpt_path,layer, label_path):  
    logger.info(f"Dumping pseudo labeling for hubert")
    p_labs = dump_pseudo_label_hubert(tsvfile,km_path,sample_rate, ckpt_path,layer)
    with open(label_path, "w")as f:
        for p_lab in p_labs:
            f.write(" ".join(map(str,p_lab)) + "\n")

    logger.info("finished successfully")
  




if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logger.info(str(args))
    dump_label(**vars(args))
