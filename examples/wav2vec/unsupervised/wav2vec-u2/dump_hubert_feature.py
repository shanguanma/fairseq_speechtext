import argparse
import logging
import math
import os
import sys
import tqdm
import fairseq
import numpy as np
from common import HubertFeatureReader, get_path_iterator
from npy_append_array import NpyAppendArray

logging.basicConfig(
     level=logging.DEBUG,
     format='%(asctime)s  (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
)
logger = logging.getLogger('dump_pseudo_label')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsvfile",type=str, help="for example, train-960.tsv for training in fariseq data format")
    parser.add_argument("--feat-dir",  type=str, help="store hubert feature directoy,this data is used to train iter2 hubert model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--ckpt-path", type=str, help="trained iter1 hubert model (e.g.using mfcc feature and its kmeans label to train a hubert model) ")
    parser.add_argument("--portion", type=float, default=0.1, help="using  a subset of the data(e.g.train-960 of librispeech)")
    parser.add_argument("--layer", type=int, default=6, help="select a number of layer of trained hubert(this model is trained on mfcc and  its kmeans label) to as hidden representation feature, it is used to replace mfcc feature to train hubert model from scratch")

    return parser

def get_hubert_feature(tsvfile, sample_rate, ckpt_path, layer,  portion, feat_dir):
    reader = HubertFeatureReader(sample_rate,ckpt_path, layer)
    
    generator, num = get_path_iterator(tsvfile, portion) ## shuffle input  wavforms
    iterator = generator()
    feat_path = f"{feat_dir}/train-960_10_percent_hubert_raw_feature.npy"
    os.makedirs(feat_dir,exist_ok=True)
    
    if os.path.exists(feat_path):
        os.remove(feat_path)

    feat_f  = NpyAppendArray(feat_path)
    for path, nsample in tqdm.tqdm(iterator, total=num):
        feat = reader.get_feats(path)
        feat_f.append(feat.cpu().numpy())
    logger.info(f"Dump hubert feature successfully, it is at {feat_path}")
   
def main(args):
    np.random.seed(args.seed)
    logger.info(f"Dump features")
    get_hubert_feature(
        tsvfile = args.tsvfile,
        sample_rate  = args.sample_rate,
        ckpt_path = args.ckpt_path,
        layer = args.layer,
        portion=args.portion,
        feat_dir=args.feat_dir
    )

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logger.info(str(args))
    main(args)

