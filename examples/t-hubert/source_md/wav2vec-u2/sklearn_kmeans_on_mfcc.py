import os 
import sys
import joblib
import argparse
import logging
import math

import numpy as np
import torch
import tqdm
from common import MfccFeatureReader, get_path_iterator, train_km_model, learn_kmeans

logging.basicConfig(
     level=logging.DEBUG,
     format='%(asctime)s  (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
)
logger = logging.getLogger('sklearn_kmeans_on_mfcc')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsvfile",type=str, help="for example, train-960.tsv for training in fariseq data format")
    parser.add_argument("--n-clusters", default=100,type=int, help="number of clusters for k-means")
    parser.add_argument("--nj", default=1, type=int, help="only support mfcc, it will  use multi thread for computing")
    parser.add_argument("--seed", default=0,type=int)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--portion", type=float, default=0.1, help="using a subset of the data(e.g.train-960.tsv")
    group  = parser.add_argument_group(description="related parameters for k-means model")
    group.add_argument("--km-path", type=str, help="path for  k-means model.")
    group.add_argument("--init", default="k-means++")
    group.add_argument("--max-iter", default=100, type=int)
    group.add_argument("--batch-size", default=10000, type=int)
    group.add_argument("--tol", default=0.0, type=float)
    group.add_argument("--max-no-improvement", default=100, type=int)
    group.add_argument("--n-init", default=20,type=int)
    group.add_argument("--reassignment-ratio", default=0.0, type=float)

    return parser


def get_mfcc_feature(tsvfile, sample_rate, nj, portion):
    reader = MfccFeatureReader(sample_rate)
    generator, num = get_path_iterator(tsvfile, portion)
    iterator = generator()
    if nj>1:
        feats = joblib.Parallel(n_jobs=nj)(joblib.delayed(reader.get_feats)(path) for path, nsample in tqdm.tqdm(iterator, total=num))
    else:
        feats = []
        for path, nsample in tqdm.tqdm(iterator, total=num):
            feat = reader.get_feats(path)
            feats.append(feat.cpu().numpy())
        np.random.shuffle(feats)
    logger.info("Getting MFCC feature successfully")
    return np.vstack(feats)




def load_feature(
    tsvfile,
    sample_rate,
    nj,
    portion,
):
    feat = get_mfcc_feature(tsvfile, sample_rate, nj, portion)
    return feat

def main(args):
    np.random.seed(args.seed)
    logger.info(f"Loading features")
    feats = load_feature(
        tsvfile=args.tsvfile,
        sample_rate=args.sample_rate,
        nj=args.nj,
        portion=args.portion,

    )
    logger.info(f"learning kmeans")
    learn_kmeans(
        feats,
        km_path=args.km_path,
        n_clusters=args.n_clusters,
        init=args.init,
        max_iter=args.max_iter,
        batch_size=args.batch_size,
        tol=args.tol,
        max_no_improvement=args.max_no_improvement,
        n_init=args.n_init,
        reassignment_ratio=args.reassignment_ratio,

    )

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logger.info(str(args))
    main(args)
