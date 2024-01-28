#!/usr/bin/env python3


import numpy as np
import logging
import argparse
from common import learn_kmeans



logging.basicConfig(
     level=logging.DEBUG,
     format='%(asctime)s  (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
)
logger = logging.getLogger('learn_kmeans')

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feats",type=str, help="store iter1 speciy layer feature of trained model(i.e.: voicelm, hubert, voicelm2")
    parser.add_argument("--n-clusters", default=500,type=int, help="number of clusters for k-means")
    parser.add_argument("--seed", default=42,type=int)
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

def load_voicelm_feature(feats):
    feats = f"{feats}" 
    feats = np.load(feats, mmap_mode="r") 
    logger.info(f"Loading featuree from npy file succcessfully, feats shape: {feats.shape}")

    return feats


def main(args):
    np.random.seed(args.seed)
    logger.info(f"loading features")
    feats = load_voicelm_feature(feats=args.feats)
    logger.info(f"learning kmeans")
    learn_kmeans(
        feats,
        km_path=args.km_path,
        n_clusters=args.n_clusters,
        init = args.init,
        max_iter = args.max_iter,
        batch_size = args.batch_size,
        tol = args.tol,
        max_no_improvement = args.max_no_improvement,
        n_init = args.n_init,
        reassignment_ratio =args.reassignment_ratio,
    )

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logger.info(str(args)) 
    main(args)
