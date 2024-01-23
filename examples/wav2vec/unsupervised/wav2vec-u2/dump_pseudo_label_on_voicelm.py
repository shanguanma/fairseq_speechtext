import os 
import sys
import joblib
import argparse
import logging
import math

import numpy as np
import torch
import tqdm
from common import VoicelmFeatureReader,ApplyKmeans,get_path_iterator_for_dump

logging.basicConfig(
     level=logging.DEBUG,
     format='%(asctime)s  (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
)
logger = logging.getLogger('dump_pseudo_label_on_voicelm')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--km-path", type=str, help="path for  k-means model.")
    parser.add_argument("--label-path", type=str, help="")
    parser.add_argument("--tsvfile",type=str, help="for example, train-960.tsv for training in fariseq data format")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--ckpt-path", type=str, help="trained iter1 voicelm model ")
    parser.add_argument("--layer", type=int, default=7, help="select a number of layer of trained voicelm to as hidden representation feature")
  
    return parser




def dump_pseudo_label_voicelm(tsvfile,km_path,sample_rate, ckpt_path,layer):
    apply_kmeans = ApplyKmeans(km_path)
    reader = VoicelmFeatureReader(sample_rate,ckpt_path, layer)
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
    p_labs = dump_pseudo_label_voicelm(tsvfile,km_path,sample_rate, ckpt_path,layer)
    with open(label_path, "w")as f:
        for p_lab in p_labs:
            f.write(" ".join(map(str,p_lab)) + "\n")

    logger.info("finished successfully")
  




if __name__ == "__main__":
    import os
    ##  the below script  is not working,  it can't limit  cpu core number for generator  (e.g. yield)  
    #cpu_num = 1 # 这里设置成你想运行的CPU个数
    #os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
    #os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    #os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
    #os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    #os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)

    # Torch's multithreaded behavior needs to be disabled or
    # it wastes a lot of CPU and slow things down.
    # Do this outside of main() in case it needs to take effect
    # even when we are not invoking the main (e.g. when spawning subprocesses).
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    parser = get_parser()
    args = parser.parse_args()
    logger.info(str(args))
    dump_label(**vars(args))
