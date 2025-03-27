#!/usr/bin/env python3
# Author: Duo MA
# Email: maduo@cuhk.edu.cn

import numpy as np
from typing import List
import torch

def int2vec(x, vec_dim=8, dtype=np.int32):
    b = ("{:0" + str(vec_dim) + "b}").format(x)
    # little-endian order: lower bit first
    return (np.array(list(b)[::-1]) == "1").astype(dtype)


def generate_pse_embedding(token_list: List, max_spk_num: int):
    embedding = np.zeros((len(token_list), max_spk_num), dtype=np.float32)
    for idx, pse_label in enumerate(token_list):
        print(f"idx: {idx}, pse_label: {pse_label}!!")
        emb = int2vec(int(pse_label), vec_dim=max_spk_num, dtype=np.float32)
        print(f"vec: {emb}")
        embedding[idx] = emb
    return torch.from_numpy(embedding)

if __name__ == "__main__":
    token_list=[1,1,1,2,3,4,6,6,0,7]
    max_spk_num=3
    x = 1
    v = int2vec(x,vec_dim=max_spk_num)
    print(f"v: {v}")

    emb = generate_pse_embedding(token_list,max_spk_num)
    print(f"emb:  {emb}")


    power_weight = torch.from_numpy(
            2 ** np.arange(max_spk_num)[np.newaxis, np.newaxis, :]
        ).float()# of shape (1, 1, max_spk_num)
    print(f"power_weight: {power_weight}, its shape: {power_weight.shape}")

    int_token_arr = torch.from_numpy(
            np.array(token_list).astype(int)[np.newaxis, np.newaxis, :]
        ).int()
    print(f"int_token_arr: {int_token_arr}, its shape: {int_token_arr.shape}")
