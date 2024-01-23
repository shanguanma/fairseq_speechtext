#!/usr/bin/env python3
from collections import OrderedDict

import torch

from fairseq.data import LanguagePairDataset, TokenBlockDataset
from fairseq.data.multi_corpus_dataset2 import MultiCorpusDataset2
from fairseq.data.multi_corpus_dataset import MultiCorpusDataset
from tests.test_train import mock_dict

if __name__ == "__main__":

    d = mock_dict()
    tokens_1 = torch.LongTensor([i for i in range(1, 50, 2)]).view(1, -1)
    tokens_ds1 = TokenBlockDataset(
        tokens_1,
        sizes=[tokens_1.size(-1)],
        block_size=1,
        pad=0,
        eos=1,
        include_targets=False,
    )
    print(len(tokens_ds1))
    ts1 = [s for s in tokens_ds1]
    print(f"{ts1}")
    dataset_1 = LanguagePairDataset(
        tokens_ds1, tokens_ds1.sizes, d, shuffle=False
    )
    print(len(dataset_1))
    ds1 = [ s for s in dataset_1]
    print(f"{ds1}")     
    tokens_2 = torch.LongTensor([i for i in range(0, 50, 2)]).view(1, -1)
    tokens_ds2 = TokenBlockDataset(
        tokens_2,
        sizes=[tokens_2.size(-1)],
        block_size=1,
        pad=0,
        eos=1,
        include_targets=False,
    )
    dataset_2 = LanguagePairDataset(
        tokens_ds2, tokens_ds2.sizes, d, shuffle=False
    )
    m = MultiCorpusDataset(
        OrderedDict({0: dataset_1, 1: dataset_2}),
        distribution=[0.5,0.5],
        seed=0,
        sort_indices=False,
    )
    print(len(m))
    #m1 = [s for s in m]
    #for s in m:
    #    print(f"{s}")
    m.set_epoch(1)
    #for s in m:
    #    print(f"{s}")
    indices = m.ordered_indices()
    print(f"{indices}")
    for j in indices:
        print(f"{m[j]}")
    count_sample_from_first_dataset = 0
    items = set()
    for i in indices:
        item = m[i]["source"].item()
        if item % 2 == 1:
            count_sample_from_first_dataset += 1

        items.add(item)
