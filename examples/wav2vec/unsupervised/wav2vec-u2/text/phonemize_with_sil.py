#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import numpy as np
import sys
import re
import logging

def get_parser():
    parser = argparse.ArgumentParser(
        description="converts words to phones adding optional silences around in between words"
    )
    parser.add_argument(
        "--sil-prob",
        "-s",
        type=float,
        default=0,
        help="probability of inserting silence between each word",
    )
    parser.add_argument(
        "--surround",
        action="store_true",
        help="if set, surrounds each example with silence",
    )
    parser.add_argument(
        "--lexicon",
        help="lexicon to convert to phones",
        required=True,
    )
    parser.add_argument(
        "--language_id",
        help="if it is Chinese, every line hasn't space between charaters, we should list(line) first.",
        default="English",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    sil_prob = args.sil_prob
    surround = args.surround
    sil = "<SIL>"

    wrd_to_phn = {}

    with open(args.lexicon, "r") as lf:
        for line in lf:
            items = line.rstrip().split()
            assert len(items) > 1, line
            assert items[0] not in wrd_to_phn, items
            wrd_to_phn[items[0]] = items[1:]

    whitespace = re.compile(r"([ \t\r\n]+)")
    for line in sys.stdin:
        if args.language_id=='English':
            words = line.strip().split()
        elif args.language_id=='Chinese':
            line = re.sub(whitespace, "", line) 
            words = list(line)

        if not all(w in wrd_to_phn for w in words):
            logging.info(f"this line ({line}) skip!!")
            continue

        phones = []
        if surround:
            phones.append(sil)

        sample_sil_probs = None
        if sil_prob > 0 and len(words) > 1:
            sample_sil_probs = np.random.random(len(words) - 1)

        for i, w in enumerate(words):
            phones.extend(wrd_to_phn[w])
            if (
                sample_sil_probs is not None
                and i < len(sample_sil_probs)
                and sample_sil_probs[i] < sil_prob
            ):
                phones.append(sil)

        if surround:
            phones.append(sil)
        print(" ".join(phones))


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
