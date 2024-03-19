#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Data pre-processing: build vocabularies and binarize training data.
"""

import argparse
import glob
import os
import random
import logging
import soundfile
logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", metavar="DIR", help="root directory containing flac files to index"
    )
    parser.add_argument(
        "--dest_file", type=str, metavar="DIR", help="output directory"
    )
    parser.add_argument(
        "--ext", default="flac", type=str, metavar="EXT", help="extension to look for"
    )
    return parser


def main(args):
    
    dir_path = os.path.realpath(args.root)
    search_path = os.path.join(dir_path, "**/*." + args.ext)
    
    
    with open(args.dest_file, "w") as f:
        print(dir_path, file=f)

        for fname in glob.iglob(search_path, recursive=True):
            file_path = os.path.realpath(fname)

            frames = soundfile.info(fname).frames
            logging.info(f"{os.path.relpath(file_path, dir_path)}\t{frames}")
            print(
                "{}\t{}".format(os.path.relpath(file_path, dir_path), frames), file=f
            )

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
