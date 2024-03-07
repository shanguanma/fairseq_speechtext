#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file takes as input a lexicon.txt and output a new lexicon,
in which each word has a unique pronunciation.

The way to do this is to keep only the first pronunciation of a word
in lexicon.txt.
"""


import argparse
import logging
from pathlib import Path
from typing import List, Tuple
import re
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang-dir",
        type=str,
        help="""Input and output directory.
        It should contain a file lexicon.txt.
        This file will generate a new file uniq_lexicon.txt
        in it.
        """,
    )

    return parser.parse_args()


def read_lexicon(filename: str) -> List[Tuple[str, List[str]]]:
    """Read a lexicon from `filename`.

    Each line in the lexicon contains "word p1 p2 p3 ...".
    That is, the first field is a word and the remaining
    fields are tokens. Fields are separated by space(s).

    Args:
      filename:
        Path to the lexicon.txt

    Returns:
      A list of tuples., e.g., [('w', ['p1', 'p2']), ('w1', ['p3, 'p4'])]
    """
    ans = []

    with open(filename, "r", encoding="utf-8") as f:
        whitespace = re.compile("[ \t]+")
        for line in f:
            a = whitespace.split(line.strip(" \t\r\n"))
            if len(a) == 0:
                continue

            if len(a) < 2:
                logging.info(f"Found bad line {line} in lexicon file {filename}")
                logging.info("Every line is expected to contain at least 2 fields")
                sys.exit(1)
            word = a[0]
            if word == "<eps>":
                logging.info(f"Found bad line {line} in lexicon file {filename}")
                logging.info("<eps> should not be a valid word")
                sys.exit(1)

            tokens = a[1:]
            ans.append((word, tokens))

    return ans


def write_lexicon(filename: str, lexicon: List[Tuple[str, List[str]]]) -> None:
    """Write a lexicon to a file.

    Args:
      filename:
        Path to the lexicon file to be generated.
      lexicon:
        It can be the return value of :func:`read_lexicon`.
    """
    with open(filename, "w", encoding="utf-8") as f:
        for word, tokens in lexicon:
            f.write(f"{word} {' '.join(tokens)}\n")



def filter_multiple_pronunications(
    lexicon: List[Tuple[str, List[str]]]
) -> List[Tuple[str, List[str]]]:
    """Remove multiple pronunciations of words from a lexicon.

    If a word has more than one pronunciation in the lexicon, only
    the first one is kept, while other pronunciations are removed
    from the lexicon.

    Args:
      lexicon:
        The input lexicon, containing a list of (word, [p1, p2, ..., pn]),
        where "p1, p2, ..., pn" are the pronunciations of the "word".
    Returns:
      Return a new lexicon where each word has a unique pronunciation.
    """
    seen = set()
    ans = []

    for word, tokens in lexicon:
        if word in seen:
            continue
        seen.add(word)
        ans.append((word, tokens))
    return ans

def remove_tone(lexicon: List[Tuple[str, List[str]]]) -> List[Tuple[str, List[str]]]:
    ans = []
    for word, tokens in lexicon:
        new_tokens=[]
        for token in tokens:
            if token[-1].isdigit():
                new_tokens.append(token[:-2]) ## remove tone
            else:
                new_tokens.append(token)

        ans.append((word, new_tokens))
    return ans
def main():
    args = get_args()
    lang_dir = Path(args.lang_dir)

    lexicon_filename = lang_dir / "lexicon.txt"

    in_lexicon = read_lexicon(lexicon_filename)

    out_lexicon = filter_multiple_pronunications(in_lexicon)

    outt_lexicon = remove_tone(out_lexicon)
    write_lexicon(lang_dir / "uniq_lexicon_remove_tone.txt", outt_lexicon)

    logging.info(f"Number of entries in lexicon.txt: {len(in_lexicon)}")
    logging.info(f"Number of entries in uniq_lexicon_remove_tone.txt: {len(outt_lexicon)}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
