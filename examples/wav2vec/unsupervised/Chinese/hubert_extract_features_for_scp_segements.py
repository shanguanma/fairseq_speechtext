#!/usr/bin/env python3
# Author: Duo MA
# Email: maduo@cuhk.edu.cn

import argparse
import os
import os.path as osp
import tqdm
import torch
import torch.nn.functional as F
from shutil import copyfile

from npy_append_array import NpyAppendArray

import fairseq
import soundfile as sf


def get_parser():
    parser = argparse.ArgumentParser(
        description="compute kmeans codebook from kaldi-computed feats"
    )
    # fmt: off
    parser.add_argument('data', help='location of tsv files')
    parser.add_argument('--split', help='which split to read', required=True)
    parser.add_argument('--save-dir', help='where to save the output', required=True)
    parser.add_argument('--checkpoint', type=str, help='checkpoint for wav2vec ctc model', required=True)
    parser.add_argument('--layer', type=int, default=14, help='which layer to use')
    # fmt: on

    return parser


class SegmentsExtractor:
    """Emulating kaldi extract-segments.cc

    Args:
        segments (str): The file format is
            "<segment-id> <recording-id> <start-time> <end-time>\n"
            "e.g. call-861225-A-0050-0065 call-861225-A 5.0 6.5\n"
    """

    def __init__(self, fname: str, segments: str = None, multi_columns: bool = False):
        assert check_argument_types()
        self.wav_scp = fname
        self.multi_columns = multi_columns
        self.wav_dict = {}
        with open(self.wav_scp, "r") as f:
            for line in f:
                recodeid, wavpath = line.strip().split(None, 1)
                if recodeid in self.wav_dict:
                    raise RuntimeError(f"{recodeid} is duplicated")
                self.wav_dict[recodeid] = wavpath

        self.segments = segments
        self.segments_dict = {}
        with open(self.segments, "r") as f:
            for line in f:
                sps = line.rstrip().split(None)
                if len(sps) != 4:
                    raise RuntimeError("Format is invalid: {}".format(line))
                uttid, recodeid, st, et = sps
                self.segments_dict[uttid] = (recodeid, float(st), float(et))

                if recodeid not in self.wav_dict:
                    raise RuntimeError(
                        'Not found "{}" in {}'.format(recodeid, self.wav_scp)
                    )

    def generator(self):
        recodeid_counter = {}
        for utt, (recodeid, st, et) in self.segments_dict.items():
            recodeid_counter[recodeid] = recodeid_counter.get(recodeid, 0) + 1

        cached = {}
        for utt, (recodeid, st, et) in self.segments_dict.items():
            wavpath = self.wav_dict[recodeid]
            if recodeid not in cached:
                if wavpath.endswith("|"):
                    if self.multi_columns:
                        raise RuntimeError(
                            "Not supporting multi_columns wav.scp for inputs by pipe"
                        )
                    # Streaming input e.g. cat a.wav |
                    with kaldiio.open_like_kaldi(wavpath, "rb") as f:
                        with BytesIO(f.read()) as g:
                            array, rate = soundfile.read(g)

                else:
                    array, rate = soundfile.read(wavpath)
                cached[recodeid] = array, rate

            array, rate = cached[recodeid]
            # Keep array until the last query
            recodeid_counter[recodeid] -= 1
            if recodeid_counter[recodeid] == 0:
                cached.pop(recodeid)
            # Convert starting time of the segment to corresponding sample number.
            # If end time is -1 then use the whole file starting from start time.
            if et != -1:
                array = array[int(st * rate) : int(et * rate)]
            else:
                array = array[int(st * rate) :]

            yield utt, (array, rate)


class HuBertFeatureReader(object):
    def __init__(self, cp_file, layer):
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [cp_file]
        )
        model = model[0]
        model.eval()
        model.cuda()
        self.model = model
        self.task = task
        self.layer = layer

    def read_audio(self, fname):
        """Load an audio file and return PCM along with the sample rate"""
        wav, sr = sf.read(fname)
        assert sr == 16e3

        return wav

    def get_feats(self, loc):
        x = self.read_audio(loc)
        with torch.no_grad():
            source = torch.from_numpy(x).float().cuda()
            if self.task.cfg.normalize:
                assert source.dim() == 1, source.dim()
                with torch.no_grad():
                    source = F.layer_norm(source, source.shape)
            source = source.view(1, -1)

            m_res = self.model(source=source, mask=False, features_only=True, layer=self.layer)
            return m_res["x"].squeeze(0).cpu()


def get_iterator(args):
    if args.segments is not None:
        extractor = SegmentsExtractor(
            args.scp, segments=args.segments, multi_columns=args.multi_columns_input
        )
        generator = extractor.generator


    with open(osp.join(args.data, args.split) + ".scp", "r") as fp:
        lines = fp.read().split("\n")
        files = [line.split()[1] for line in lines if len(line) > 0]
        #lines = fp.read().split("\n")
        #root = lines.pop(0).strip()
        #files = [osp.join(root, line.split("\t")[0]) for line in lines if len(line) > 0]

        num = len(files)
        reader = HuBertFeatureReader(args.checkpoint, args.layer)

        def iterate():
            for fname in files:
                w2v_feats = reader.get_feats(fname)
                yield w2v_feats

    return iterate, num

def main():
    parser = get_parser()
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    def create_files(dest):
        copyfile(osp.join(args.data, args.split) + ".tsv", dest + ".tsv")
        if osp.exists(osp.join(args.data, args.split) + ".wrd"):
            copyfile(osp.join(args.data, args.split) + ".wrd", dest + ".wrd")
        if osp.exists(osp.join(args.data, args.split) + ".phn"):
            copyfile(osp.join(args.data, args.split) + ".phn", dest + ".phn")

        if osp.exists(dest + ".npy"):
            os.remove(dest + ".npy")
        npaa = NpyAppendArray(dest + ".npy")
        return npaa

    save_path = osp.join(args.save_dir, args.split)
    npaa = create_files(save_path)

    generator, num = get_iterator(args)
    iterator = generator()

    with open(save_path + ".lengths", "w") as l_f:
        for w2v_feats in tqdm.tqdm(iterator, total=num):
            print(len(w2v_feats), file=l_f)

            if len(w2v_feats) > 0:
                npaa.append(w2v_feats.numpy())


if __name__ == "__main__":
    main()
