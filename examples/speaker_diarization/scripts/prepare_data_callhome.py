import glob, os, soundfile, json, argparse
from collections import defaultdict
from tqdm import tqdm


class Segment(object):
    def __init__(self, uttid, spkr, stime, etime, name):
        self.uttid = uttid
        self.spkr = spkr
        self.stime = round(stime, 2)
        self.etime = round(etime, 2)
        self.name = name

    def change_stime(self, time):
        self.stime = time

    def change_etime(self, time):
        self.etime = time


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--path_grid", help="the path for the ami")
    parser.add_argument("--path_wav", help="the path for the wav")
    parser.add_argument("--out_text", help="the path for the wav")
    parser.add_argument("--rate", default=100, type=int, help="the path for the tsv")

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    outs = open(args.out_text, "w")
    segments = {}
    with open(args.path_grid) as f:
        for line in f:
            _, uttid, _, stime, offset, _, _, spk_id, _ = line.strip("\n").split()
            if uttid not in segments:
                num_spk = 1
                spk = {}
                segments[uttid] = []
            if spk_id not in spk:
                spk[spk_id] = num_spk
                num_spk += 1
            segments[uttid].append(
                Segment(
                    uttid,
                    spk[spk_id],
                    float(stime),
                    float(stime) + float(offset),
                    spk_id,
                )
            )

    for uttid in tqdm(segments):
        segments[uttid] = sorted(segments[uttid], key=lambda x: x.spkr)
        intervals = defaultdict(list)

        dic = {}
        # Summary the intervals for all speakers
        for i in range(len(segments[uttid])):
            interval = [segments[uttid][i].stime, segments[uttid][i].etime]
            intervals[segments[uttid][i].spkr].append(interval)
            if (
                str(segments[uttid][i].uttid) + "_" + str(segments[uttid][i].spkr)
                not in dic
            ):
                dic[
                    str(segments[uttid][i].uttid) + "_" + str(segments[uttid][i].spkr)
                ] = segments[uttid][i].name

        # Save the labels
        wav_file = os.path.join(args.path_wav, uttid) + ".wav"
        # import pdb; pdb.set_trace()
        orig_audio, sample_rate = soundfile.read(wav_file)
        if len(orig_audio.shape) > 1:
            orig_audio = orig_audio[:, 0]
        length = len(orig_audio)
        id_full = wav_file.split("/")[-1][:-4]
        for key in intervals:
            labels = [0] * int(length / sample_rate * args.rate)  # 40ms, one label
            for interval in intervals[key]:
                s, e = interval
                for i in range(
                    int(s * args.rate), min(int(e * args.rate) + 1, len(labels))
                ):
                    labels[i] = 1

            room_speaker_id = id_full + "_" + str(key)
            speaker_id = dic[room_speaker_id]

            # for aux_id in utt2aux[id_full]:
            # 	if speaker_id in aux_id:
            # 		aux_ct = aux_id
            # 		break

            res = {
                "filename": id_full,
                "speaker_key": key,
                "speaker_id": speaker_id,
                "labels": labels,
                "real_data": "true",
                "aux": speaker_id,
                "num_spk": 6,
            }
            json.dump(res, outs)
            outs.write("\n")


if __name__ == "__main__":
    main()
