import torch, os, tqdm, numpy, soundfile, argparse
from ts_vad.models.modules.speakerEncoder2 import ECAPA_TDNN
import soundfile as sf


def init_speaker_encoder(source):
    speaker_encoder = ECAPA_TDNN(C=1024).cuda()
    speaker_encoder.eval()
    loadedState = torch.load(source, map_location="cuda")
    selfState = speaker_encoder.state_dict()
    for name, param in loadedState.items():
        name = name.replace("speaker_encoder.", "")
        if name in selfState:
            selfState[name].copy_(param)
        else:
            print("Not exist ", name)
    for param in speaker_encoder.parameters():
        param.requires_grad = False
    return speaker_encoder


def extract_embeddings(batch, model):
    batch = torch.stack(batch)
    with torch.no_grad():
        embeddings = model.forward(batch.cuda())
    return embeddings


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--target_audio_path", help="the path for the audio tsv")
    parser.add_argument(
        "--target_embedding_path", help="the path for the output embeddings"
    )
    parser.add_argument("--source", help="the part for the speaker encoder")
    parser.add_argument(
        "--length_embedding",
        type=float,
        default=6,
        help="length of embeddings, seconds",
    )
    parser.add_argument(
        "--step_embedding", type=float, default=1, help="step of embeddings, seconds"
    )
    parser.add_argument(
        "--batch_size", type=int, default=96, help="step of embeddings, seconds"
    )
    args = parser.parse_args()

    return args


def main():
    args = get_args()
    files = []
    with open(args.target_audio_path) as f:
        for line in f:
            files.append(line.split("\t")[-3])

    model = init_speaker_encoder(args.source)
    for file in tqdm.tqdm(files):
        # print(file)
        output_file = (
            args.target_embedding_path
            + "/"
            + file.split("/")[-1].replace(".flac", ".pt")
        )
        if not os.path.isfile(output_file):
            batch = []
            embeddings = []
            wav_length = sf.SoundFile(file).frames
            if wav_length > int(args.length_embedding * 16000):
                for start in range(
                    0,
                    wav_length - int(args.length_embedding * 16000),
                    int(args.step_embedding * 16000),
                ):
                    stop = start + int(args.length_embedding * 16000)
                    target_speech, _ = soundfile.read(file)
                    target_speech = target_speech[start:stop]
                    target_speech = torch.FloatTensor(numpy.array(target_speech))
                    batch.append(target_speech)
                    if len(batch) == args.batch_size:
                        embeddings.extend(extract_embeddings(batch, model))
                        batch = []
            else:
                target_speech, _ = soundfile.read(file)
                target_speech = torch.FloatTensor(numpy.array(target_speech))
                embeddings.extend(extract_embeddings([target_speech], model))
            if len(batch) != 0:
                embeddings.extend(extract_embeddings(batch, model))
            embeddings = torch.stack(embeddings)
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            torch.save(embeddings, output_file)


if __name__ == "__main__":
    main()
