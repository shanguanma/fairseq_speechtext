import argparse, os, random

from tqdm import tqdm
import soundfile as sf


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_librispeech", help="the path to root of the LibriSpeech dataset, eg. ....../LibriSpeech/")
    # parser.add_argument("--tsv_folder", help="the path to the tsv file of the split, e.g. ../Libri2Mix/wav16k/max/tsv_files/")
    parser.add_argument("--path_librimix", help="the path to the librimix dataset e.g. ../wav16k/max/")
    parser.add_argument("--output_folder", help="the relative path to write the aux file inside the Librimix folder, e.g. tsv_files")
    parser.add_argument("--type" , choices=["train", "test"], help="split of the dataset, it can be train or test")
    parser.add_argument("--random_seed", type=int, help="Random seed for reproducibility, default value is 42", default=42)

    args = parser.parse_args()

    args.output_folder = os.path.join(args.path_librimix, args.output_folder)
    if not os.path.exists(args.output_folder):
         os.makedirs(args.output_folder)

    args.path_librispeech = os.path.join(args.path_librispeech, 
                 f"{args.type}-clean{'-100' if args.type=='train' else ''}")
    
    args.path_librimix = os.path.join(
        args.path_librimix, 
        f"{args.type}{'-100' if args.type=='train' else ''}",
        "mix_both"
    )

    return args


def main(args):
    random.seed(args.random_seed)

    sample_list = os.listdir(args.path_librimix)
    uttid_list = [sample_name.split(".")[0] for sample_name in sample_list]

    if args.type == "train":
         uttid_list_dev = random.sample(uttid_list, int(0.02*len(uttid_list)))
         uttid_list = [elem for elem in uttid_list if elem not in uttid_list_dev]

    processed = []
    numspk_list = []

    output_path = os.path.join(args.output_folder, f"{args.type}.tsv")

    with open(output_path, "w") as output_file:
        for uttid in tqdm(uttid_list):
                # Process each sample one time
                if uttid in processed:
                    continue
                    
                processed.append(uttid)

                # Find it in LibriSpeech
                librispeech_ids = uttid.split("_")

                # Get the number of speakers and save it
                numspk = len(librispeech_ids)
                if numspk not in numspk_list:
                    numspk_list.append(numspk)

                for librispeech_id in librispeech_ids:
                    curr_list = librispeech_id.split("-")
                    currid_folder = os.path.join(args.path_librispeech, curr_list[0], curr_list[1])
                    assert os.path.exists(currid_folder), f"{librispeech_id} led to folder {currid_folder} but this folder does not exist"

                    all_uttids = [elem.split(".")[0] for elem in os.listdir(currid_folder) if elem.endswith(".flac")]
                    assert librispeech_id in all_uttids, f"{librispeech_id} led to folder {currid_folder} but this folder does not have this utterance"

                    all_uttids.remove(librispeech_id)

                    aux_id = random.choice(all_uttids)
                    assert aux_id != librispeech_id, f"{aux_id=} cannot be equal to the {librispeech_id=}"

                    full_aux_path = os.path.abspath(os.path.join(currid_folder, f"{aux_id}.flac"))

                    full_uttid_path = os.path.abspath(os.path.join(args.path_librimix, f"{uttid}.wav"))
                    curr_uttid_length = sf.info(full_uttid_path).frames
                    curr_speaker_fullid = curr_list[0]

                    aux_entry = "\t".join([uttid, full_uttid_path, full_aux_path, str(curr_uttid_length), curr_speaker_fullid])
                    output_file.write(aux_entry)
                    output_file.write("\n")

        print(f"The dataset consists of {len(processed)} mixtures with folowing number of speakers: {numspk_list}")
    
    if args.type == "train":
        processed = []
        numspk_list = []

        output_path = os.path.join(args.output_folder, f"dev.tsv")
        with open(output_path, "w") as output_file:
            for uttid in tqdm(uttid_list_dev):
                    # Process each sample one time
                    if uttid in processed:
                        continue
                        
                    processed.append(uttid)

                    # Find it in LibriSpeech
                    librispeech_ids = uttid.split("_")

                    # Get the number of speakers and save it
                    numspk = len(librispeech_ids)
                    if numspk not in numspk_list:
                        numspk_list.append(numspk)

                    for librispeech_id in librispeech_ids:
                        curr_list = librispeech_id.split("-")
                        currid_folder = os.path.join(args.path_librispeech, curr_list[0], curr_list[1])
                        assert os.path.exists(currid_folder), f"{librispeech_id} led to folder {currid_folder} but this folder does not exist"

                        all_uttids = [elem.split(".")[0] for elem in os.listdir(currid_folder) if elem.endswith(".flac")]
                        assert librispeech_id in all_uttids, f"{librispeech_id} led to folder {currid_folder} but this folder does not have this utterance"

                        all_uttids.remove(librispeech_id)

                        aux_id = random.choice(all_uttids)
                        assert aux_id != librispeech_id, f"{aux_id=} cannot be equal to the {librispeech_id=}"

                        full_aux_path = os.path.abspath(os.path.join(currid_folder, f"{aux_id}.flac"))

                        full_uttid_path = os.path.abspath(os.path.join(args.path_librimix, f"{uttid}.wav"))
                        curr_uttid_length = sf.info(full_uttid_path).frames
                        curr_speaker_fullid = curr_list[0]

                        aux_entry = "\t".join([uttid, full_uttid_path, full_aux_path, str(curr_uttid_length), curr_speaker_fullid])
                        output_file.write(aux_entry)
                        output_file.write("\n")

            print(f"The DEV dataset consists of {len(processed)} mixtures with folowing number of speakers: {numspk_list}")
        



if __name__=="__main__":
    args = get_args()
    main(args)

