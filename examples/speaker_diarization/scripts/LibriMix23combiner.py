import argparse, os, shutil, subprocess

from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--libri2mix_folder", help="Root folder to Libri2Mix dataset, e.g ../Libri2Mix"
    )
    parser.add_argument(
        "--libri3mix_folder", help="Root folder to Libri3Mix dataset, e.g ../Libri3Mix"
    )
    parser.add_argument(
        "--libri23mix_folder",
        help="Root folder to Libri23Mix dataset, e.g ../Libri23Mix",
    )
    parser.add_argument("--fs", type=int, help="Sampling rate", choices=[8000, 16000])
    parser.add_argument(
        "--mode", type=str, help="Max or min mode", choices=["max", "min"]
    )

    args = parser.parse_args()

    args.libri2mix_folder = os.path.join(
        args.libri2mix_folder, f"wav{args.fs//1000}k", args.mode
    )
    args.libri3mix_folder = os.path.join(
        args.libri3mix_folder, f"wav{args.fs//1000}k", args.mode
    )
    args.libri23mix_folder = os.path.join(
        args.libri23mix_folder, f"wav{args.fs//1000}k", args.mode
    )

    return args


def copy_all_files(from_folder, to_folder):
    if not os.path.exists(to_folder):
        os.makedirs(to_folder)

    all_filenames = os.listdir(from_folder)
    all_filenames = [os.path.join(from_folder, elem) for elem in all_filenames]

    print(f"Copying from {from_folder} to {to_folder}.")
    for filename in tqdm(all_filenames):
        shutil.copy(filename, to_folder)


def link_all_files(from_folder, to_folder):
    if not os.path.exists(to_folder):
        os.makedirs(to_folder)

    print(f"Linking between {from_folder} and {to_folder}.")

    all_filenames = os.listdir(from_folder)

    for filename in tqdm(all_filenames):
        os.symlink(
            os.path.join(os.path.abspath(from_folder), filename),
            os.path.join(to_folder, filename),
        )


def main(args):

    # Copy the wav files
    for split_name in ["train-100", "dev", "test"]:
        subfolders = ["mix_both", "noise", "s1", "s2"]
        for subfolder in subfolders:

            curr_from = os.path.join(args.libri2mix_folder, split_name, subfolder)
            curr_to = os.path.join(args.libri23mix_folder, split_name, subfolder)
            assert os.path.exists(curr_from), f"{curr_from} does not exist"
            link_all_files(curr_from, curr_to)

            curr_from = os.path.join(args.libri3mix_folder, split_name, subfolder)
            assert os.path.exists(curr_from), f"{curr_from} does not exist"
            link_all_files(curr_from, curr_to)

        # s3 is only for Libri3Mix
        subfolder = "s3"
        curr_to = os.path.join(args.libri23mix_folder, split_name, subfolder)
        curr_from = os.path.join(args.libri3mix_folder, split_name, subfolder)
        assert os.path.exists(curr_from), f"{curr_from} does not exist"
        link_all_files(curr_from, curr_to)

    # for folder_name in ["diar"]:
    #     for split_name in ["train", "dev", "test"]:
    #         libri2_rttm_path = os.path.join(args.libri2mix_folder, folder_name, f"{split_name}2", "rttm")
    #         assert os.path.exists(libri2_rttm_path), f"{libri2_rttm_path} does not exist"
    #         libri3_rttm_path = os.path.join(args.libri3mix_folder, folder_name, f"{split_name}3", "rttm")
    #         assert os.path.exists(libri3_rttm_path), f"{libri3_rttm_path} does not exist"

    #         output_path = os.path.join(args.libri23mix_folder, folder_name, split_name, "rttm")
    #         if not os.path.exists(output_path):
    #             os.makedirs(os.path.dirname(output_path))

    #         cat_command = ["cat", libri2_rttm_path, libri3_rttm_path]
    #         with open(output_path, "w") as output_file:
    #             subprocess.run(cat_command, stdout=output_file, text=True)

    # for folder_name in ["tsv"]:
    #     for split_name in ["train", "dev", "test"]:
    #         libri2_tsv_path = os.path.join(args.libri2mix_folder, folder_name, f"{split_name}.tsv")
    #         assert os.path.exists(libri2_tsv_path), f"{libri2_tsv_path} does not exist"
    #         libri3_tsv_path = os.path.join(args.libri3mix_folder, folder_name, f"{split_name}.tsv")
    #         assert os.path.exists(libri3_tsv_path), f"{libri3_tsv_path} does not exist"

    #         output_path = os.path.join(args.libri23mix_folder, folder_name, f"{split_name}.tsv")
    #         if not os.path.exists(os.path.dirname(output_path)):
    #             os.makedirs(os.path.dirname(output_path))

    #         cat_command = ["cat", libri2_tsv_path, libri3_tsv_path]
    #         with open(output_path, "w") as output_file:
    #             subprocess.run(cat_command, stdout=output_file, text=True)


if __name__ == "__main__":
    args = get_args()
    main(args)
