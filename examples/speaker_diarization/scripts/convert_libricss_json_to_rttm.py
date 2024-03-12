import json
from tqdm import tqdm

dataset = "dev"

rttm_path = f"/mlx_devbox/users/ajy/playground/mnt/bn/junyi-nas-hl2/datasets/jsalt2020_simulate/data/SimLibriCSS-{dataset}/rttm"

rttm = open(rttm_path, "w")
with open(
    f"/mlx_devbox/users/ajy/playground/mnt/bn/junyi-nas-hl2/datasets/jsalt2020_simulate/data/SimLibriCSS-{dataset}/mixspec.json"
) as f:
    mix_info = json.load(f)

    for utt_info in tqdm(mix_info):
        utt_id = utt_info["output"].split("/")[-1][:-4]
        for sub_utt_info in utt_info["inputs"]:
            spk_id = sub_utt_info["speaker_id"]
            start = sub_utt_info["offset"]
            duration = sub_utt_info["length_in_seconds"]

            print(
                f"SPEAKER {utt_id} 1 {start} {duration} <NA> <NA> {spk_id} <NA> <NA>",
                file=rttm,
            )

rttm.close()
