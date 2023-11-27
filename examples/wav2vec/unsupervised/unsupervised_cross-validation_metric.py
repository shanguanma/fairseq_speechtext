#!/usr/bin/env  python3
## Copyright 2023  The Chinese University of Hong Kong,Shen Zhen (Author: Junyi Ao
##                                                                         Duo  Ma)

import argparse
import glob
import math

parser = argparse.ArgumentParser()
parser.add_argument("--exp_path")
parser.add_argument("--dev_name", default="dev-other")
args = parser.parse_args()

best_uer = (math.inf, math.inf, "", "", "")
best_ppl = (math.inf, math.inf, "", "", "")


##(TODO) the script want to choice
all_infos = {}
# for file_path in glob.glob(f"{args.exp_path}/*/*.txt"):
if __name__ == "__main__":
    #for file_path in glob.glob(f"{args.exp_path}/*/*.log"):
    for file_path in glob.glob(f"{args.exp_path}/*.log"):
        #setting_index_name = file_path.split("/")[-2]
        exp_name = file_path.split("/")[-2]
        #print(f"exp_name: {exp_name}")
        with open(file_path) as f:
            next(f)  ## skip first line
            for line in f:
                line = line.strip()
                if (
                    line.find(f"[{args.dev_name}][INFO] - ") != -1 ## match
                    and line.find(f"{args.dev_name}_best_weighted_lm_ppl") != -1 ## match
                ):
                    #print(f"info: {line}")
                    info = eval(line.strip().split(f"[{args.dev_name}][INFO] - ")[-1])
                    # info = info.split(f'[{args.dev_name}]')
                    ## setting_index_name and update steps as  experiment name
                    #exp_name=f"exp{setting_index_name}_"+info[f"{args.dev_name}_num_updates"]
                    all_infos[exp_name] = info
                    #print(f"info::::: {info}")
                    if (
                        float(info[f"{args.dev_name}_best_weighted_lm_ppl"])
                        < best_ppl[0]
                    ):
                        best_ppl = (
                            float(info[f"{args.dev_name}_best_weighted_lm_ppl"]),
                            float(info[f"{args.dev_name}_uer"]),
                            info[f"{args.dev_name}_num_updates"],
                            exp_name,
                            info["epoch"],
                        )
                        #all_infos[exp_name] = info
                    if float(info[f"{args.dev_name}_uer"]) < best_uer[1]:
                        best_uer = (
                            float(info[f"{args.dev_name}_best_weighted_lm_ppl"]),
                            float(info[f"{args.dev_name}_uer"]),
                            info[f"{args.dev_name}_num_updates"],
                            exp_name,
                            info["epoch"],
                        )
                        #all_infos[exp_name] = info

    step_2 = []
    print(f"all_infos:::::{all_infos}")   
    for exp_name in all_infos:
        if (
            float(all_infos[exp_name][f"{args.dev_name}_best_weighted_lm_ppl"])
            < best_ppl[0] * 1.2
        ):
            #print(f"all_infos:::::: {all_infos}")
            step_2.append((all_infos[exp_name], exp_name))
    step_2.sort(key=lambda x: x[0][f"{args.dev_name}_lm_score_sum"])

    print(f"best_ppl:::{best_ppl}")
    print(f"best_uer:::{best_uer}")

    print("Best checkpoint is:")
    print(step_2)
