import argparse
import glob
import math

parser = argparse.ArgumentParser()
parser.add_argument("exp_path")
args = parser.parse_args()

best_uer = (math.inf, math.inf, "", "", "")
best_ppl = (math.inf, math.inf, "", "", "")

all_infos = {}
for file_path in glob.glob(f"{args.exp_path}/*/*.txt"):
    exp_name = file_path.split('/')[-2]
    with open(file_path) as f:
        info = f.readline()
        info = eval(info.split('[INFO] - ')[-1])
        all_infos[exp_name] = info
        if float(info['dev_other_best_weighted_lm_ppl']) < best_ppl[0]:
            best_ppl = (float(info['dev_other_best_weighted_lm_ppl']), float(info['dev_other_uer']), info['dev_other_num_updates'], exp_name, info['epoch'])

        if float(info['dev_other_uer']) < best_uer[1]:
            best_uer = (float(info['dev_other_best_weighted_lm_ppl']), float(info['dev_other_uer']), info['dev_other_num_updates'], exp_name, info['epoch'])

step_2 = []
for exp_name in all_infos:
    if float(all_infos[exp_name]['dev_other_best_weighted_lm_ppl']) < best_ppl[0] * 1.2:
        step_2.append((all_infos[exp_name], exp_name))
step_2.sort(key=lambda x: x[0]["dev_other_lm_score_sum"])

print(best_ppl)
print(best_uer)

print("Best checkpoint is:")
print(step_2)
