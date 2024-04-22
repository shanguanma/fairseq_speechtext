import argparse
import glob
import re
import os 
import torch


def get_args():
    parser = argparse.ArgumentParser(description='average model')
    parser.add_argument('--dst_model', required=True, help='averaged model')
    parser.add_argument('--src_path',
                        required=True,
                        help='src model path for average')
    parser.add_argument('--num',
                        default=5,
                        type=int,
                        help='nums for averaged model')
    parser.add_argument('--min_epoch',
                        default=0,
                        type=int,
                        help='min epoch used for averaging model')
    parser.add_argument(
        '--max_epoch',
        default=65536,  # Big enough
        type=int,
        help='max epoch used for averaging model')
    args = parser.parse_args()
    print(args)
    return args




def main():
    args = get_args()

     #path_list = glob.glob('{}/[!checkpoint_last][!checkpoint_best].pt'.format(
     #   args.src_path))

    path_list = glob.glob('{}/*.pt'.format(
        args.src_path))

    #print(f"path_list: {path_list}")
    #re.findall(r"(?<=model_)\d*(?=.pt)", )
    #pattern = re.compile(r"checkpoint")
    pattern = re.compile(r"checkpoint(\d+)|(_\d+)\.pt")

    entries = []
    for i, f in enumerate(path_list):
        m = pattern.search(f)
        print(m)
        if m is not None:
            idx = float(m.group(1)) if len(m.groups()) > 0 else i
            entries.append((idx, m.group(0)))
    print(f"entries: {entries}")
    keep_match=False
    if keep_match:
        return [(os.path.join(path, x[1]), x[0]) for x in sorted(entries, reverse=True)]
    else:
        return [os.path.join(path, x[1]) for x in sorted(entries, reverse=True)]


"""
    iter_checkpoints = []
    for c in path_list:
        #a = re.findall(r"(?<=checkpoint)(\d*)(?=.pt)", c)

        print(f"a: {a}")
        result = pattern.search(c)
        #result = pattern.match(c)
        if not result:
            print(f"Invalid checkpoint filename {c}")
            continue
        #print(f"checkpoint : {c}")  
        #print(f"{dir(result)}")
        #print(f"{result.group()}")
        #if re.match('.*?([0-9]+)$', c).group(1):
        ckpt_name = c.split("/")[-1].split(".")[0]
        if re.match("_",ckpt_name):
            print(f"match _: {ckpt_name}")
        else:
            print(f"not match _: {ckpt_name}")
        #if  re.compile(r'_').search(ckpt_name):
        #    print(f"end of with number ckpt: {c}")
        #else:
        #    print(f"none end of with number ckpt: {c}")
        #ckpt_name = c.split("/")[-1].split(".")[0]
        #if ckpt_name.endswith()
        #print(f"number: {int(result.group(1))}, {result.group(2)}")
        #iter_checkpoints.append((int(result.group(1)), c))

    # iter_checkpoints is a list of tuples. Each tuple contains
    # two elements: (iteration_number, checkpoint-iteration_number.pt)
    iter_checkpoints = sorted(iter_checkpoints, reverse=True, key=lambda x: x[0])    
    



    path_list = path_list[-args.num:]
    print(path_list)
    avg = None
    num = args.num
    assert num == len(path_list)
    for path in path_list:
        print('Processing {}'.format(path))
        states = torch.load(path, map_location=torch.device('cpu'))
        states = states['model'] if 'model' in states else states
        if avg is None:
            avg = states
        else:
            for k in avg.keys():
                avg[k] += states[k]
    # average
    for k in avg.keys():
        if avg[k] is not None:
            # pytorch 1.6 use true_divide instead of /=
            avg[k] = torch.true_divide(avg[k], num)
    print('Saving to {}'.format(args.dst_model))
    torch.save(avg, args.dst_model)

"""
if __name__ == '__main__':
    main()
