import os
import sys
import argparse
import numpy as np
import torch

"""
Launch sim_eval_policy.py with a list of paths to ckpt.pth files

Takes a path to a top-level directory containing many exp_*/ directories, and will find 
the latest checkpoint in all exp_*/ckpts/ sub-directories
"""

def main(top_dir):
    
    ckpt_num = 0
    ckpt_path_list = []

    for item_name in os.listdir(top_dir):
        
        if os.path.isfile(item_name): continue # skip non-directories

        # Check for exp_dir/eval/
        exp_dir = os.path.join(top_dir, item_name)
        ckpts_dir = os.path.join(exp_dir, "ckpts")
        if not os.path.exists(ckpts_dir): continue # skip directory if it doesn't contain ckpts/
        
        # Find latest checkpoint in ckpts_dir
        epoch = 0
        for item in os.listdir(ckpts_dir):
            if item.endswith('ckpt.pth'):
                epoch = max(epoch, int(item.split('_')[1]))

        ckpt_name = f"epoch_{epoch}_ckpt.pth"
        ckpt_path = os.path.join(ckpts_dir, ckpt_name)
        ckpt_path_list.append(ckpt_path) 
        ckpt_num +=1 
    
    print(f"Found {ckpt_num} checkpoints to eval")

    ckpt_paths_str = " ".join(ckpt_path_list)
    cmd = f"python sim_eval_policy.py --log_paths {ckpt_paths_str} --eval_train_and_test"
    os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("top_dir", default=None, help="""Filepath of top-level directory containing experiment directories""")
    args = parser.parse_args()
    main(args.top_dir)



