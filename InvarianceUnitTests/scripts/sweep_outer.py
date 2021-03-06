# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import main
import random
import models
import datasets
import argparse
import getpass
import subprocess
import copy
import os

def ask_for_confirmation():
    response = input('Are you sure? (y/n) ')
    if not response.lower().strip()[:1] == "y":
        print('Nevermind!')
        exit(0)

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Synthetic invariances')
    parser.add_argument('--models', nargs='+', default=[])
    parser.add_argument('--num_iterations', type=int, default=10000)
    parser.add_argument('--hparams', type=str, default="default")
    parser.add_argument('--datasets', nargs='+', default=[])
    parser.add_argument('--dim_inv', type=int, default=5)
    parser.add_argument('--dim_spu', type=int, default=5)
    parser.add_argument('--n_envs', type=int, default=3)
    parser.add_argument('--num_samples', type=int, default=10000)
    #parser.add_argument('--num_data_seeds', type=int, default=50)
    #parser.add_argument('--num_model_seeds', type=int, default=20)

    parser.add_argument('--m_start', default=0, type=int,
                        help='')
    parser.add_argument('--m_end', default=20, type=int,
                        help='')
    parser.add_argument('--d_start', default=0, type=int,
                        help='')
    parser.add_argument('--d_end', default=50, type=int,
                        help='')
    
    parser.add_argument('--output_dir', type=str, default="results")
    parser.add_argument('--scratch_dir', type=str, default="None")
    parser.add_argument('--callback', type=str2bool, default=False)
    parser.add_argument('--cluster', type=str2bool, default=False)
    parser.add_argument('--jobs_cluster', type=int, default=512)
    parser.add_argument('--skip_confirmation', type=str2bool, default=False)

    parser.add_argument('--bias', type=str2bool, default=True)

    #example 2 mods
    parser.add_argument('--snr_fg', type=float, default=1e-2)
    parser.add_argument('--snr_bg', type=float, default=1)
    parser.add_argument('--inv_var', type=float, default=10)
    parser.add_argument('--spur_var', type=float, default=10)

    args = vars(parser.parse_args())

    #"""
    cmd = "python3 scripts/sweep.py"
    if not args["skip_confirmation"]:
        ask_for_confirmation()
    args_copy = copy.deepcopy(args)
    for i in range(args['d_start'], args['d_end']):
        for j in args:
            if j == 'm_start': 
                cmd += " --" + 'm_start' + " " + str(args['m_start'])
            elif j == 'm_end': 
                cmd += " --" + 'm_end' + " " + str(args['m_end'])
            elif j == 'd_start':
                cmd += " --" + 'd_start' + " " + str(i)
            elif j == 'd_end':
                cmd += " --" + 'd_end' + " " + str(i+1)            
            elif j == 'models':
                cmd += " --" + 'models'
                for a in args[j]:
                    cmd += " " + str(a)
            elif j == 'datasets':
                cmd += " --" + 'datasets'
                for a in args[j]:
                    cmd += " " + str(a)
            else:
                cmd += " --" + j + " " + str(args[j])
        #subprocess.Popen(cmd, shell=True)
        os.system(cmd)
    #"""
