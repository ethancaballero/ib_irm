# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import main
import random
import models
import datasets
import argparse
import getpass
import subprocess

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
    #parser.add_argument('--callback', action='store_true')
    #parser.add_argument('--cluster', action="store_true")
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

    parser.add_argument('--new_hparam_interval', type=str2bool, default=False)

    parser.add_argument('--ib_lambda_l', type=float, default=-.05)
    parser.add_argument('--ib_lambda_r', type=float, default=0.0)
    parser.add_argument('--irm_lambda_l', type=float, default=-3.0)
    parser.add_argument('--irm_lambda_r', type=float, default=-.3)

    args = vars(parser.parse_args())

    try:
        import submitit
    except:
        args["cluster"] = False
        pass

    all_jobs = []
    if len(args["models"]):
        model_lists = args["models"]
    else:
        model_lists = models.MODELS.keys()
    if len(args["datasets"]):
        dataset_lists = args["datasets"]
    else:
        dataset_lists = datasets.DATASETS.keys()

    for model in model_lists:
        for dataset in dataset_lists:
            for data_seed in range(args["d_start"], args["d_end"]):
                for model_seed in range(args["m_start"], args["m_end"]):
                    train_args = {
                        "model": model,
                        "num_iterations": args["num_iterations"],
                        "hparams": "random" if model_seed else "default",
                        "dataset": dataset,
                        "dim_inv": args["dim_inv"],
                        "dim_spu": args["dim_spu"],
                        "n_envs": args["n_envs"],
                        "num_samples": args["num_samples"],
                        "data_seed": data_seed,
                        "model_seed": model_seed,
                        "output_dir": args["output_dir"],
                        "scratch_dir": args["scratch_dir"],
                        "callback": args["callback"],

                        "bias": args["bias"],

                        "snr_fg": args["snr_fg"],
                        "snr_bg": args["snr_bg"],
                        "inv_var": args["inv_var"],
                        "spur_var": args["spur_var"],

                        "new_hparam_interval": args["new_hparam_interval"]
                    }

                    all_jobs.append(train_args)

    random.shuffle(all_jobs)

    print("Launching {} jobs...".format(len(all_jobs)))

    if args["cluster"]:
        executor = submitit.SlurmExecutor(
            folder=f"/checkpoint/{getpass.getuser()}/submitit/")
        executor.update_parameters(
            time=3*24*60,
            gpus_per_node=0,
            array_parallelism=args["jobs_cluster"],
            cpus_per_task=1,
            comment="",
            partition="learnfair")

        executor.map_array(main.run_experiment, all_jobs)
    else:
        #"""
        for job in all_jobs:
            print(main.run_experiment(job))
        #"""
        """
        cmd = "python3 scripts/main.py"
        if not args["skip_confirmation"]:
            ask_for_confirmation()
        for job in all_jobs:
            for j in job:
                cmd += " --" + j + " " + str(job[j])
            subprocess.Popen(cmd, shell=True)
        #"""
