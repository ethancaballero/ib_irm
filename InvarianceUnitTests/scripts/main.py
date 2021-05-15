# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import hashlib
import pprint
import json
import git
import os
import datasets
import models
import utils
import torch


def run_experiment(args):
    # build directory name
    commit = git.Repo(search_parent_directories=True).head.object.hexsha[:10]
    if args["scratch_dir"] == "None":
        results_dirname = os.path.join(args["output_dir"], commit + "/")
    else:
        #scratch = '/scratch/ethancab'
        results_dirname = args["scratch_dir"] + "/" + args["output_dir"] + "/" + commit + "/"
    os.makedirs(results_dirname, exist_ok=True)

    # build file name
    md5_fname = hashlib.md5(str(args).encode('utf-8')).hexdigest()
    results_fname = os.path.join(results_dirname, md5_fname + ".jsonl")
    results_file = open(results_fname, "w")

    utils.set_seed(args["data_seed"])
    dataset = datasets.DATASETS[args["dataset"]](
        args=args,
        dim_inv=args["dim_inv"],
        dim_spu=args["dim_spu"],
        n_envs=args["n_envs"]
    )

    # Oracle trained on test mode (scrambled)
    train_split = "train" if args["model"] != "Oracle" else "test"

    # sample the envs
    envs = {}
    for key_split, split in zip(("train", "validation", "test", "test_peak"),
                                (train_split, train_split, "test", "test")):
        envs[key_split] = {"keys": [], "envs": []}
        if key_split == "test_peak":
            num_samples = args["num_samples_test_peak"]
        else:
            num_samples = args["num_samples"]

        for env in dataset.envs:
            envs[key_split]["envs"].append(dataset.sample(
                n=num_samples,
                env=env,
                split=split)
            )
            envs[key_split]["keys"].append(env)

    # offsetting model seed to avoid overlap with data_seed
    utils.set_seed(args["model_seed"] + 1000)

    # selecting model
    args["num_dim"] = args["dim_inv"] + args["dim_spu"]
    model = models.MODELS[args["model"]](
        args=args,
        in_features=args["num_dim"],
        out_features=1,
        bias=args["bias"],
        task=dataset.task,
        hparams=args["hparams"]
    )

    # update this field for printing purposes
    args["hparams"] = model.hparams

    # fit the dataset
    model.fit(
        envs=envs,
        num_iterations=args["num_iterations"],
        callback=args["callback"])

    print([_ for _ in model.parameters()])

    if args["bias"]:
        edge = 1
    else:
        edge = 0

    params = [_ for _ in model.parameters()][-(1+edge)]
    params_inv = params[0][:args["dim_inv"]]
    params_spur = params[0][args["dim_inv"]:]

    norm = torch.norm(params_spur)/torch.norm(params)
    print("norm:", norm.data.item())
    args["norm"] = norm.data.item()

    # compute the train, validation and test errors
    for split in ("train", "validation", "test", "test_peak"):
        key = "error_" + split
        for k_env, env in zip(envs[split]["keys"], envs[split]["envs"]):
            if args["model"] == "IB_IRM_NN":
                args[key + "_" +
                    k_env] = utils.compute_error_nonlinear(model, *env)
            else:
                args[key + "_" +
                    k_env] = utils.compute_error(model, *env)

    # write results
    results_file.write(json.dumps(args))
    results_file.close()
    return args

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Synthetic invariances')
    parser.add_argument('--model', type=str, default="ERM")
    parser.add_argument('--num_iterations', type=int, default=10000)
    parser.add_argument('--hparams', type=str, default="default")
    parser.add_argument('--dataset', type=str, default="Example1")
    parser.add_argument('--dim_inv', type=int, default=5)
    parser.add_argument('--dim_spu', type=int, default=5)
    parser.add_argument('--n_envs', type=int, default=3)
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--data_seed', type=int, default=0)
    parser.add_argument('--model_seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default="results")
    parser.add_argument('--scratch_dir', type=str, default="None")
    #parser.add_argument('--callback', action='store_true')
    parser.add_argument('--callback', type=str2bool, default=False)
    
    parser.add_argument('--bias', type=str2bool, default=True)

    #example 2 mods
    parser.add_argument('--snr_fg', type=float, default=1e-2)
    parser.add_argument('--snr_bg', type=float, default=1)
    parser.add_argument('--inv_var', type=float, default=10)
    parser.add_argument('--spur_var', type=float, default=10)

    parser.add_argument('--num_samples_test_peak', type=int, default=20)
    args = parser.parse_args()

    pprint.pprint(run_experiment(vars(args)))
