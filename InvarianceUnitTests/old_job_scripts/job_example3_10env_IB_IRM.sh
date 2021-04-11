#!/bin/bash
#SBATCH --account=rrg-bengioy-ad         # Yoshua pays for your job
#SBATCH --cpus-per-task=40                # Ask for 40 CPUs
#SBATCH --gres=gpu:0                     # Ask for 0 GPU
#SBATCH --mem=752G                        # Ask for 752 GB of RAM
#SBATCH --time=75:00:00                   # The job will run for 3 hours
#SBATCH -o /scratch/ethancab/slurm-%j.out  # Write the log in $SCRATCH

# 1. Create your environement locally
module load python/3.8
cd /home/ethancab
source invariance_env/bin/activate
cd /home/ethancab/research/invariance_unit_test/ib_irm/InvarianceUnitTests
python scripts/sweep_outer.py --skip_confirmation True --n_envs 10 --models IB_IRM --datasets Example3 --scratch_dir "/scratch/ethancab/res_full_10env/example3_10env/IB_IRM"