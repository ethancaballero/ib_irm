#!/bin/bash
#SBATCH --account=rrg-bengioy-ad         # Yoshua pays for your job
#SBATCH --cpus-per-task=6                # Ask for 40 CPUs
#SBATCH --gres=gpu:0                     # Ask for 0 GPU
#SBATCH --mem=32G                        # Ask for 752 GB of RAM
#SBATCH --time=1:00:00                   # The job will run for 3 hours
#SBATCH -o /scratch/ethancab/slurm-%j.out  # Write the log in $SCRATCH

#!/usr/bin/env bash

# max cpu is 40 and max mem is 752G


#python scripts/sweep.py --skip_confirmation True --models ERM --datasets Example2 --num_samples 2 --m_start $M_START --m_end $M_END --d_start 0 --d_end 50
python scripts/sweep.py --skip_confirmation True --models ERM --datasets Example2 --num_samples 2 --m_start $1 --m_end $2 --d_start 0 --d_end 1