#!/bin/bash
#SBATCH --account=rrg-bengioy-ad         # Yoshua pays for your job
#SBATCH --cpus-per-task=6                # Ask for 6 CPUs
#SBATCH --gres=gpu:1                     # Ask for 1 GPU
#SBATCH --mem=32G                        # Ask for 32 GB of RAM
#SBATCH --time=3:00:00                   # The job will run for 3 hours
#SBATCH -o /scratch/ethancab/slurm-%j.out  # Write the log in $SCRATCH

cd $SLURM_TMPDIR

echo $1
echo $2

echo $1 >> myfile.txt

# 5. Copy whatever you want to save on $SCRATCH
cp $SLURM_TMPDIR/myfile.txt $SCRATCH