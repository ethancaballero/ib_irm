#!/bin/bash
#SBATCH --account=rrg-bengioy-ad         # Yoshua pays for your job
#SBATCH --cpus-per-task=6                # Ask for 40 CPUs
#SBATCH --gres=gpu:0                     # Ask for 0 GPU
#SBATCH --mem=32G                        # Ask for 752 GB of RAM
#SBATCH --time=1:00:00                   # The job will run for 3 hours
#SBATCH -o /scratch/ethancab/slurm-%j.out  # Write the log in $SCRATCH

# 1. Create your environement locally
module load python/3.8
cd /home/ethancab
source invariance_env/bin/activate
cd /home/ethancab/research/invariance_unit_test/ib_irm/InvarianceUnitTests

#python scripts/sweep.py --models ERM IRMv1 IB_ERM IB_IRM --datasets Example2 --num_samples 2 --num_data_seeds 2 --num_model_seeds 2

#python scripts/sweep.py --skip_confirmation True --models ERM --datasets Example2 --num_samples 2 --num_data_seeds 2 --num_model_seeds 2
python scripts/sweep.py --skip_confirmation True --models ERM --datasets Example2 --num_samples 2 --m_start 0 --m_end 1 --d_start 0 --d_end 1

# 5. Copy whatever you want to save on $SCRATCH
# cp $SLURM_TMPDIR/<to_save> $SCRATCH
cp -R /home/ethancab/research/invariance_unit_test/ib_irm/InvarianceUnitTests/results $SCRATCH