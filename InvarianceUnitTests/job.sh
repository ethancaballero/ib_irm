#!/bin/bash
#SBATCH --account=rrg-bengioy-ad         # Yoshua pays for your job
#SBATCH --cpus-per-task=40                # Ask for 40 CPUs
#SBATCH --gres=gpu:0                     # Ask for 0 GPU
#SBATCH --mem=752G                        # Ask for 752 GB of RAM
#SBATCH --time=75:00:00                   # The job will run for 3 hours

new_hparam_interval="True" # this just determines whether irm_loss is normalized by number of envs
ib_bool="False" # whether or not ib_on is used as a hparam
mod_folder_name=""
d_start="0"
d_end="20"
m_start="0"
m_end="50"

user=$USER
echo "$user"

#n_envs="init"
#models="init"
#datasets="init"

for i in "$@"
do
case $i in
    -nhi=*|--new_hparam_interval=*)
    new_hparam_interval="${i#*=}"
    shift # past argument=value
    ;;
    -ibb=*|--ib_bool=*)
    ib_bool="${i#*=}"
    shift # past argument=value
    ;;
    -ne=*|--n_envs=*)
    n_envs="${i#*=}"
    shift # past argument=value
    ;;
    -m=*|--models=*)
    models="${i#*=}"
    shift # past argument=value
    ;;
    -d=*|--datasets=*)
    datasets="${i#*=}"
    shift # past argument=value
    ;;
    -ds=*|--d_start*)
    d_start="${i#*=}"
    shift # past argument=value
    ;;
    -de=*|--d_end*)
    d_end="${i#*=}"
    shift # past argument=value
    ;;
    -ms=*|--m_start*)
    m_start="${i#*=}"
    shift # past argument=value
    ;;
    -me=*|--m_end*)
    m_end="${i#*=}"
    shift # past argument=value
    ;;
    -mfn=*|--mod_folder_name=*)
    mod_folder_name="${i#*=}"
    shift # past argument=value
    ;;
    *)
          # unknown option
    ;;
esac
done

#scratch_dir = "${a} ${b}"
scratch_dir="/scratch/${user}/res_full_${n_envs}${mod_folder_name}/${datasets}/${models}"
#scratch_dir="/Users/ethancaballero/zzz/debug/res_full_${n_envs}env/${datasets}/${models}"

cd1="/home/${user}"
cd2="/home/${user}/research/invariance_unit_test/ib_irm/InvarianceUnitTests"

echo "$new_hparam_interval"
echo "$ib_bool"
echo "$n_envs"
echo "$models"
echo "$datasets"
echo "$scratch_dir"

if [[ -n $1 ]]; then
    echo "Argument not recognised"
    exit
fi

# 1. Create your environement locally
module load python/3.8
cd $cd1
source invariance_env/bin/activate
cd $cd2
python scripts/sweep_outer.py --skip_confirmation True --new_hparam_interval $new_hparam_interval --ib_bool $ib_bool --n_envs $n_envs --models $models --datasets $datasets --d_start $d_start --d_end $d_end --m_start $m_start --m_end $m_end --scratch_dir $scratch_dir
#python scripts/sweep_outer.py --skip_confirmation True --new_hparam_interval $new_hparam_interval --n_envs $n_envs --models $models --datasets $datasets --scratch_dir $scratch_dir --num_samples 2 --m_start 0 --m_end 1 --d_start 0 --d_end 1