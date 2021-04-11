
new_hparam_interval="True"

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
    *)
          # unknown option
    ;;
esac
done

#scratch_dir = "${a} ${b}"
#scratch_dir="/scratch/ethancab/res_full_${n_envs}/${datasets}/${models}"
scratch_dir="/Users/ethancaballero/zzz/debug/res_full_${n_envs}env/${datasets}/${models}"

echo "$new_hparam_interval"
echo "$n_envs"
echo "$models"
echo "$datasets"
echo "$scratch_dir"

if [[ -n $1 ]]; then
    echo "Argument not recognised"
    exit
fi

# 1. Create your environement locally
#module load python/3.8
#cd /home/ethancab
#source invariance_env/bin/activate
#cd /home/ethancab/research/invariance_unit_test/ib_irm/InvarianceUnitTests
#python scripts/sweep_outer.py --skip_confirmation True --new_hparam_interval True --n_envs 6 --models IB_ERM --datasets Example3 --scratch_dir $scratch_dir
python scripts/sweep_outer.py --skip_confirmation True --new_hparam_interval $new_hparam_interval --n_envs $n_envs --models $models --datasets $datasets --scratch_dir $scratch_dir --num_samples 2 --m_start 0 --m_end 1 --d_start 0 --d_end 1