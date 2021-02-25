#!/bin/bash


#python scripts/sweep.py --models ERM IRMv1 IB_ERM IB_IRM --datasets Example2 --num_samples 2 --num_data_seeds 2 --num_model_seeds 2

#python scripts/sweep.py --skip_confirmation True --models ERM --datasets Example2 --num_samples 2 --num_data_seeds 2 --num_model_seeds 2
#python scripts/sweep.py --skip_confirmation True --models ERM --datasets Example2 --num_samples 2 --m_start 0 --m_end 1 --d_start 0 --d_end 1
python scripts/sweep.py --skip_confirmation True --models ERM --datasets Example2 --num_samples 2 --m_start $1 --m_end $2 --d_start 0 --d_end 1
