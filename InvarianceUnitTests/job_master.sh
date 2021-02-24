#!/bin/bash

: '
for {5..11..2}; do
    echo $i
done
'

for i in `seq 0 1 19`; do
    sbatch job_node.sh 
done

    parser.add_argument('--m_start', default=0, type=int,
                        help='')
    parser.add_argument('--m_end', default=20, type=int,
                        help='')
    parser.add_argument('--d_start', default=0, type=int,
                        help='')
    parser.add_argument('--d_end', default=50, type=int,
                        help='')