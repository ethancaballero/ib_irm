#!/bin/bash

: '
for {5..11..2}; do
    echo $i
done
'

#for i in `seq 0 1 19`; do
for i in `seq 0 1 2`; do
    sbatch job_node.sh --m_start $i --m_end $(( $i + 1 ))
    #sbatch job_node.sh $i $(( $i + 1 ))
    #echo $i + 1
    #echo $(( $i + 2 ))
done

: '
parser.add_argument('--m_start', default=0, type=int,
                    help='')
parser.add_argument('--m_end', default=20, type=int,
                    help='')
parser.add_argument('--d_start', default=0, type=int,
                    help='')
parser.add_argument('--d_end', default=50, type=int,
                    help='')
'