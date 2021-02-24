#!/bin/bash
#SBATCH --account=rrg-bengioy-ad         # Yoshua pays for your job
#SBATCH --cpus-per-task=6                # Ask for 40 CPUs
#SBATCH --gres=gpu:0                     # Ask for 0 GPU
#SBATCH --mem=32G                        # Ask for 752 GB of RAM
#SBATCH --time=3:00:00                   # The job will run for 3 hours
#SBATCH -o /scratch/ethancab/slurm-%j.out  # Write the log in $SCRATCH

#!/usr/bin/env bash

# max cpu is 40 and max mem is 752G

argparse(){
    argparser=$(mktemp 2>/dev/null || mktemp -t argparser)
    cat > "$argparser" <<EOF
from __future__ import print_function
import sys
import argparse
import os
class MyArgumentParser(argparse.ArgumentParser):
    def print_help(self, file=None):
        """Print help and exit with error"""
        super(MyArgumentParser, self).print_help(file=file)
        sys.exit(1)
parser = MyArgumentParser(prog=os.path.basename("$0"),
            description="""$ARGPARSE_DESCRIPTION""")
EOF

    # stdin to this function should contain the parser definition
    cat >> "$argparser"

    cat >> "$argparser" <<EOF
args = parser.parse_args()
for arg in [a for a in dir(args) if not a.startswith('_')]:
    key = arg.upper()
    value = getattr(args, arg, None)
    if isinstance(value, bool) or value is None:
        print('{0}="{1}";'.format(key, 'yes' if value else ''))
    elif isinstance(value, list):
        print('{0}=({1});'.format(key, ' '.join('"{0}"'.format(s) for s in value)))
    else:
        print('{0}="{1}";'.format(key, value))
EOF

    # Define variables corresponding to the options if the args can be
    # parsed without errors; otherwise, print the text of the error
    # message.
    if python "$argparser" "$@" &> /dev/null; then
        eval $(python "$argparser" "$@")
        retval=0
    else
        python "$argparser" "$@"
        retval=1
    fi

    rm "$argparser"
    return $retval
}

# print a script template when this script is executed
if [[ $0 == *argparse.bash ]]; then
    cat <<FOO
#!/usr/bin/env bash
source \$(dirname \$0)/argparse.bash || exit 1
argparse "\$@" <<EOF || exit 1
parser.add_argument('infile')
parser.add_argument('-o', '--outfile')
EOF
echo "INFILE: \${INFILE}"
echo "OUTFILE: \${OUTFILE}"
FOO
fi

argparse "$@" <<EOF || exit 1
parser.add_argument('--m_start', default=0, type=int,
                    help='')
parser.add_argument('--m_end', default=20, type=int,
                    help='')

parser.add_argument('--d_start', default=0, type=int,
                    help='')
parser.add_argument('--d_end', default=50, type=int,
                    help='')
EOF

#echo the answer: $M_START
#echo the answer: $M_END

python scripts/sweep.py --skip_confirmation True --models ERM --datasets Example2 --num_samples 2 --m_start $M_START --m_end $M_END --d_start 0 --d_end 50