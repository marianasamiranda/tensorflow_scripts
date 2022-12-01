#!/bin/bash

#SBATCH -J tf-tensorflow      # Job name
#SBATCH -o tf-tensorflow.o%j  # Name of stdout output file
#SBATCH -e tf-tensorflow.e%j  # Name of stderr error file
#SBATCH -p rtx                # Queue (partition) name
#SBATCH -N 3                  # Total # of nodes (must be 1 for serial)
#SBATCH -n 3                  # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 00:30:00           # Run time (hh:mm:ss)

ibrun -n 1 -o 0 ./job_tensorflow_distributed.sh 0 & 
ibrun -n 1 -o 1 ./job_tensorflow_distributed.sh 1 & 
ibrun -n 1 -o 2 ./job_tensorflow_distributed.sh 2

