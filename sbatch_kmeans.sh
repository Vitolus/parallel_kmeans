#!/bin/sh
#SBATCH --job-name=p_kmeans
#SBATCH --time=240
srun parallel_kmeans 20