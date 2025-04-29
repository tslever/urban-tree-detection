#!/bin/bash

#Submit this script with: sbatch myjob.slurm

#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu
#SBATCH --constraint="a100|a40|a6000|v100"
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH -J "TESTING only"
#SBATCH --mail-user=bdj9wf@virginia.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --account=sds_capstone_atashman


source /project/SDS/capstones_yang/urban-tree-detection/env/bin/activate

python3 -m scripts.test fullaug.hdf5 logs --center_crop --rearrange_channels
