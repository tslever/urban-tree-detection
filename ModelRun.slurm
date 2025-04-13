#!/bin/bash

#Submit this script with: sbatch myjob.slurm

#SBATCH --time=0:24:00   # job time limit
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # number of tasks per node
#SBATCH --cpus-per-task=40   # number of CPU cores per task
#SBATCH --gres=gpu:a6000:4   # gpu devices per node
#SBATCH --partition=gpu   # partition
#SBATCH --mem=768G   # memory
#SBATCH -J "ModelRun"   # job name
#SBATCH --mail-user=bdj9wf@virginia.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --account=sds_capstone_atashman   # allocation name



python -m venv env
source env/bin/activate
pip install tensorflow[and-cuda]==2.18.0
pip install numpy
pip install imageio
pip install rasterio
pip install geopandas
pip install h5py
pip install scipy
pip install tqdm
pip install scikit-image
pip install scikit-learn
pip install optuna
pip install matplotlib

python3 -m scripts.prepare /project/SDS/capstones_yang/urban-tree-detection-data output.hdf5 --data_mode full --augment
python3 -m scripts.train output.hdf5 logs
python3 -m scripts.tune output.hdf5 logs
python3 -m scripts.test output.hdf5 logs --center_crop --rearrange_channels
