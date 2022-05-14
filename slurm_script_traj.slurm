#!/bin/bash

#SBATCH --job-name=trajectory-transformer
##SBATCH --time=100:00:00
#SBATCH --time=20:00:00
##SBATCH --time=01:00:00
##SBATCH --qos=qos_gpu-t4
#SBATCH --qos=qos_gpu-t3
##SBATCH --qos=qos_gpu-dev
##SBATCH --ntasks=1 --cpus-per-task=24 --gres=gpu:4
##SBATCH --ntasks=1 --cpus-per-task=24 --gres=gpu:1
##SBATCH --nodes=4 --ntasks-per-node=4 --cpus-per-task=6 --gres=gpu:4
##SBATCH --nodes=2 --ntasks-per-node=4 --cpus-per-task=6 --gres=gpu:4
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=12 --gres=gpu:1
##SBATCH --nodes=1 --ntasks-per-node=8 --cpus-per-task=4 --gres=gpu:8
##SBATCH --partition=gpu_p2
#SBATCH -A imi@gpu
#SBATCH -C v100-32g
#SBATCH --exclusive

export MASTER_PORT=1234
slurm_nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
echo $slurm_nodes
export MASTER_ADDRESS=$(echo $slurm_nodes | cut -d' ' -f1)
echo $MASTER_ADDRESS

#module load tensorflow-gpu/py3/2.6.0
module load pytorch-gpu/py3/1.8.0

export PYTHONPATH=${PYTHONPATH}:/gpfswork/rech/imi/usc19dv/trajectory-transformer
set GIT_PYTHON_GIT_EXECUTABLE=/usr/bin/git
export GIT_PYTHON_REFRESH=quiet
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/linkhome/rech/genini01/usc19dv/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export C_INCLUDE_PATH=$C_INCLUDE_PATH:/gpfswork/rech/imi/usc19dv/glew-2.1.0/include:/usr/include

#exp=$1
srun python scripts/train.py --dataset LangRobot-v1 --batch_size 32
#srun python scripts/train.py --dataset LangRobot-v0 --batch_size 64
#srun python scripts/train.py --dataset LangRobot-v1 --batch_size 64
