#!/bin/bash

#SBATCH --job-name=trajectory-transformer_onoise2
##SBATCH --time=100:00:00
##SBATCH --time=20:00:00
#SBATCH --time=01:00:00
##SBATCH --qos=qos_gpu-t4
##SBATCH --qos=qos_gpu-t3
#SBATCH --qos=qos_gpu-dev
##SBATCH --ntasks=1 --cpus-per-task=24 --gres=gpu:4
##SBATCH --ntasks=1 --cpus-per-task=24 --gres=gpu:1
##SBATCH --nodes=4 --ntasks-per-node=4 --cpus-per-task=6 --gres=gpu:4
##SBATCH --nodes=2 --ntasks-per-node=4 --cpus-per-task=6 --gres=gpu:4
##SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=12 --gres=gpu:1
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=2 --gres=gpu:1
##SBATCH --nodes=1 --ntasks-per-node=8 --cpus-per-task=4 --gres=gpu:8
##SBATCH --partition=gpu_p2
#SBATCH -A imi@v100
#SBATCH -C v100-32g
#SBATCH --exclusive
#SBATCH --mail-type=ALL
#SBATCH --mail-user=guillefix@gmail.com

export MASTER_PORT=1234
slurm_nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
echo $slurm_nodes
export MASTER_ADDRESS=$(echo $slurm_nodes | cut -d' ' -f1)
echo $MASTER_ADDRESS

#module purge
#module load tensorflow-gpu/py3/2.6.0
#module load pytorch-gpu/py3/1.8.0
module purge
#module load pytorch-gpu/py3/1.10.0
module load pytorch-gpu/py3/1.8.1
#module load pytorch-cpu/py3/1.7.1
conda activate --stack /gpfsscratch/rech/imi/usc19dv/lib/awo

export GIT_PYTHON_REFRESH=quiet
export PYTHONPATH=/gpfswork/rech/imi/usc19dv/lib/python3.8/site-packages:/gpfsssd/scratch/rech/imi/usc19dv/lib/python3.7/site-packages:/gpfswork/rech/imi/usc19dv/trajectory-transformer


#export PYTHONPATH=${PYTHONPATH}:/gpfswork/rech/imi/usc19dv/lib/python3.7/site-packages:/gpfsscratch/rech/imi/usc19dv/lib/python3.7/site-packages/
set GIT_PYTHON_GIT_EXECUTABLE=/usr/bin/git
export GIT_PYTHON_REFRESH=quiet
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/linkhome/rech/genini01/usc19dv/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export C_INCLUDE_PATH=$C_INCLUDE_PATH:/gpfswork/rech/imi/usc19dv/glew-2.1.0/include:/usr/include
export CPATH=/gpfsscratch/rech/imi/usc19dv/lib/awo/include/
export LIBRARY_PATH=$LIBRARY_PATH:/gpfsscratch/rech/imi/usc19dv/lib/awo/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gpfsscratch/rech/imi/usc19dv/lib/awo/lib/

export ROOT_FOLDER=/gpfswork/rech/imi/usc19dv/captionRLenv/
export DATA_FOLDER=/gpfsscratch/rech/imi/usc19dv/data/UR5/
export PROCESSED_DATA_FOLDER=/gpfsscratch/rech/imi/usc19dv/data/UR5_processed/
export ROOT_DIR_MODEL=/gpfswork/rech/imi/usc19dv/mt-lightning/
export PRETRAINED_FOLDER=/gpfswork/rech/imi/usc19dv/mt-lightning/training/experiments/
export ROOT_GENERATED_DATA=/gpfsscratch/rech/imi/usc19dv/data/


#exp=$1
#srun python scripts/train.py --dataset LangRobot-v1 --batch_size 64 --subsampled_sequence_length 1 --n_epochs_ref 1000 --exp_name short1_big2 --n_layer 12 --n_embd 100 --continue_train
#srun python scripts/train.py --dataset LangRobot-v1 --batch_size 64 --subsampled_sequence_length 1 --n_epochs_ref 1000 --exp_name short1_big_bigN --n_layer 12 --n_embd 256 --learning_rate 1e-4 --embd_pdrop 0.0 --resid_pdrop 0.0 --attn_pdrop 0.0 --N 512
#srun python scripts/train.py --dataset LangRobot-v1 --batch_size 64 --subsampled_sequence_length 1 --n_epochs_ref 1000 --exp_name short1_big_bigN --n_layer 12 --n_embd 256 --learning_rate 1e-4 --embd_pdrop 0.0 --resid_pdrop 0.0 --attn_pdrop 0.0 --N 512 --continue_train
#srun python scripts/train.py --dataset LangRobot-v1 --batch_size 64 --subsampled_sequence_length 1 --n_epochs_ref 1000 --exp_name short1_big_bigN_nowd --n_layer 12 --n_embd 256 --learning_rate 1e-4 --embd_pdrop 0.0 --resid_pdrop 0.0 --attn_pdrop 0.0 --N 512 --weight_decay 0.0
#srun python scripts/train.py --dataset LangRobot-v1 --batch_size 128 --subsampled_sequence_length 2 --n_epochs_ref 1000 --exp_name long1_big_bigN --n_layer 8 --n_embd 128 --learning_rate 7e-5 --embd_pdrop 0.0 --resid_pdrop 0.0 --attn_pdrop 0.0 --N 512
srun /gpfslocalsup/pub/anaconda-py3/2021.05/envs/pytorch-1.8.1+py3.8.8-lts/bin/python3 scripts/train.py --dataset LangRobot-v1 --batch_size 128 --subsampled_sequence_length 3 --n_epochs_ref 1000 --exp_name testing_new_disc_long3 --n_layer 8 --n_embd 64 --learning_rate 5e-4 --embd_pdrop 0.0 --resid_pdrop 0.0 --attn_pdrop 0.0 --weight_decay 0.0 --N 100 --n_saves 500
#srun python scripts/train.py --dataset LangRobot-v1 --batch_size 128 --subsampled_sequence_length 3 --n_epochs_ref 1000 --exp_name testing_new_disc_long_onoise2 --n_layer 8 --n_embd 64 --learning_rate 1e-3 --embd_pdrop 0.0 --resid_pdrop 0.0 --attn_pdrop 0.0 --weight_decay 0.0 --N 512 --n_saves 500 --obs_noise 0.002
#srun python scripts/train.py --dataset LangRobot-v1 --batch_size 128 --subsampled_sequence_length 2 --n_epochs_ref 1000 --exp_name long1_big_bigN_smol --n_layer 8 --n_embd 128 --learning_rate 7e-5 --embd_pdrop 0.0 --resid_pdrop 0.0 --attn_pdrop 0.0 --N 512 --dataset_size 3000
#srun python scripts/train.py --dataset LangRobot-v0 --batch_size 64
#srun python scripts/train.py --dataset LangRobot-v1 --batch_size 64
