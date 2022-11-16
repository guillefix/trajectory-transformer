#!/bin/bash
##SBATCH --job-name=eval_demos_tw
#SBATCH --job-name=tt_eval
##SBATCH -A imi@cpu
#SBATCH -A imi@v100
#SBATCH -C v100-32g
##SBATCH --qos=qos_gpu-dev
##SBATCH --qos=qos_gpu-t3
#SBATCH --qos=qos_gpu-t4
##SBATCH --qos=qos_cpu-dev
##SBATCH --qos=qos_cpu-t3 
##SBATCH --partition=gpu_p2
#SBATCH --ntasks=40
##SBATCH --ntasks=2
##SBATCH --ntasks=160
##SBATCH --ntasks=640
#SBATCH --gres=gpu:4
##SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
##SBATCH --time=20:00:00
#SBATCH --time=100:00:00
##SBATCH --time=2:00:00
##SBATCH --output=out.out
##SBATCH --error=err.err


module purge
#module load pytorch-gpu/py3/1.10.0
module load pytorch-gpu/py3/1.8.1
#module load pytorch-cpu/py3/1.7.1
conda activate --stack /gpfsscratch/rech/imi/usc19dv/lib/awo

export GIT_PYTHON_REFRESH=quiet
#export PYTHONPATH=${PYTHONPATH}:/gpfswork/rech/imi/usc19dv/lib/python3.7/site-packages:/gpfsscratch/rech/imi/usc19dv/lib/python3.7/site-packages/:/gpfslocalsup/pub/anaconda-py3/2020.02/envs/pytorch-cpu-1.7.1/lib/python3.7/site-packages/:/gpfsssd/scratch/rech/imi/usc19dv/lib/python3.7/site-packages
#export PYTHONPATH=${PYTHONPATH}:/gpfswork/rech/imi/usc19dv/lib/python3.7/site-packages:/gpfsscratch/rech/imi/usc19dv/lib/python3.7/site-packages/:/gpfslocalsup/pub/anaconda-py3/2021.05/envs/pytorch-gpu-1.10.0+py3.9.7/lib/python3.9/site-packages/torch/
#export PYTHONPATH=${PYTHONPATH}:/gpfswork/rech/imi/usc19dv/lib/python3.9/site-packages:/gpfslocalsup/pub/anaconda-py3/2021.05/envs/pytorch-gpu-1.10.0+py3.9.7/lib/python3.9/site-packages/
#export PYTHONPATH=/gpfswork/rech/imi/usc19dv/lib/python3.9/site-packages:/gpfslocalsup/pub/anaconda-py3/2021.05/envs/pytorch-gpu-1.10.0+py3.9.7/lib/python3.9/site-packages/
#export PYTHONPATH=/gpfswork/rech/imi/usc19dv/lib/python3.7/site-packages
export PYTHONPATH=/gpfswork/rech/imi/usc19dv/lib/python3.8/site-packages:/gpfsssd/scratch/rech/imi/usc19dv/lib/python3.7/site-packages
export CPATH=/gpfsscratch/rech/imi/usc19dv/lib/awo/include/
export LIBRARY_PATH=$LIBRARY_PATH:/gpfsscratch/rech/imi/usc19dv/lib/awo/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gpfsscratch/rech/imi/usc19dv/lib/awo/lib/
export ROOT_FOLDER=/gpfswork/rech/imi/usc19dv/RobotLangEnv/
export DATA_FOLDER=/gpfsscratch/rech/imi/usc19dv/data/UR5/
export PROCESSED_DATA_FOLDER=/gpfsscratch/rech/imi/usc19dv/data/UR5_processed/
export ROOT_DIR_MODEL=/gpfswork/rech/imi/usc19dv/mt-lightning/
export PRETRAINED_FOLDER=/gpfswork/rech/imi/usc19dv/mt-lightning/training/experiments/
export ROOT_GENERATED_DATA=/gpfsscratch/rech/imi/usc19dv/data/
export PROCESSED_GENERATED_DATA_FOLDER=/gpfsscratch/rech/imi/usc19dv/data/generated_data_processed/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/linkhome/rech/genini01/usc19dv/.mujoco/mujoco210/bin

#srun --wait=0 -n 80 python3 inference_mpi.py --dataset LangRobot-v1 --gpt_loadpath testing_new_disc_long6_bigger_L6_h64_l1e5 --base_filenames_file base_filenames_single_objs_filtered.txt --save_eval_results --num_repeats 20 --suffix new --savepath /gpfswork/rech/imi/usc19dv/trajectory-transformer/
srun --wait=0 -n 40 python3 inference_mpi.py --dataset LangRobot-v1 --gpt_loadpath testing_new_disc_long6_bigger_L6_h256_l1e5 --base_filenames_file base_filenames_single_objs_filtered.txt --save_eval_results --num_repeats 20 --suffix new_h256
#srun --wait=0 -n 2 python3 inference_mpi.py --dataset LangRobot-v1 --gpt_loadpath testing_new_disc_long6_bigger_L6_h64_l1e5 --base_filenames_file base_filenames_single_objs_filtered.txt --save_eval_results --num_repeats 20 --max_episode_length 10 --savepath results
#srun --wait=0 -n 2 python3 inference_mpi.py --dataset LangRobot-v1 --gpt_loadpath short1_big2 --base_filenames_file base_filenames_single_objs_filtered.txt --save_eval_results --num_repeats 20
#srun --wait=0 -n 2 python3 inference_mpi.py --dataset LangRobot-v1 --gpt_loadpath testing_new_disc_long6_bigger_L6_h64_l1e5 --base_filenames_file base_filenames_single_objs_filtered.txt --save_eval_results --num_repeats 20
