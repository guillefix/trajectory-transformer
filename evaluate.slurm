#!/bin/bash
##SBATCH --job-name=eval_demos_tw
#SBATCH --job-name=tt_eval
#SBATCH -A imi@cpu
##SBATCH --qos=qos_gpu-dev
##SBATCH --partition=gpu_p2
#SBATCH --ntasks=320
##SBATCH --ntasks=640
##SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
##SBATCH --qos=qos_cpu-dev
#SBATCH --qos=qos_cpu-t3 
##SBATCH --output=out.out
##SBATCH --error=err.err


module purge
#module load pytorch-gpu/py3/1.10.0
module load pytorch-cpu/py3/1.7.1

export ROOT_FOLDER=/gpfswork/rech/imi/usc19dv/captionRLenv/
export DATA_FOLDER=/gpfsscratch/rech/imi/usc19dv/data/UR5/
export PROCESSED_DATA_FOLDER=/gpfsscratch/rech/imi/usc19dv/data/UR5_processed/
export ROOT_DIR_MODEL=/gpfswork/rech/imi/usc19dv/mt-lightning/
export PRETRAINED_FOLDER=/gpfswork/rech/imi/usc19dv/mt-lightning/training/experiments/
export ROOT_GENERATED_DATA=/gpfsscratch/rech/imi/usc19dv/data/
export PROCESSED_GENERATED_DATA_FOLDER=/gpfsscratch/rech/imi/usc19dv/data/generated_data_processed/

srun --wait=0 -n 320 python3 inference_mpi.py --gpt_loadpath short1_big2 --base_filenames_file ${PROCESSED_DATA_FOLDER}base_filenames_single_objs_filtered.txt --save_eval_results --num_repeats 20
