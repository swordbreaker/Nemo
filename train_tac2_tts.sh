#!/bin/bash

#SBATCH --job-name=train_mixer_tts     #Name of your job
#SBATCH --cpus-per-task=4    #Number of cores to reserve
#SBATCH --mem-per-cpu=32G     #Amount of RAM/core to reserve
#SBATCH --time=4-00:00:00      #Maximum allocated time
#SBATCH --qos=1week       #Selected queue to allocate your job
#SBATCH --partition=a100
#SBATCH --gres=gpu:4
#SBATCH --output=log/train_mixer_tts.o%j   #Path and name to the file for the STDOUT
#SBATCH --error=log/train_mixer_tts.o%j    #Path and name to the file for the STDERR

ml CUDA/11.7.0
ml Miniconda2/4.3.30
ml intel/2022.00
source activate nemo

export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
srun python train/tts/tacotron2.py --config-name tacotron2.yaml
