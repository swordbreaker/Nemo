# NVIDIA NeMo for sciCORE

NeMo: [https://github.com/NVIDIA/NeMo](https://github.com/NVIDIA/NeMo)

I only tested the TTS part.

## Installation worked for me

```bash
ml CUDA/11.7.0
ml Miniconda2/4.3.30
ml intel/2022.00

conda create --name nemo python==3.9 //do not use 3.8 the readme is lying :(
source activate nemo
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c conda-forge libsndfile
conda install -c conda-forge ffmpeg

git clone https://github.com/swordbreaker/Nemo.git
cd NeMo
pip install nemo_toolkit['all']

conda remove numba
pip uninstall numba
conda install -c conda-forge numba
conda install -c conda-forge librosa
conda install -c conda-forge llvmlite
conda install scikit-learny
pip install pytorch-lightning==1.9.0
pip install transformers -y
conda install -c conda-forge youtokentome -y
conda install -c conda-forge pyannote.core -y
conda install -c conda-forge editdistance -y
conda install -c conda-forge jiwer -y
conda install -c conda-forge pynini -y
```

## Create a data set
See /data/swissDial for an example.

The format is described here: https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/tts/datasets.html#

## Train vits
I copied the examples into the train directory and altered the config files.

```batch 
sbatch train_vits_tts.sh
```

the content of the sh file:
```batch
#!/bin/bash
#SBATCH --job-name=train_vits_nemo     #Name of your job
#SBATCH --cpus-per-task=4    #Number of cores to reserve
#SBATCH --mem-per-cpu=32G     #Amount of RAM/core to reserve
#SBATCH --time=4-00:00:00      #Maximum allocated time
#SBATCH --qos=1week       #Selected queue to allocate your job
#SBATCH --partition=a100
#SBATCH --gres=gpu:4
#SBATCH --output=log/train_vits_tts.o%j   #Path and name to the file for the STDOUT
#SBATCH --error=log/train_vits_tts.o%j    #Path and name to the file for the STDERR

ml CUDA/11.7.0
ml Miniconda2/4.3.30
ml intel/2022.00
source activate nemo

export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
srun python train/tts/vits.py
```