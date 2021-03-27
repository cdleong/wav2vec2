#!/bin/bash
pip install git+https://github.com/huggingface/datasets.git
pip install torchaudio
pip install librosa
pip install jiwer
pip install wandb 
pip install git+https://github.com/huggingface/transformers.git
pip install ftfy
export WANDB_ENTITY=wandb
export WANDB_PROJECT=huggingface
export WANDB_LOG_MODEL=true
wandb login
