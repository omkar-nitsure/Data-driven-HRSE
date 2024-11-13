#!/usr/bin/bash

CUDA_VISIBLE_DEVICES=3 python3 train.py \
	--n_epochs_fr 50 \
	--n_epochs_fc 50 \
	--output_dir /Accented_ASR/DeepFreq/models/SNR_30 \