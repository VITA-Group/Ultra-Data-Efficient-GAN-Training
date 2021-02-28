#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=4
python test.py \
--img_size 32 \
--model sngan_cifar10 \
--latent_dim 128 \
--gf_dim 256 \
--g_spectral_norm False \
--load_path pre_trained/sngan_cifar10.pth \
--exp_name test_sngan_cifar10