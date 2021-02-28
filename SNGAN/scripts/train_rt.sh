#!/bin/bash
percent=$1
start_gpu=$2
end_gpu=$3

for i in $(seq ${start_gpu} ${end_gpu});
do
    CUDA_VISIBLE_DEVICES=$i nohup python train_rewind.py -gen_bs 128 -dis_bs 64 --dataset cifar10 --img_size 32 --max_iter 50000 --model sngan_cifar10 --latent_dim 128 --gf_dim 256 --df_dim 128 --g_spectral_norm False --d_spectral_norm True --g_lr 0.0002 --d_lr 0.0002 --beta1 0.0 --beta2 0.9 --init_type xavier_uniform --n_critic 5 --val_freq 20 --exp_name sngan_cifar10_random_ticket_${i} --percent ${percent} --load_path logs/sngan_cifar10_2020_07_04_03_54_30 --init-path initial_weights_v2 > $i.out &
done