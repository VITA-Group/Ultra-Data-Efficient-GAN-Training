CUDA_VISIBLE_DEVICES=2 nohup python train_adv_g_less.py --max_iter 50000 --val_freq 200 --exp_name sngan_cifar10_adv_g --init-path initial_weights --gamma 0.01 --step 1 > 0112_adv_g_gpu2.out &

CUDA_VISIBLE_DEVICES=3 nohup python train_adv_d_less.py --max_iter 50000 --val_freq 200 --exp_name sngan_cifar10_adv_d --init-path initial_weights --gamma 0.01 --step 1 > 0112_adv_d_gpu3.out &

CUDA_VISIBLE_DEVICES=4 nohup python train_adv_gd_less.py --max_iter 50000 --val_freq 200 --exp_name sngan_cifar10_adv_gd --init-path initial_weights --gamma 0.01 --step 1 > 0112_adv_gd_gpu4.out &