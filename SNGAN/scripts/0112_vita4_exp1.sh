CUDA_VISIBLE_DEVICES=5 nohup python train_with_masks_adv_g_less.py --max_iter 50000 --val_freq 200 --exp_name sngan_cifar10_adv_g_0 --init-path initial_weights --gamma 0.01 --step 1 --rewind-path sngan_less/Model/checkpoint_0.pth > 0112_adv_g_0_gpu5.out &

CUDA_VISIBLE_DEVICES=6 nohup python train_with_masks_adv_d_less.py --max_iter 50000 --val_freq 200 --exp_name sngan_cifar10_adv_d_0 --init-path initial_weights --gamma 0.01 --step 1 --rewind-path sngan_less/Model/checkpoint_0.pth > 0112_adv_d_0_gpu6.out &

CUDA_VISIBLE_DEVICES=7 nohup python train_with_masks_adv_gd_less.py --max_iter 50000 --val_freq 200 --exp_name sngan_cifar10_adv_gd_0 --init-path initial_weights --gamma 0.01 --step 1 --rewind-path sngan_less/Model/checkpoint_0.pth > 0112_adv_gd_0_gpu7.out &