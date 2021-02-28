CUDA_VISIBLE_DEVICES=5 nohup python train_with_masks_adv_g_less.py --max_iter 50000 --val_freq 200 --exp_name sngan_cifar10_adv_g_3 --init-path initial_weights --gamma 0.01 --step 1 --rewind-path sngan_less/Model/checkpoint_3.pth > 0114_adv_g_3_gpu5.out &

CUDA_VISIBLE_DEVICES=6 nohup python train_with_masks_adv_gd_less.py --max_iter 50000 --val_freq 200 --exp_name sngan_cifar10_adv_gd_0 --init-path initial_weights --gamma 0.01 --step 1 --rewind-path sngan_less/Model/checkpoint_0.pth > 0114_adv_gd_0_gpu7.out &

CUDA_VISIBLE_DEVICES=7 nohup python train_with_masks_adv_gd_less.py --max_iter 50000 --val_freq 200 --exp_name sngan_cifar10_adv_gd_1 --init-path initial_weights --gamma 0.01 --step 1 --rewind-path sngan_less/Model/checkpoint_1.pth > 0114_adv_gd_1_gpu0.out &