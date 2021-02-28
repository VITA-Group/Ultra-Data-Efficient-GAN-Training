


CUDA_VISIBLE_DEVICES=1 nohup python train_with_masks_adv_gd_less.py --max_iter 30000 --val_freq 200 --exp_name sngan_cifar10_adv_gd_0 --init-path initial_weights --gamma 0.01 --step 1 --rewind-path sngan_less/Model/checkpoint_0.pth > 0112_adv_gd_0_gpu1.out &

CUDA_VISIBLE_DEVICES=2 nohup python train_with_masks_adv_gd_less.py --max_iter 30000 --val_freq 200 --exp_name sngan_cifar10_adv_gd_1 --init-path initial_weights --gamma 0.01 --step 1 --rewind-path sngan_less/Model/checkpoint_1.pth > 0112_adv_gd_1_gpu2.out &

CUDA_VISIBLE_DEVICES=3 nohup python train_with_masks_adv_gd_less.py --max_iter 30000 --val_freq 200 --exp_name sngan_cifar10_adv_gd_2 --init-path initial_weights --gamma 0.01 --step 1 --rewind-path sngan_less/Model/checkpoint_2.pth > 0112_adv_gd_2_gpu3.out &

CUDA_VISIBLE_DEVICES=4 nohup python train_with_masks_adv_gd_less.py --max_iter 30000 --val_freq 200 --exp_name sngan_cifar10_adv_gd_3 --init-path initial_weights --gamma 0.01 --step 1 --rewind-path sngan_less/Model/checkpoint_3.pth > 0112_adv_gd_3_gpu4.out &

CUDA_VISIBLE_DEVICES=5 nohup python train_with_masks_adv_gd_less.py --max_iter 30000 --val_freq 200 --exp_name sngan_cifar10_adv_gd_4 --init-path initial_weights --gamma 0.01 --step 1 --rewind-path sngan_less/Model/checkpoint_4.pth > 0112_adv_gd_4_gpu5.out &

CUDA_VISIBLE_DEVICES=6 nohup python train_with_masks_adv_gd_less.py --max_iter 30000 --val_freq 200 --exp_name sngan_cifar10_adv_gd_5 --init-path initial_weights --gamma 0.01 --step 1 --rewind-path sngan_less/Model/checkpoint_5.pth > 0112_adv_gd_5_gpu6.out &



