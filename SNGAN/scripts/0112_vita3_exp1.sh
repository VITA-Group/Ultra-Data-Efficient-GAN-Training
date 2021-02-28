sleep 3h

CUDA_VISIBLE_DEVICES=2 nohup python train_adv_gd_less.py --max_iter 50000 --val_freq 200 --exp_name sngan_cifar10_adv_gd --init-path initial_weights --gamma 0.01 --step 3 > 0112_adv_gd_0.01_3_gpu2.out &
sleep 2s
CUDA_VISIBLE_DEVICES=3 nohup python train_adv_gd_less.py --max_iter 50000 --val_freq 200 --exp_name sngan_cifar10_adv_gd --init-path initial_weights --gamma 0.01 --step 5 > 0112_adv_gd_0.01_5_gpu3.out &
sleep 2s
CUDA_VISIBLE_DEVICES=4 nohup python train_adv_gd_less.py --max_iter 50000 --val_freq 200 --exp_name sngan_cifar10_adv_gd --init-path initial_weights --gamma 0.01 --step 7 > 0112_adv_gd_0.01_7_gpu4.out &