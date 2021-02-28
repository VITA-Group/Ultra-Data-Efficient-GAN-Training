

CUDA_VISIBLE_DEVICES=1 nohup python train_adv_gd_less.py --max_iter 50000  --exp_name sngan_cifar10_adv_gd_0.01_3  --gamma 0.01 --step 3 --rewind-path sngan_less/Model/checkpoint_3.pth > 0112_adv_gd_0_gpu1.out &

CUDA_VISIBLE_DEVICES=6 nohup python train_adv_gd_less.py --max_iter 50000  --exp_name sngan_cifar10_adv_gd_0.01_5  --gamma 0.01 --step 5 > 0112_gd_0.01_5_gpu6.out & 

CUDA_VISIBLE_DEVICES=3 nohup python train_adv_gd_less.py --max_iter 50000  --exp_name sngan_cifar10_adv_gd_0.01_7  --gamma 0.01 --step 7 > 0112_gd_0.01_7_gpu3.out &




