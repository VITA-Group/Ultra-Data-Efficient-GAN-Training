

CUDA_VISIBLE_DEVICES=5 nohup python train_adv_gd.py --max_iter 50000  --exp_name sngan_cifar10_adv_gd_0.01_3  --gamma 0.01 --step 3 > 0115_adv_gd_gpu5.out &

CUDA_VISIBLE_DEVICES=6 nohup python train_adv_gd.py --max_iter 50000  --exp_name sngan_cifar10_adv_gd_0.01_5  --gamma 0.01 --step 5 > 0115_adv_gd_gpu6.out &

CUDA_VISIBLE_DEVICES=7 nohup python train_adv_gd.py --max_iter 50000  --exp_name sngan_cifar10_adv_gd_0.01_7  --gamma 0.01 --step 7 > 0115_adv_gd_gpu7.out &