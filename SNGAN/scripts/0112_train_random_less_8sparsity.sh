
CUDA_VISIBLE_DEVICES=0 nohup python train_random_less_oneshot.py --max_iter 50000 --model sngan_cifar10 --val_freq 200 --exp_name sngan_cifar10_random_less --percent 0.2 > 0112_d_0.2_gpu0.out &
sleep 2s

CUDA_VISIBLE_DEVICES=0 nohup python train_random_less_oneshot.py --max_iter 50000 --model sngan_cifar10 --val_freq 200 --exp_name sngan_cifar10_random_less --percent 0.36 > 0112_d_0.36_gpu0.out &
sleep 2s

CUDA_VISIBLE_DEVICES=1 nohup python train_random_less_oneshot.py --max_iter 50000 --model sngan_cifar10 --val_freq 200 --exp_name sngan_cifar10_random_less --percent 0.488 > 0112_d_0.488_gpu1.out &
sleep 2s

CUDA_VISIBLE_DEVICES=1 nohup python train_random_less_oneshot.py --max_iter 50000 --model sngan_cifar10 --val_freq 200 --exp_name sngan_cifar10_random_less --percent 0.5904 > 0112_d_0._gpu1.out &
sleep 2s

CUDA_VISIBLE_DEVICES=2 nohup python train_random_less_oneshot.py --max_iter 50000 --model sngan_cifar10 --val_freq 200 --exp_name sngan_cifar10_random_less --percent 0.67232 > 0112_d_0.672_gpu2.out &
sleep 2s

CUDA_VISIBLE_DEVICES=2 nohup python train_random_less_oneshot.py --max_iter 50000 --model sngan_cifar10 --val_freq 200 --exp_name sngan_cifar10_random_less --percent 0.737856 > 0112_d_0.737856_gpu2.out &
sleep 2s

CUDA_VISIBLE_DEVICES=3 nohup python train_random_less_oneshot.py --max_iter 50000 --model sngan_cifar10 --val_freq 200 --exp_name sngan_cifar10_random_less --percent 0.790285 > 0112_d_0.790285_gpu3.out &
sleep 2s

CUDA_VISIBLE_DEVICES=3 nohup python train_random_less_oneshot.py --max_iter 50000 --model sngan_cifar10 --val_freq 200 --exp_name sngan_cifar10_random_less --percent 0.83222 > 0112_d_0.83222_gpu3.out &