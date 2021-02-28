

CUDA_VISIBLE_DEVICES=1 nohup python train_less.py --max_iter 50000  --exp_name sngan_cifar10_less_0.2 --ratio 0.2 --val_freq 200 > 0113_less_0.2_gpu1.out &
CUDA_VISIBLE_DEVICES=2 nohup python train_less.py --max_iter 50000  --exp_name sngan_cifar10_less_0.3 --ratio 0.3 --val_freq 200 > 0113_less_0.3_gpu2.out &
CUDA_VISIBLE_DEVICES=3 nohup python train_less.py --max_iter 50000  --exp_name sngan_cifar10_less_0.4 --ratio 0.4 --val_freq 100 > 0113_less_0.4_gpu3.out &
CUDA_VISIBLE_DEVICES=4 nohup python train_less.py --max_iter 50000  --exp_name sngan_cifar10_less_0.5 --ratio 0.5 --val_freq 100 > 0113_less_0.5_gpu4.out &
CUDA_VISIBLE_DEVICES=5 nohup python train_less.py --max_iter 50000  --exp_name sngan_cifar10_less_0.6 --ratio 0.6  --val_freq 50 > 0113_less_0.6_gpu5.out &
CUDA_VISIBLE_DEVICES=6 nohup python train_less.py --max_iter 50000  --exp_name sngan_cifar10_less_0.7 --ratio 0.7  --val_freq 50 > 0113_less_0.7_gpu6.out &



