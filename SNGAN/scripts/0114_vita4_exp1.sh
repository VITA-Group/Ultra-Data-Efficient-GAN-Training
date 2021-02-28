

CUDA_VISIBLE_DEVICES=0 nohup python train_with_masks_adv_d_less.py --max_iter 50000  --exp_name sngan_cifar10_adv_d_2  --gamma 0.01 --step 1 --val_freq 200 --rewind-path sngan_less/Model/checkpoint_2.pth > 0114_d_2_gpu0.out &
sleep 2s
CUDA_VISIBLE_DEVICES=1 nohup python train_with_masks_adv_d_less.py --max_iter 50000  --exp_name sngan_cifar10_adv_d_2  --gamma 0.01 --step 1 --val_freq 200 --rewind-path sngan_less/Model/checkpoint_2.pth > 0114_d_2_gpu1.out &
sleep 2s
CUDA_VISIBLE_DEVICES=2 nohup python train_with_masks_adv_d_less.py --max_iter 50000  --exp_name sngan_cifar10_adv_d_2  --gamma 0.01 --step 1 --val_freq 200 --rewind-path sngan_less/Model/checkpoint_2.pth > 0114_d_2_gpu2.out &
sleep 2s
CUDA_VISIBLE_DEVICES=4 nohup python train_with_masks_adv_g_less.py --max_iter 50000  --exp_name sngan_cifar10_adv_g_3  --gamma 0.01 --step 1 --val_freq 200 --rewind-path sngan_less/Model/checkpoint_3.pth > 0114_g_3_gpu4.out &
sleep 2s
CUDA_VISIBLE_DEVICES=5 nohup python train_with_masks_adv_g_less.py --max_iter 50000  --exp_name sngan_cifar10_adv_g_3  --gamma 0.01 --step 1 --val_freq 200 --rewind-path sngan_less/Model/checkpoint_3.pth > 0114_g_3_gpu5.out &
sleep 2s
CUDA_VISIBLE_DEVICES=7 nohup python train_with_masks_adv_g_less.py --max_iter 50000  --exp_name sngan_cifar10_adv_g_3  --gamma 0.01 --step 1 --val_freq 200 --rewind-path sngan_less/Model/checkpoint_3.pth > 0114_g_3_gpu7.out &
