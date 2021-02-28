
CUDA_VISIBLE_DEVICES=5 nohup python train_with_masks_adv_g_less.py --max_iter 50000  --exp_name sngan_cifar10_adv_g_less_3  --gamma 0.01 --step 1 --val_freq 200 --rewind-path sngan_less/Model/checkpoint_3.pth > 0115_adv_g_less_3_gpu5.out &
sleep 2s
CUDA_VISIBLE_DEVICES=6 nohup python train_with_masks_adv_g_less.py --max_iter 50000  --exp_name sngan_cifar10_adv_g_less_3  --gamma 0.01 --step 1 --val_freq 200 --rewind-path sngan_less/Model/checkpoint_3.pth > 0115_adv_g_less_3_gpu6.out &
sleep 2s
CUDA_VISIBLE_DEVICES=7 nohup python train_with_masks_adv_g_less.py --max_iter 50000  --exp_name sngan_cifar10_adv_g_less_3  --gamma 0.01 --step 1 --val_freq 200 --rewind-path sngan_less/Model/checkpoint_3.pth > 0115_adv_g_less_3_gpu7.out &