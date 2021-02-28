import torch
import models
import cfg
import numpy as np
from utils.utils import set_log_dir, save_checkpoint, create_logger, pruning_generate, see_remain_rate, rewind_weight, see_remain_rate_orig
args = cfg.parse_args()
gen_net = eval('models.sngan_cifar10.Generator')(args=args).cuda()
pruning_generate(gen_net, 1-0.8**10)
checkpoint = torch.load(args.resume)
print(checkpoint['gen_state_dict'].keys())
gen_net.load_state_dict(checkpoint['gen_state_dict'])
see_remain_rate(gen_net)

num_kernel = 0
zero_kernel = 0
n_kernel = 0
state_dict = checkpoint['gen_state_dict']
for key in state_dict.keys():
    if 'mask' in key:
        mask = state_dict[key]
        print(mask.shape)
        num_kernel = num_kernel + mask.shape[1]
        for i in range(mask.shape[1]):
            if np.all(mask[:, i, :, :].cpu().numpy() == 0):
                zero_kernel  = zero_kernel + 1
            if np.sum(mask[:, i, :, :].cpu().numpy() == 0) > mask[:,i,:,:].reshape(-1).shape[0] * 0.9:
                n_kernel  = n_kernel + 1
print(zero_kernel)
print(n_kernel)
print(num_kernel)
