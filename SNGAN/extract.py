# -*- coding: utf-8 -*-
# @Date    : 2019-07-25
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import cfg
import models
from functions import validate

import os
import numpy as np
import random
from tensorboardX import SummaryWriter
from imageio import imsave
from torchvision.utils import make_grid

import torch.nn as nn
import torch.nn.utils.prune as prune



def validate_cp(fixed_z, G):
	# eval mode
	G = G.eval()

	# generate images
	
	gen_img_big = (gen_img_big + 1)/2 # [-1,1] -> [0,1]

	return gen_img_big
            
def pruning_generate(model, state_dict):

    parameters_to_prune =[]
    for (name, m) in model.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            m = prune.custom_from_mask(m, name = 'weight', mask = state_dict[name + ".weight_mask"])


def main():
    args = cfg.parse_args()
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.backends.cudnn.benchmark = True
    
    # set tf env

    # import network
    gen_net = eval('models.'+args.model+'.Generator')(args=args).cuda()
    # initial
    np.random.seed(args.random_seed)
    

    # set writer
    print(f'=> resuming from {args.load_path}')
    checkpoint_file = args.load_path
    assert os.path.exists(checkpoint_file)
    checkpoint = torch.load(checkpoint_file)
    gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
        
    if 'avg_gen_state_dict' in checkpoint:
        gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
        epoch = checkpoint['epoch']
        print(f'=> loaded checkpoint {checkpoint_file} (epoch {epoch})')
    else:
        gen_net.load_state_dict(checkpoint)
        print(f'=> loaded checkpoint {checkpoint_file}')

    print(args)
    
    count = 0
    for _ in range(1000):
        fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (60, args.latent_dim)))
        gen_imgs = gen_net(fixed_z)
        gen_imgs = np.moveaxis(gen_imgs.detach().cpu().numpy(), 1, -1)
        for i in range(gen_imgs.shape[0]):
            img = gen_imgs[i]
            img = (img + 1) / 2
            imsave(os.path.join(args.save_path, 'test_result_{}.png'.format(count)), img)
            count = count + 1
    
if __name__ == '__main__':
    main()
