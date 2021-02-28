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


def fourD2threeD(batch, n_row=10):
	'''
	Convert a batch of images (N,W,H,C) to a single big image (W*n, H*m, C)
	Input:
		batch: type=ndarray, shape=(N,W,H,C)
	Return:
		rows: type=ndarray, shape=(W*n, H*m, C)
	'''
	N = batch.shape[0]
	img_list = np.split(batch, N)
	for i, img in enumerate(img_list):
		img_list[i] = img.squeeze(axis=0)
	one_row = np.concatenate(img_list, axis=1)
	# print('one_row:', one_row.shape)
	row_list = np.split(one_row, n_row, axis=1)
	rows = np.concatenate(row_list, axis=0)
	return rows

def validate_cp(fixed_z, G, n_row=5):
	# eval mode
	G = G.eval()

	# generate images
	gen_imgs = G(fixed_z)
	gen_img_big = fourD2threeD( np.moveaxis(gen_imgs.detach().cpu().numpy(), 1, -1), n_row=n_row )

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
    fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (16, args.latent_dim)))

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
    imgs = validate_cp(fixed_z, gen_net, n_row=4)
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    imsave(os.path.join(args.save_path, 'test_result.png'), imgs)
    
if __name__ == '__main__':
    main()
