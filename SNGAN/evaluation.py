import os
import torch
import torch.nn as nn 

import numpy as np
from imageio import imsave
from torchvision.utils import make_grid
import torch.nn.utils.prune as prune

import argparse
from tqdm import tqdm
from copy import deepcopy

from utils.inception_score import get_inception_score
from utils.fid_score import calculate_fid_given_paths
from utils.inception_score import _init_inception
from utils.fid_score import create_inception_graph, check_or_download_inception

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-path", type=str)
    parser.add_argument('--bottom_width', type=int, default=4)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--gf_dim', type=int, default=256)
    parser.add_argument('--df_dim', type=int, default=128)
    parser.add_argument('--round', type=int, default=3)
    parser.add_argument('--eval_batch_size', type=int, default=100)
    parser.add_argument('--num_eval_imgs', type=int, default=50000)
    args = parser.parse_args()
    return args


class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 activation=nn.ReLU(), upsample=False, n_classes=0):
        super(GenBlock, self).__init__()
        self.activation = activation
        self.upsample = upsample
        self.learnable_sc = in_channels != out_channels or upsample
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        self.n_classes = n_classes
        self.c1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=ksize, padding=pad)
        self.c2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=ksize, padding=pad)

        self.b1 = nn.BatchNorm2d(in_channels)
        self.b2 = nn.BatchNorm2d(hidden_channels)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def upsample_conv(self, x, conv):
        return conv(nn.UpsamplingNearest2d(scale_factor=2)(x))

    def residual(self, x):
        h = x
        h = self.b1(h)
        h = self.activation(h)
        h = self.upsample_conv(h, self.c1) if self.upsample else self.c1(h)
        h = self.b2(h)
        h = self.activation(h)
        h = self.c2(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.upsample_conv(x, self.c_sc) if self.upsample else self.c_sc(x)
            return x
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class Generator(nn.Module):
    def __init__(self, args, activation=nn.ReLU(), n_classes=0):
        super(Generator, self).__init__()
        self.bottom_width = args.bottom_width
        self.activation = activation
        self.n_classes = n_classes
        self.ch = args.gf_dim
        self.l1 = nn.Linear(args.latent_dim, (self.bottom_width ** 2) * self.ch)
        self.block2 = GenBlock(self.ch, self.ch, activation=activation, upsample=True, n_classes=n_classes)
        self.block3 = GenBlock(self.ch, self.ch, activation=activation, upsample=True, n_classes=n_classes)
        self.block4 = GenBlock(self.ch, self.ch, activation=activation, upsample=True, n_classes=n_classes)
        self.b5 = nn.BatchNorm2d(self.ch)
        self.c5 = nn.Conv2d(self.ch, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, z):

        h = z
        h = self.l1(h).view(-1, self.ch, self.bottom_width, self.bottom_width)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.b5(h)
        h = self.activation(h)
        h = nn.Tanh()(self.c5(h))
        return h
    
def pruning_generate(model, state_dict):

    parameters_to_prune =[]
    for (name, m) in model.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            m = prune.custom_from_mask(m, name = 'weight', mask = state_dict[name + ".weight_mask"])

def see_remain_rate(model):
    sum_list = 0
    zero_sum = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            sum_list = sum_list+float(m.weight.nelement())
            zero_sum = zero_sum+float(torch.sum(m.weight == 0))     
    print('remain weight = ', 100*(1-zero_sum/sum_list),'%')
    
def see_remain_rate_orig(model):
    sum_list = 0
    zero_sum = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            sum_list = sum_list+float(m.weight_orig.nelement())
            zero_sum = zero_sum+float(torch.sum(m.weight_orig == 0))     
    print('remain weight = ', 100*(1-zero_sum/sum_list),'%')
    
def see_remain_rate_mask(model):
    sum_list = 0
    zero_sum = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            sum_list = sum_list+float(m.weight_orig.nelement())
            zero_sum = zero_sum+float(torch.sum(m.weight_mask == 0))     
    print('remain weight = ', 100*(1-zero_sum/sum_list),'%')

def evaluate(args, fixed_z, fid_stat, gen_net: nn.Module):
    # eval mode
    gen_net = gen_net.eval()

    # generate images
    sample_imgs = gen_net(fixed_z)
    img_grid = make_grid(sample_imgs, nrow=5, normalize=True, scale_each=True)

    # get fid and inception score
    fid_buffer_dir = 'fid_buffer_test'
    if not os.path.exists(fid_buffer_dir): os.makedirs(fid_buffer_dir)

    eval_iter = args.num_eval_imgs // args.eval_batch_size
    img_list = list()
    for iter_idx in tqdm(range(eval_iter), desc='sample images'):
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim)))

        # Generate a batch of images
        gen_imgs = gen_net(z).mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
        for img_idx, img in enumerate(gen_imgs):
            file_name = os.path.join(fid_buffer_dir, 'iter%d_b%d.png' % (iter_idx, img_idx))
            imsave(file_name, img)
        img_list.extend(list(gen_imgs))

    # get inception score
    print('=> calculate inception score')
    mean, std = get_inception_score(img_list)

    # get fid score
    print('=> calculate fid score')
    fid_score = calculate_fid_given_paths([fid_buffer_dir, fid_stat], inception_path=None)

    os.system('rm -r {}'.format(fid_buffer_dir))
    return mean, fid_score


def main():
    args = parse_args()
    gen_net = Generator(args).cuda()
    _init_inception()
    inception_path = check_or_download_inception(None)
    create_inception_graph(inception_path)
    assert os.path.exists(args.load_path), "checkpoint file {} is not found".format(args.load_path)
    checkpoint = torch.load(args.load_path)
    torch.manual_seed(12345)
    torch.cuda.manual_seed(12345)
    np.random.seed(12345)
    #print("remaining percent: {}".format(0.8 ** checkpoint['round']))
    pruning_generate(gen_net, checkpoint['avg_gen_state_dict']) # Create a buffer for mask]
    gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
    
    see_remain_rate_mask(gen_net)
    see_remain_rate(gen_net)
    print("Best FID:{}".format(checkpoint['best_fid'])) 
    fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (25, args.latent_dim)))
    fid_stat = 'fid_stat/fid_stats_cifar10_train.npz'
    inception_score, fid_score = evaluate(args, fixed_z, fid_stat, gen_net)
    
    print('Inception score: %.4f, FID score: %.4f' % (inception_score, fid_score))
    
if __name__ == '__main__':
    main()