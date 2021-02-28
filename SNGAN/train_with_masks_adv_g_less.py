import cfg
import models
import datasets
import random
from functions import validate, LinearLrDecay, load_params, copy_params
from functions_adv import train_adv_g
from utils.utils import set_log_dir, save_checkpoint, create_logger
from utils.inception_score import _init_inception
from utils.fid_score import create_inception_graph, check_or_download_inception

import torch
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import numpy as np
import torch.nn as nn
import torch.nn.utils.prune as prune
from tensorboardX import SummaryWriter


def pruning_generate(model, state_dict):
    for (name, m) in model.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            m = prune.custom_from_mask(m, name='weight', mask=state_dict[name + ".weight_mask"])


def main():
    args = cfg.parse_args()
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)
    
    # set tf env
    _init_inception()
    inception_path = check_or_download_inception(None)
    create_inception_graph(inception_path)

    # import network
    gen_net = eval('models.'+args.model+'.Generator')(args=args)
    dis_net = eval('models.'+args.model+'.Discriminator')(args=args)
    avg_gen_net = eval('models.'+args.model+'.Generator')(args=args)
    initial_gen_net_weight = torch.load(os.path.join(args.init_path, 'initial_gen_net.pth'), map_location="cpu")
    initial_dis_net_weight = torch.load(os.path.join(args.init_path, 'initial_dis_net.pth'), map_location="cpu")
        
    gen_net.load_state_dict(initial_gen_net_weight)
    avg_gen_net.load_state_dict(initial_gen_net_weight)
    dis_net.load_state_dict(initial_dis_net_weight)
    
    # set optimizer
    gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gen_net.parameters()),
                                     args.g_lr, (args.beta1, args.beta2))
    dis_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, dis_net.parameters()),
                                     args.d_lr, (args.beta1, args.beta2))
    gen_scheduler = LinearLrDecay(gen_optimizer, args.g_lr, 0.0, 0, args.max_iter * args.n_critic)
    dis_scheduler = LinearLrDecay(dis_optimizer, args.d_lr, 0.0, 0, args.max_iter * args.n_critic)

    # set up data_loader
    dataset = datasets.ImageDatasetLess(args)
    train_loader = dataset.train

    if args.dataset.lower() == 'cifar10':
        fid_stat = 'fid_stat/fid_stats_cifar10_train.npz'
    elif args.dataset.lower() == 'stl10':
        fid_stat = 'fid_stat/fid_stats_stl10_train.npz'
    else:
        raise NotImplementedError('no fid stat for %s' % args.dataset.lower())
    assert os.path.exists(fid_stat)

    # epoch number for dis_net
    args.max_epoch = args.max_epoch * args.n_critic
    if args.max_iter:
        args.max_epoch = np.ceil(args.max_iter * args.n_critic / len(train_loader))

    # initial
    fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (25, args.latent_dim)))
    start_epoch = 0
    best_fid = 1e4

    # set writer
    args.path_helper = set_log_dir('logs', args.exp_name + "_{}".format(args.percent))
    logger = create_logger(args.path_helper['log_path'])
    logger.info(args)
    writer_dict = {
        'writer': SummaryWriter(args.path_helper['log_path']),
        'train_global_steps': start_epoch * len(train_loader),
        'valid_global_steps': start_epoch // args.val_freq,
    }

    pruning_generate(gen_net, torch.load(os.path.join(args.rewind_path), map_location="cpu")['avg_gen_state_dict'])
    pruning_generate(avg_gen_net, torch.load(os.path.join(args.rewind_path), map_location="cpu")['avg_gen_state_dict'])
    
    gen_net = gen_net.cuda()
    dis_net = dis_net.cuda()
    avg_gen_net = avg_gen_net.cuda()
    gen_avg_param = copy_params(gen_net)
    logger.info(args)
    writer_dict = {
        'writer': SummaryWriter(args.path_helper['log_path']),
        'train_global_steps': start_epoch * len(train_loader),
        'valid_global_steps': start_epoch // args.val_freq,
    }

    # train loop
    for epoch in range(int(start_epoch), int(args.max_epoch)):
            
        lr_schedulers = (gen_scheduler, dis_scheduler) if args.lr_decay else None
        train_adv_g(args, gen_net, dis_net, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch, writer_dict, lr_schedulers, steps=args.steps, gamma=args.gamma)

        if epoch and epoch % args.val_freq == 0 or epoch == int(args.max_epoch)-1:
            backup_param = copy_params(gen_net)
            load_params(gen_net, gen_avg_param)
            inception_score, fid_score = validate(args, fixed_z, fid_stat, gen_net, writer_dict, epoch)
            logger.info('Inception score: %.4f, FID score: %.4f || @ epoch %d.' % (inception_score, fid_score, epoch))
            load_params(gen_net, backup_param)
            if fid_score < best_fid:
                best_fid = fid_score
                is_best = True
            else:
                is_best = False
        else:
            is_best = False

        load_params(avg_gen_net, gen_avg_param)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': args.model,
            'gen_state_dict': gen_net.state_dict(),
            'dis_state_dict': dis_net.state_dict(),
            'avg_gen_state_dict': avg_gen_net.state_dict(),
            'gen_optimizer': gen_optimizer.state_dict(),
            'dis_optimizer': dis_optimizer.state_dict(),
            'best_fid': best_fid,
            'path_helper': args.path_helper,
            'seed': args.random_seed
        }, is_best, args.path_helper['ckpt_path'])


if __name__ == '__main__':
    main()
