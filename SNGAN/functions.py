# -*- coding: utf-8 -*-
# @Date    : 2019-07-25
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from imageio import imsave
from copy import deepcopy
import logging
from utils.inception_score import get_inception_score
from utils.fid_score import calculate_fid_given_paths

logger = logging.getLogger(__name__)


def train(args, gen_net: nn.Module, dis_net: nn.Module, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch,
          writer_dict, schedulers=None):
    
    np.random.seed(args.random_seed + epoch ** 2)
    writer = writer_dict['writer']
    gen_step = 0

    # train mode
    gen_net = gen_net.train()
    dis_net = dis_net.train()
    tps = []
    tns = []
    fns = []
    fps = []
    
    gen_net.train()
    for iter_idx, (imgs, _) in enumerate(train_loader):
        global_steps = writer_dict['train_global_steps']

        # Adversarial ground truths
        real_imgs = imgs.type(torch.cuda.FloatTensor)
        #real_imgs = DiffAugment(real_imgs, policy="translation,cutout")
        # Sample noise as generator input
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim)))
        # ---------------------
        #  Train Discriminator
        # ---------------------
        dis_optimizer.zero_grad()

        real_validity = dis_net(real_imgs)
        fake_imgs = gen_net(z).detach()
        #fake_imgs = DiffAugment(fake_imgs, policy="translation,cutout")

        assert fake_imgs.size() == real_imgs.size()

        fake_validity = dis_net(fake_imgs)
        
        # cal loss
        d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                 torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
        d_loss.backward()
        dis_optimizer.step()
        writer.add_scalar('d_loss', d_loss.item(), global_steps)
        
        tp = torch.sum(real_validity > 0)
        tn = torch.sum(fake_validity < 0)
        fn = torch.sum(real_validity <= 0)
        fp = torch.sum(fake_validity >= 0)
        precision = tp / (tp + fp + 1e-3)
        recall = tp / (tp + fn + 1e-3)
        accuracy = (tp + tn) / (tp + fn + fp + tn)
        
        fps.append(fp.item())
        tps.append(tp.item())
        fns.append(fn.item())
        tns.append(tn.item())

        writer.add_scalar('precision', precision.item(), global_steps)
        writer.add_scalar('recall', recall.item(), global_steps)
        writer.add_scalar('accuracy', accuracy.item(), global_steps)
        # -----------------
        #  Train Generator
        # -----------------
        if global_steps % args.n_critic == 0:
            gen_optimizer.zero_grad()

            gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim)))
            gen_imgs = gen_net(gen_z)
            
            #gen_imgs = DiffAugment(gen_imgs, policy="translation,cutout")

            fake_validity = dis_net(gen_imgs)
            tn = torch.sum(fake_validity < 0)
            
            writer.add_scalar('accuracy_evaluate', tn.item()/fake_validity.numel(), global_steps)

            # cal loss
            g_loss = -torch.mean(fake_validity)
        
            g_loss.backward()
            gen_optimizer.step()

            # adjust learning rate
            if schedulers:
                gen_scheduler, dis_scheduler = schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)
                writer.add_scalar('LR/g_lr', g_lr, global_steps)
                writer.add_scalar('LR/d_lr', d_lr, global_steps)

            # moving average weight
            for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
                avg_p.mul_(0.999).add_(0.001, p.data)

            writer.add_scalar('g_loss', g_loss.item(), global_steps)
            gen_step += 1

        # verbose
        if gen_step and iter_idx % args.print_freq == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                (epoch, args.max_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss.item()))

        writer_dict['train_global_steps'] = global_steps + 1
    
    writer.add_scalar('precision_epoch', sum(tps) / (sum(tps) + sum(fps) + 1e-3), global_steps)
    writer.add_scalar('recall_epoch', sum(tps) / (sum(tps) + sum(fns) + 1e-3), global_steps)
    writer.add_scalar('accuracy_epoch', (sum(tps) + sum(tns)) / (sum(tps) + sum(tns) + sum(fps) + sum(fns) + 1e-3), global_steps)


def train_diffaug(args, gen_net: nn.Module, dis_net: nn.Module, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch, writer_dict, schedulers=None):
    
    np.random.seed(args.random_seed + epoch ** 2)
    writer = writer_dict['writer']
    gen_step = 0

    # train mode
    gen_net = gen_net.train()
    dis_net = dis_net.train()
    tps = []
    tns = []
    fns = []
    fps = []
    
    gen_net.train()
    for iter_idx, (imgs, _) in enumerate(train_loader):
        global_steps = writer_dict['train_global_steps']

        # Adversarial ground truths
        real_imgs = imgs.type(torch.cuda.FloatTensor)
        real_imgs = DiffAugment(real_imgs, policy="translation,cutout")
        # Sample noise as generator input
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim)))
        # ---------------------
        #  Train Discriminator
        # ---------------------
        dis_optimizer.zero_grad()

        real_validity = dis_net(real_imgs)
        fake_imgs = gen_net(z).detach()
        fake_imgs = DiffAugment(fake_imgs, policy="translation,cutout")

        assert fake_imgs.size() == real_imgs.size()

        fake_validity = dis_net(fake_imgs)
        
        # cal loss
        d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                 torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
        d_loss.backward()
        dis_optimizer.step()
        writer.add_scalar('d_loss', d_loss.item(), global_steps)
        
        tp = torch.sum(real_validity > 0)
        tn = torch.sum(fake_validity < 0)
        fn = torch.sum(real_validity <= 0)
        fp = torch.sum(fake_validity >= 0)
        precision = tp / (tp + fp + 1e-3)
        recall = tp / (tp + fn + 1e-3)
        accuracy = (tp + tn) / (tp + fn + fp + tn)
        
        fps.append(fp.item())
        tps.append(tp.item())
        fns.append(fn.item())
        tns.append(tn.item())

        writer.add_scalar('precision', precision.item(), global_steps)
        writer.add_scalar('recall', recall.item(), global_steps)
        writer.add_scalar('accuracy', accuracy.item(), global_steps)
        # -----------------
        #  Train Generator
        # -----------------
        if global_steps % args.n_critic == 0:
            gen_optimizer.zero_grad()

            gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim)))
            gen_imgs = gen_net(gen_z)
            
            gen_imgs = DiffAugment(gen_imgs, policy="translation,cutout")

            fake_validity = dis_net(gen_imgs)

            # cal loss
            g_loss = -torch.mean(fake_validity)
        
            g_loss.backward()
            gen_optimizer.step()

            # adjust learning rate
            if schedulers:
                gen_scheduler, dis_scheduler = schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)
                writer.add_scalar('LR/g_lr', g_lr, global_steps)
                writer.add_scalar('LR/d_lr', d_lr, global_steps)

            # moving average weight
            for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
                avg_p.mul_(0.999).add_(0.001, p.data)

            writer.add_scalar('g_loss', g_loss.item(), global_steps)
            gen_step += 1

        # verbose
        if gen_step and iter_idx % args.print_freq == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                (epoch, args.max_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss.item()))

        writer_dict['train_global_steps'] = global_steps + 1
    
    writer.add_scalar('precision_epoch', sum(tps) / (sum(tps) + sum(fps) + 1e-3), global_steps)
    writer.add_scalar('recall_epoch', sum(tps) / (sum(tps) + sum(fns) + 1e-3), global_steps)
    writer.add_scalar('accuracy_epoch', (sum(tps) + sum(tns)) / (sum(tps) + sum(tns) + sum(fps) + sum(fns) + 1e-3), global_steps)


def train_kd(args, gen_net: nn.Module, dis_net: nn.Module, orig_dis_net, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch, writer_dict, schedulers=None):
    np.random.seed(args.random_seed + epoch ** 2)
    writer = writer_dict['writer']
    gen_step = 0

    # train mode
    gen_net.train()
    dis_net.train()
    orig_dis_net.eval()
    tps = []
    tns = []
    fns = []
    fps = []
    
    gen_net.train()
    for iter_idx, (imgs, _) in enumerate(train_loader):
        global_steps = writer_dict['train_global_steps']

        # Adversarial ground truths
        real_imgs = imgs.type(torch.cuda.FloatTensor)

        # Sample noise as generator input
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim)))

        # ---------------------
        #  Train Discriminator
        # ---------------------
        dis_optimizer.zero_grad()

        real_validity = dis_net(real_imgs)
        fake_imgs = gen_net(z).detach()
        assert fake_imgs.size() == real_imgs.size()

        fake_validity = dis_net(fake_imgs)

        orig_real_validity = orig_dis_net(real_imgs)
        orig_fake_validity = orig_dis_net(fake_imgs)
        
        # cal loss
        d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                 torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
        
        sp = torch.cat([torch.sigmoid(real_validity), torch.sigmoid(fake_validity)], axis = 0)
        sp = torch.cat([sp, 1 - sp], axis = 1)
        
        tp = torch.cat([torch.sigmoid(orig_real_validity), torch.sigmoid(orig_fake_validity)], axis = 0)
        tp = torch.cat([tp, 1 - tp], axis = 1)
        
        teacher_loss = torch.nn.functional.kl_div(torch.log(sp + 1e-10), tp.detach())
        d_loss = d_loss + teacher_loss
        d_loss.backward()
        dis_optimizer.step()

        writer.add_scalar('d_loss', d_loss.item(), global_steps)
        
        tp = torch.sum(real_validity > 0)
        tn = torch.sum(fake_validity < 0)
        fn = torch.sum(real_validity <= 0)
        fp = torch.sum(fake_validity >= 0)
        precision = tp / (tp + fp + 1e-3)
        recall = tp / (tp + fn + 1e-3)
        accuracy = (tp + tn) / (tp + fn + fp + tn)
        
        fps.append(fp.item())
        tps.append(tp.item())
        fns.append(fn.item())
        tns.append(tn.item())

        writer.add_scalar('precision', precision.item(), global_steps)
        writer.add_scalar('recall', recall.item(), global_steps)
        writer.add_scalar('accuracy', accuracy.item(), global_steps)
        # -----------------
        #  Train Generator
        # -----------------
        if global_steps % args.n_critic == 0:
            gen_optimizer.zero_grad()

            gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim)))
            gen_imgs = gen_net(gen_z)
            fake_validity = dis_net(gen_imgs)

            # cal loss
            g_loss = -torch.mean(fake_validity)
            g_loss.backward()
            gen_optimizer.step()

            # adjust learning rate
            if schedulers:
                gen_scheduler, dis_scheduler = schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)
                writer.add_scalar('LR/g_lr', g_lr, global_steps)
                writer.add_scalar('LR/d_lr', d_lr, global_steps)

            # moving average weight
            for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
                avg_p.mul_(0.999).add_(0.001, p.data)

            writer.add_scalar('g_loss', g_loss.item(), global_steps)
            gen_step += 1

        # verbose
        if gen_step and iter_idx % args.print_freq == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                (epoch, args.max_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss.item()))

        writer_dict['train_global_steps'] = global_steps + 1
    
    writer.add_scalar('precision_epoch', sum(tps) / (sum(tps) + sum(fps) + 1e-3), global_steps)
    writer.add_scalar('recall_epoch', sum(tps) / (sum(tps) + sum(fns) + 1e-3), global_steps)
    writer.add_scalar('accuracy_epoch', (sum(tps) + sum(tns)) / (sum(tps) + sum(tns) + sum(fps) + sum(fns) + 1e-3), global_steps)


def train_with_mask(args, gen_net: nn.Module, dis_net: nn.Module, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch,
          writer_dict, masks, schedulers=None):
    np.random.seed(args.random_seed + epoch ** 2)
    writer = writer_dict['writer']
    gen_step = 0

    # train mode
    gen_net.train()
    dis_net.train()
    tps = []
    tns = []
    fns = []
    fps = []
        
    for iter_idx, (imgs, _) in enumerate(train_loader):
        global_steps = writer_dict['train_global_steps']

        # Adversarial ground truths
        real_imgs = imgs.type(torch.cuda.FloatTensor)

        # Sample noise as generator input
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim)))

        # ---------------------
        #  Train Discriminator
        # ---------------------
        
        dis_optimizer.zero_grad()

        real_validity = dis_net(real_imgs)
        fake_imgs = gen_net(z).detach()
        assert fake_imgs.size() == real_imgs.size()

        fake_validity = dis_net(fake_imgs)

        # cal loss
        d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                 torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
        d_loss.backward()
        
        for k, m in enumerate(dis_net.modules()):
            if isinstance(m, nn.Conv2d):
                m.weight_orig.grad.mul_(masks[k])
                
        dis_optimizer.step()

        writer.add_scalar('d_loss', d_loss.item(), global_steps)
        
        tp = torch.sum(real_validity > 0)
        tn = torch.sum(fake_validity < 0)
        fn = torch.sum(real_validity <= 0)
        fp = torch.sum(fake_validity >= 0)
        precision = tp / (tp + fp + 1e-3)
        recall = tp / (tp + fn + 1e-3)
        accuracy = (tp + tn) / (tp + fp + tn + tn)
        
        fps.append(fp.item())
        tps.append(tp.item())
        fns.append(fn.item())
        tns.append(tn.item())
        # -----------------
        #  Train Generator
        # -----------------
        if global_steps % args.n_critic == 0:
            gen_optimizer.zero_grad()

            gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim)))
            gen_imgs = gen_net(gen_z)
            fake_validity = dis_net(gen_imgs)

            # cal loss
            g_loss = -torch.mean(fake_validity)
            g_loss.backward()
            gen_optimizer.step()
            # adjust learning rate
            if schedulers:
                gen_scheduler, dis_scheduler = schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)
                writer.add_scalar('LR/g_lr', g_lr, global_steps)
                writer.add_scalar('LR/d_lr', d_lr, global_steps)

            # moving average weight
            for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
                avg_p.mul_(0.999).add_(0.001, p.data)

            writer.add_scalar('g_loss', g_loss.item(), global_steps)
            gen_step += 1

        # verbose
        if gen_step and iter_idx % args.print_freq == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                (epoch, args.max_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss.item()))

        writer_dict['train_global_steps'] = global_steps + 1
    
    writer.add_scalar('precision_epoch', sum(tps) / (sum(tps) + sum(fps) + 1e-3), global_steps)
    writer.add_scalar('recall_epoch', sum(tps) / (sum(tps) + sum(fns) + 1e-3), global_steps)
    writer.add_scalar('accuracy_epoch', (sum(tps) + sum(tns)) / (sum(tps) + sum(tns) + sum(fps) + sum(fns) + 1e-3), global_steps)

def train_with_mask_kd(args, gen_net: nn.Module, dis_net: nn.Module, orig_dis_net, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch,
          writer_dict, masks, schedulers=None):
    np.random.seed(args.random_seed + epoch ** 2)
    writer = writer_dict['writer']
    gen_step = 0
    orig_dis_net.eval()
    gen_net.train()
    dis_net.train()
    # train mode
    gen_net = gen_net.train()
    dis_net = dis_net.train()
    tps = []
    tns = []
    fns = []
    fps = []
        
    for iter_idx, (imgs, _) in enumerate((train_loader)):
        global_steps = writer_dict['train_global_steps']

        # Adversarial ground truths
        real_imgs = imgs.type(torch.cuda.FloatTensor)

        # Sample noise as generator input
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim)))

        # ---------------------
        #  Train Discriminator
        # ---------------------
        
        dis_optimizer.zero_grad()

        real_validity = dis_net(real_imgs)
        fake_imgs = gen_net(z).detach()
        assert fake_imgs.size() == real_imgs.size()

        fake_validity = dis_net(fake_imgs)

        orig_real_validity = orig_dis_net(real_imgs)
        orig_fake_validity = orig_dis_net(fake_imgs)
        
        # cal loss
        d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                 torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
        
        sp = torch.cat([torch.sigmoid(real_validity), torch.sigmoid(fake_validity)], axis = 0)
        sp = torch.cat([sp, 1 - sp], axis = 1)
        
        tp = torch.cat([torch.sigmoid(orig_real_validity), torch.sigmoid(orig_fake_validity)], axis = 0)
        tp = torch.cat([tp, 1 - tp], axis = 1)
        
        teacher_loss = torch.nn.functional.kl_div(torch.log(sp + 1e-10), tp.detach())
        d_loss = d_loss + teacher_loss
        d_loss.backward()
        
        for k, m in enumerate(dis_net.modules()):
            if isinstance(m, nn.Conv2d):
                m.weight_orig.grad.mul_(masks[k])
                
        dis_optimizer.step()

        writer.add_scalar('d_loss', d_loss.item(), global_steps)
        
        tp = torch.sum(real_validity > 0)
        tn = torch.sum(fake_validity < 0)
        fn = torch.sum(real_validity <= 0)
        fp = torch.sum(fake_validity >= 0)
        precision = tp / (tp + fp + 1e-3)
        recall = tp / (tp + fn + 1e-3)
        accuracy = (tp + tn) / (tp + fn + fp + tn)
        
        fps.append(fp.item())
        tps.append(tp.item())
        fns.append(fn.item())
        tns.append(tn.item())

        writer.add_scalar('precision', precision.item(), global_steps)
        writer.add_scalar('recall', recall.item(), global_steps)
        writer.add_scalar('accuracy', accuracy.item(), global_steps)
        # -----------------
        #  Train Generator
        # -----------------
        if global_steps % args.n_critic == 0:
            gen_optimizer.zero_grad()

            gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim)))
            gen_imgs = gen_net(gen_z)
            fake_validity = dis_net(gen_imgs)

            # cal loss
            g_loss = -torch.mean(fake_validity)
            g_loss.backward()
            gen_optimizer.step()
            # adjust learning rate
            if schedulers:
                gen_scheduler, dis_scheduler = schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)
                writer.add_scalar('LR/g_lr', g_lr, global_steps)
                writer.add_scalar('LR/d_lr', d_lr, global_steps)

            # moving average weight
            for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
                avg_p.mul_(0.999).add_(0.001, p.data)

            writer.add_scalar('g_loss', g_loss.item(), global_steps)
            gen_step += 1

        # verbose
        if gen_step and iter_idx % args.print_freq == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                (epoch, args.max_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss.item()))

        writer_dict['train_global_steps'] = global_steps + 1
    
    writer.add_scalar('precision_epoch', sum(tps) / (sum(tps) + sum(fps) + 1e-3), global_steps)
    writer.add_scalar('recall_epoch', sum(tps) / (sum(tps) + sum(fns) + 1e-3), global_steps)
    writer.add_scalar('accuracy_epoch', (sum(tps) + sum(tns)) / (sum(tps) + sum(tns) + sum(fps) + sum(fns) + 1e-3), global_steps)


def validate(args, fixed_z, fid_stat, gen_net: nn.Module, writer_dict, epoch):
    np.random.seed(args.random_seed ** 2 + epoch)
    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']

    # eval mode
    gen_net = gen_net.eval()

    # generate images
    sample_imgs = gen_net(fixed_z)
    img_grid = make_grid(sample_imgs, nrow=5, normalize=True, scale_each=True)

    # get fid and inception score
    fid_buffer_dir = os.path.join(args.path_helper['sample_path'], 'fid_buffer')
    os.makedirs(fid_buffer_dir)

    eval_iter = args.num_eval_imgs // args.eval_batch_size
    img_list = list()
    for iter_idx in range(eval_iter):
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim)))

        # Generate a batch of images
        gen_imgs = gen_net(z).mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
        for img_idx, img in enumerate(gen_imgs):
            file_name = os.path.join(fid_buffer_dir, 'iter%d_b%d.png' % (iter_idx, img_idx))
            imsave(file_name, img)
        img_list.extend(list(gen_imgs))

    # get inception score
    logger.info('=> calculate inception score')
    mean, std = get_inception_score(img_list)

    # get fid score
    logger.info('=> calculate fid score')
    fid_score = calculate_fid_given_paths([fid_buffer_dir, fid_stat], inception_path=None)

    os.system('rm -r {}'.format(fid_buffer_dir))

    writer.add_image('sampled_images', img_grid, global_steps)
    writer.add_scalar('Inception_score/mean', mean, global_steps)
    writer.add_scalar('Inception_score/std', std, global_steps)
    writer.add_scalar('FID_score', fid_score, global_steps)

    writer_dict['valid_global_steps'] = global_steps + 1

    return mean, fid_score


class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return lr


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten

    
def train_adv_d(args, gen_net: nn.Module, dis_net: nn.Module, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch,
          writer_dict, schedulers=None, steps=1, gamma=0.01):
    
    np.random.seed(args.random_seed + epoch ** 2)
    writer = writer_dict['writer']
    gen_step = 0

    # train mode
    gen_net = gen_net.train()
    dis_net = dis_net.train()
    tps = []
    tns = []
    fns = []
    fps = []
    
    gen_net.train()
    for iter_idx, (imgs, _) in enumerate(train_loader):
        global_steps = writer_dict['train_global_steps']

        # Adversarial ground truths
        real_imgs = imgs.type(torch.cuda.FloatTensor)

        # Sample noise as generator input
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim)))
        # ---------------------
        #  Train Discriminator
        # ---------------------
        dis_optimizer.zero_grad()

        clean_feature_real = dis_net(real_imgs, fc=False)
        adv_feature_real = PGD(clean_feature_real, 
                model=dis_net.l5)
        real_validity_clean = dis_net(clean_feature_real, fc=False, only_fc=True)
        real_validity_adv   = dis_net(adv_feature_real, fc=False, only_fc=True)
       
        #real_validity_clean = dis_net(real_imgs)
        fake_imgs = gen_net(z).detach()
        assert fake_imgs.size() == real_imgs.size()

        clean_feature_fake = dis_net(fake_imgs, fc=False)
        adv_feature_fake = PGD(clean_feature_fake, model=dis_net.l5, steps=steps, gamma=gamma)
        fake_validity_clean = dis_net(clean_feature_fake, fc=False, only_fc=True)
        fake_validity_adv   = dis_net(adv_feature_fake, fc=False, only_fc=True)
        # cal loss
        d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - (real_validity_clean) )) / 2+ \
                 torch.mean(nn.ReLU(inplace=True)(1.0 + (fake_validity_clean) )) / 2+ \
                 torch.mean(nn.ReLU(inplace=True)(1.0 - (real_validity_adv) )) / 2 + \
                 torch.mean(nn.ReLU(inplace=True)(1.0 + (fake_validity_adv) )) / 2 
                 
        d_loss.backward()
        dis_optimizer.step()
        writer.add_scalar('d_loss', d_loss.item(), global_steps)
        
        tp = torch.sum(real_validity_clean > 0)
        tn = torch.sum(fake_validity_clean < 0)
        fn = torch.sum(real_validity_clean <= 0)
        fp = torch.sum(fake_validity_clean >= 0)
        precision = tp / (tp + fp + 1e-3)
        recall = tp / (tp + fn + 1e-3)
        accuracy = (tp + tn) / (tp + fn + fp + tn)
        
        fps.append(fp.item())
        tps.append(tp.item())
        fns.append(fn.item())
        tns.append(tn.item())

        writer.add_scalar('precision', precision.item(), global_steps)
        writer.add_scalar('recall', recall.item(), global_steps)
        writer.add_scalar('accuracy', accuracy.item(), global_steps)
        # -----------------
        #  Train Generator
        # -----------------
        if global_steps % args.n_critic == 0:
            gen_optimizer.zero_grad()

            gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim)))
            gen_imgs = gen_net(gen_z)
            fake_validity = dis_net(gen_imgs)
            # cal loss
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            gen_optimizer.step()

            # adjust learning rate
            if schedulers:
                gen_scheduler, dis_scheduler = schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)
                writer.add_scalar('LR/g_lr', g_lr, global_steps)
                writer.add_scalar('LR/d_lr', d_lr, global_steps)

            # moving average weight
            for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
                avg_p.mul_(0.999).add_(0.001, p.data)

            writer.add_scalar('g_loss', g_loss.item(), global_steps)
            gen_step += 1

        # verbose
        if gen_step and iter_idx % args.print_freq == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                (epoch, args.max_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss.item()))

        writer_dict['train_global_steps'] = global_steps + 1
        
    writer.add_scalar('precision_epoch', sum(tps) / (sum(tps) + sum(fps) + 1e-3), global_steps)
    writer.add_scalar('recall_epoch', sum(tps) / (sum(tps) + sum(fns) + 1e-3), global_steps)
    writer.add_scalar('accuracy_epoch', (sum(tps) + sum(tns)) / (sum(tps) + sum(tns) + sum(fps) + sum(fns) + 1e-3), global_steps)


def train_adv_g(args, gen_net: nn.Module, dis_net: nn.Module, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch, writer_dict, schedulers=None, steps=1, gamma=0.01):
    
    np.random.seed(args.random_seed + epoch ** 2)
    writer = writer_dict['writer']
    gen_step = 0

    # train mode
    gen_net = gen_net.train()
    dis_net = dis_net.train()
    tps = []
    tns = []
    fns = []
    fps = []
    
    gen_net.train()
    for iter_idx, (imgs, _) in enumerate(train_loader):
        global_steps = writer_dict['train_global_steps']

        # Adversarial ground truths
        real_imgs = imgs.type(torch.cuda.FloatTensor)
        #real_imgs = DiffAugment(real_imgs, policy="translation,cutout")
        # Sample noise as generator input
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim)))
        # ---------------------
        #  Train Discriminator
        # ---------------------
        dis_optimizer.zero_grad()

        real_validity = dis_net(real_imgs)
        
        fake_imgs = gen_net(z).detach()
        #fake_imgs = DiffAugment(fake_imgs, policy="translation,cutout")

        assert fake_imgs.size() == real_imgs.size()

        fake_validity = dis_net(fake_imgs)
        
        # cal loss
        d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                 torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
        d_loss.backward()
        dis_optimizer.step()
        writer.add_scalar('d_loss', d_loss.item(), global_steps)
        
        tp = torch.sum(real_validity > 0)
        tn = torch.sum(fake_validity < 0)
        fn = torch.sum(real_validity <= 0)
        fp = torch.sum(fake_validity >= 0)
        precision = tp / (tp + fp + 1e-3)
        recall = tp / (tp + fn + 1e-3)
        accuracy = (tp + tn) / (tp + fn + fp + tn)
        
        fps.append(fp.item())
        tps.append(tp.item())
        fns.append(fn.item())
        tns.append(tn.item())

        writer.add_scalar('precision', precision.item(), global_steps)
        writer.add_scalar('recall', recall.item(), global_steps)
        writer.add_scalar('accuracy', accuracy.item(), global_steps)
        # -----------------
        #  Train Generator
        # -----------------
        if global_steps % args.n_critic == 0:
            gen_optimizer.zero_grad()

            gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim)))

            # TRAIN ADV START
            gen_imgs_feature = gen_net(gen_z, only_l1=True)

            def forward(z):
                z = gen_net(z, l1=False)
                z = dis_net(z)
                return z

            gen_imgs_feature_adv = PGD(gen_imgs_feature, 
                model = forward, steps=steps, gamma=gamma
            )
        
            gen_imgs_feature_clean = gen_net(gen_z, only_l1=True)
            
            gen_imgs_clean = gen_net(gen_imgs_feature_clean, l1=False)
            gen_imgs_adv   = gen_net(gen_imgs_feature_adv, l1=False)
            #gen_imgs = DiffAugment(gen_imgs, policy="translation,cutout")

            fake_validity_clean = dis_net(gen_imgs_clean)
            fake_validity_adv = dis_net(gen_imgs_adv)
            # cal loss
            g_loss = (-torch.mean(fake_validity_clean) - torch.mean(fake_validity_adv)) / 2
        
            g_loss.backward()
            gen_optimizer.step()

            # adjust learning rate
            if schedulers:
                gen_scheduler, dis_scheduler = schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)
                writer.add_scalar('LR/g_lr', g_lr, global_steps)
                writer.add_scalar('LR/d_lr', d_lr, global_steps)

            # moving average weight
            for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
                avg_p.mul_(0.999).add_(0.001, p.data)

            writer.add_scalar('g_loss', g_loss.item(), global_steps)
            gen_step += 1

        # verbose
        if gen_step and iter_idx % args.print_freq == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                (epoch, args.max_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss.item()))

        writer_dict['train_global_steps'] = global_steps + 1
    
    writer.add_scalar('precision_epoch', sum(tps) / (sum(tps) + sum(fps) + 1e-3), global_steps)
    writer.add_scalar('recall_epoch', sum(tps) / (sum(tps) + sum(fns) + 1e-3), global_steps)
    writer.add_scalar('accuracy_epoch', (sum(tps) + sum(tns)) / (sum(tps) + sum(tns) + sum(fps) + sum(fns) + 1e-3), global_steps)

def train_adv_gd(args, gen_net: nn.Module, dis_net: nn.Module, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch, writer_dict, schedulers=None, steps=1, gamma=0.01):
    
    np.random.seed(args.random_seed + epoch ** 2)
    writer = writer_dict['writer']
    gen_step = 0

    # train mode
    gen_net = gen_net.train()
    dis_net = dis_net.train()
    tps = []
    tns = []
    fns = []
    fps = []
    
    gen_net.train()
    for iter_idx, (imgs, _) in enumerate(train_loader):
        global_steps = writer_dict['train_global_steps']

        # Adversarial ground truths
        real_imgs = imgs.type(torch.cuda.FloatTensor)
        #real_imgs = DiffAugment(real_imgs, policy="translation,cutout")
        # Sample noise as generator input
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim)))
        # ---------------------
        #  Train Discriminator
        # ---------------------
        dis_optimizer.zero_grad()

        clean_feature_real = dis_net(real_imgs, fc=False)
        adv_feature_real = PGD(clean_feature_real, 
                model=dis_net.l5, steps=steps, gamma=gamma)
        real_validity_clean = dis_net(clean_feature_real, fc=False, only_fc=True)
        real_validity_adv   = dis_net(adv_feature_real, fc=False, only_fc=True)
       
        #real_validity_clean = dis_net(real_imgs)
        fake_imgs = gen_net(z).detach()
        assert fake_imgs.size() == real_imgs.size()

        clean_feature_fake = dis_net(fake_imgs, fc=False)
        adv_feature_fake = PGD(clean_feature_fake, 
                model=dis_net.l5, steps=steps, gamma=gamma)
        fake_validity_clean = dis_net(clean_feature_fake, fc=False, only_fc=True)
        fake_validity_adv   = dis_net(adv_feature_fake, fc=False, only_fc=True)
        # cal loss
        d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - (real_validity_clean) )) / 2+ \
                 torch.mean(nn.ReLU(inplace=True)(1.0 + (fake_validity_clean) )) / 2+ \
                 torch.mean(nn.ReLU(inplace=True)(1.0 - (real_validity_adv) )) / 2 + \
                 torch.mean(nn.ReLU(inplace=True)(1.0 + (fake_validity_adv) )) / 2 
                 
        d_loss.backward()
        dis_optimizer.step()
        writer.add_scalar('d_loss', d_loss.item(), global_steps)
        # -----------------
        #  Train Generator
        # -----------------
        if global_steps % args.n_critic == 0:
            gen_optimizer.zero_grad()

            gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim)))

            # TRAIN ADV START
            gen_imgs_feature = gen_net(gen_z, only_l1=True)

            def forward(z):
                z = gen_net(z, l1=False)
                z = dis_net(z)
                return z

            gen_imgs_feature_adv = PGD(gen_imgs_feature, 
                model = forward, steps=steps, gamma=gamma
            )
        
            gen_imgs_feature_clean = gen_net(gen_z, only_l1=True)
            
            gen_imgs_clean = gen_net(gen_imgs_feature_clean, l1=False)
            gen_imgs_adv   = gen_net(gen_imgs_feature_adv, l1=False)

            fake_validity_clean = dis_net(gen_imgs_clean)
            fake_validity_adv = dis_net(gen_imgs_adv)
            # cal loss
            g_loss = (-torch.mean(fake_validity_clean) - torch.mean(fake_validity_adv)) / 2
        
            g_loss.backward()
            gen_optimizer.step()

            # adjust learning rate
            if schedulers:
                gen_scheduler, dis_scheduler = schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)
                writer.add_scalar('LR/g_lr', g_lr, global_steps)
                writer.add_scalar('LR/d_lr', d_lr, global_steps)

            # moving average weight
            for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
                avg_p.mul_(0.999).add_(0.001, p.data)

            writer.add_scalar('g_loss', g_loss.item(), global_steps)
            gen_step += 1

        # verbose
        if gen_step and iter_idx % args.print_freq == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                (epoch, args.max_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss.item()))

        writer_dict['train_global_steps'] = global_steps + 1
    
    writer.add_scalar('precision_epoch', sum(tps) / (sum(tps) + sum(fps) + 1e-3), global_steps)
    writer.add_scalar('recall_epoch', sum(tps) / (sum(tps) + sum(fns) + 1e-3), global_steps)
    writer.add_scalar('accuracy_epoch', (sum(tps) + sum(tns)) / (sum(tps) + sum(tns) + sum(fps) + sum(fns) + 1e-3), global_steps)



def train_gau_gd(args, gen_net: nn.Module, dis_net: nn.Module, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch, writer_dict, schedulers=None, steps=1, gamma=0.01):
    
    np.random.seed(args.random_seed + epoch ** 2)
    writer = writer_dict['writer']
    gen_step = 0

    # train mode
    gen_net = gen_net.train()
    dis_net = dis_net.train()
    tps = []
    tns = []
    fns = []
    fps = []
    
    gen_net.train()
    for iter_idx, (imgs, _) in enumerate(train_loader):
        global_steps = writer_dict['train_global_steps']

        # Adversarial ground truths
        real_imgs = imgs.type(torch.cuda.FloatTensor)
        #real_imgs = DiffAugment(real_imgs, policy="translation,cutout")
        # Sample noise as generator input
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim)))
        # ---------------------
        #  Train Discriminator
        # ---------------------
        dis_optimizer.zero_grad()

        clean_feature_real = dis_net(real_imgs, fc=False)
        with torch.no_grad():
            adv_feature_real = clean_feature_real + torch.rand(clean_feature_real.shape, device=clean_feature_real.device) * 0.1
        real_validity_clean = dis_net(clean_feature_real, fc=False, only_fc=True)
        real_validity_adv   = dis_net(adv_feature_real, fc=False, only_fc=True)
       
        #real_validity_clean = dis_net(real_imgs)
        fake_imgs = gen_net(z).detach()
        assert fake_imgs.size() == real_imgs.size()

        clean_feature_fake = dis_net(fake_imgs, fc=False)
        with torch.no_grad():
            adv_feature_fake = clean_feature_fake + torch.rand(clean_feature_real.shape, device=clean_feature_fake.device) * 0.1
        fake_validity_clean = dis_net(clean_feature_fake, fc=False, only_fc=True)
        fake_validity_adv   = dis_net(adv_feature_fake, fc=False, only_fc=True)
        # cal loss
        d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - (real_validity_clean) )) / 2+ \
                 torch.mean(nn.ReLU(inplace=True)(1.0 + (fake_validity_clean) )) / 2+ \
                 torch.mean(nn.ReLU(inplace=True)(1.0 - (real_validity_adv) )) / 2 + \
                 torch.mean(nn.ReLU(inplace=True)(1.0 + (fake_validity_adv) )) / 2 
                 
        d_loss.backward()
        dis_optimizer.step()
        writer.add_scalar('d_loss', d_loss.item(), global_steps)
        # -----------------
        #  Train Generator
        # -----------------
        if global_steps % args.n_critic == 0:
            gen_optimizer.zero_grad()

            gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim)))

            # TRAIN ADV START
            gen_imgs_feature = gen_net(gen_z, only_l1=True)

            def forward(z):
                z = gen_net(z, l1=False)
                z = dis_net(z)
                return z
            
            with torch.no_grad():
                gen_imgs_feature_adv = gen_imgs_feature + torch.rand(gen_imgs_feature.shape, device=gen_imgs_feature.device) * 0.1
        
            gen_imgs_feature_clean = gen_net(gen_z, only_l1=True)
            
            gen_imgs_clean = gen_net(gen_imgs_feature_clean, l1=False)
            gen_imgs_adv   = gen_net(gen_imgs_feature_adv, l1=False)

            fake_validity_clean = dis_net(gen_imgs_clean)
            fake_validity_adv = dis_net(gen_imgs_adv)
            # cal loss
            g_loss = (-torch.mean(fake_validity_clean) - torch.mean(fake_validity_adv)) / 2
        
            g_loss.backward()
            gen_optimizer.step()

            # adjust learning rate
            if schedulers:
                gen_scheduler, dis_scheduler = schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)
                writer.add_scalar('LR/g_lr', g_lr, global_steps)
                writer.add_scalar('LR/d_lr', d_lr, global_steps)

            # moving average weight
            for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
                avg_p.mul_(0.999).add_(0.001, p.data)

            writer.add_scalar('g_loss', g_loss.item(), global_steps)
            gen_step += 1

        # verbose
        if gen_step and iter_idx % args.print_freq == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                (epoch, args.max_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss.item()))

        writer_dict['train_global_steps'] = global_steps + 1
    
    writer.add_scalar('precision_epoch', sum(tps) / (sum(tps) + sum(fps) + 1e-3), global_steps)
    writer.add_scalar('recall_epoch', sum(tps) / (sum(tps) + sum(fns) + 1e-3), global_steps)
    writer.add_scalar('accuracy_epoch', (sum(tps) + sum(tns)) / (sum(tps) + sum(tns) + sum(fps) + sum(fns) + 1e-3), global_steps)




def tensor_clamp(t, min, max, in_place=True):
    if not in_place:
        res = t.clone()
    else:
        res = t
    idx = res.data < min
    res.data[idx] = min[idx]
    idx = res.data > max
    res.data[idx] = max[idx]

    return res

def linfball_proj(center, radius, t, in_place=True):
    return tensor_clamp(t, min=center - radius, max=center + radius, in_place=in_place)

def PGD(x, model=None, steps=1, gamma=0.1, eps=(1/255), randinit=False, clip=False):
    
    # Compute loss
    x_adv = x.clone()
    if randinit:
        # adv noise (-eps, eps)
        x_adv += (2.0 * torch.rand(x_adv.shape).cuda() - 1.0) * eps
    x_adv = x_adv.cuda()
    x = x.cuda()

    for t in range(steps):

        out = model(x_adv)
        loss_adv0 = torch.mean(out)
        grad0 = torch.autograd.grad(loss_adv0, x_adv, only_inputs=True)[0]
        x_adv.data.add_(gamma * torch.sign(grad0.data))

        if clip:
            linfball_proj(x, eps, x_adv, in_place=True)

    return x_adv
