# -*- coding: utf-8 -*-
# @Date    : 2019-07-25
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

import os
import numpy as np
import torch
import torch.nn as nn
import logging
logger = logging.getLogger(__name__)

    
def train_adv_d(args, gen_net: nn.Module, dis_net: nn.Module, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch, writer_dict, schedulers=None, steps=1, gamma=0.01):
    
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
