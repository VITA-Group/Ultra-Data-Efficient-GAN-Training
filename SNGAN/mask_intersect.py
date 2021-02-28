import torch
import torch.nn.utils.prune as prune

for i in range(10):
    inter = 0
    uni = 0
    mask_100 = torch.load('logs/impg/sngan_cifar10_impg_0.8_imp_2020_09_17_04_03_31/Model/checkpoint_{}.pth'.format(i))
    mask_10 = torch.load('logs/sngan_cifar10_less_0.8_imp_2020_10_17_08_12_05/Model/checkpoint_{}.pth'.format(i))

    mask_100_weight = mask_100['gen_state_dict']
    mask_10_weight = mask_10['gen_state_dict']

    for name in mask_100_weight.keys():
        if 'mask' in name:
            m1 = mask_100_weight[name]
            m2 = mask_10_weight[name]
        
            intersect = m1 * m2
            union = torch.clamp(m1 + m2, max = 1)
            inter += torch.sum(intersect.view(-1)).item()
            uni += torch.sum(union.view(-1)).item()

    print("{}: {}/{}, {}".format(i, inter, uni, inter / uni * 1.0))