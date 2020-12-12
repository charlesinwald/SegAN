# CUDA_VISIBLE_DEVICES=X python train.py --cuda --outpath ./outputs
from __future__ import print_function
import argparse
import os
import numpy as np
from PIL import Image
import torch
from torch.utils import data
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch import nn
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
from net import NetS, NetC,NetC_BCE
from LoadDataCovid import Dataset, loader
import math
from tqdm import tqdm
import shutil

# Training settings
parser = argparse.ArgumentParser(description='Example')
parser.add_argument('--batchSize', type=int, default=15, help='training batch size')
parser.add_argument('--niter', type=int, default=10000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.002, help='Learning Rate. Default=0.02')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use, for now it only supports one GPU')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--decay', type=float, default=0.5, help='Learning rate decay. default=0.5')
parser.add_argument('--cuda', action='store_true', help='using GPU or not')
parser.add_argument('--seed', type=int, default=666, help='random seed to use. Default=1111')
parser.add_argument('--outpath', default='./outputs', help='folder to output images and model checkpoints')
opt = parser.parse_args()
opt.cuda = True
device = 'cuda:0'
# device='cpu'
print(opt)

gammas = [0.5, 0.5, 0.5, 0.5]
schedule = [50, 100, 150, 200]
modelG_file = 'generator_bce.pt'
modelG_file_best = 'generator_bce_best.pt'
modelD_file = 'discriminator_bce.pt'

activation = nn.LeakyReLU
evel_every = 1

def save_checkpoint(checkpoint, is_best, bestmodel):
    if is_best:
        shutil.copyfile(checkpoint, bestmodel)



def load_checkpoint(model, optimizer, max_dice, filename, device):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename, map_location=device)
        start_epoch = checkpoint['epoch']
        max_dice = checkpoint['max_dice']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(filename, checkpoint['epoch']))
        del checkpoint
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch, max_dice


def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = opt.lr
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if (epoch >= step):
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


try:
    os.makedirs(opt.outpath)
except OSError:
    pass


# custom weights initialization called on NetS and NetC
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def dice_loss(input, target):
    num = input * target

    num = torch.sum(num, dim=2)

    num = torch.sum(num, dim=2)

    den1 = input * input
    den1 = torch.sum(den1, dim=2)
    den1 = torch.sum(den1, dim=2)

    den2 = target * target
    den2 = torch.sum(den2, dim=2)
    den2 = torch.sum(den2, dim=2)

    dice = 2 * (num / (den1 + den2 + 0.0001))

    dice_total = 1 - 1 * torch.sum(dice) / dice.size(0)  # divide by batchsize

    return dice_total


def dice_loss_numpy(input, target):
    num = input * target

    num = np.sum(num, axis=2)

    num = np.sum(num, axis=2)

    den1 = input * input
    den1 = np.sum(den1, axis=2)
    den1 = np.sum(den1, axis=2)

    den2 = target * target
    den2 = np.sum(den2, axis=2)
    den2 = np.sum(den2, axis=2)

    # indices_to_use=[i for i in range(len(den2)) if den2[i]!=0]
    #
    # num=np.array([num[i] for i in range(len(num)) if i in indices_to_use])
    # den2 = np.array([den2[i] for i in range(len(den2)) if i in indices_to_use])
    # den1 = np.array([den1[i] for i in range(len(den1)) if i in indices_to_use])

    dice = 2 * (num / (den1 + den2 + 0.0001))
    # print(num)
    # print(den1)
    # print(den2)
    # print(dice)

    dice_total = 1 - 1 * np.sum(dice) / dice.shape[0]  # divide by batchsize

    return dice_total


cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

# load training data
data_path = '/home/mohamedt/COVID-19-20/COVID-19-20_v2'
dataloader = loader(Dataset(data_path, 'Train'), opt.batchSize)
# load testing data
Testing_data = Dataset(data_path, 'val')
dataloader_val = loader(Testing_data, opt.batchSize, shuffle=False)

cudnn.benchmark = True
print('===> Building model')
NetS = NetS(ngpu=opt.ngpu)
# NetS.apply(weights_init)
print(NetS)
NetC = NetC_BCE(ngpu=opt.ngpu)
# NetC.apply(weights_init)
print(NetC)

if cuda:
    NetS = NetS.cuda()
    NetC = NetC.cuda()

    # criterion = criterion.cuda()

# setup optimizer
lr = opt.lr
decay = opt.decay
optimizerG = optim.Adam(NetS.parameters(), lr=lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
# optimizerG = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr, weight_decay=1e-5)
optimizerD = optim.Adam(NetC.parameters(), lr=lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)

max_dice = 0
NetS, optimizerG, _, _ = load_checkpoint(NetS, optimizerG, max_dice, modelG_file, device)
NetC, optimizerD, start_epoch, max_dice = load_checkpoint(NetC, optimizerD, max_dice, modelD_file, device)
print('max_dice={}'.format(max_dice))

all_eval = []
all_G_loss = []
loss = nn.MSELoss()
adversarial_loss = torch.nn.BCELoss()

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

NetS.train()
for epoch in range(start_epoch + 1, start_epoch + opt.niter + 1):

    epoch_loss = 0
    epoch_loss_D = 0
    current_learning_rate = adjust_learning_rate(optimizerG, epoch, gammas, schedule)
    current_learning_rate = adjust_learning_rate(optimizerD, epoch, gammas, schedule)

    with tqdm(total=len(dataloader)) as pbar0:
        for i, data in enumerate(dataloader, 1):

            # Adversarial ground truths
            valid = Variable(Tensor(data[0].size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(data[0].size(0), 1).fill_(0.0), requires_grad=False)

            # train C
            # NetC.zero_grad()
            input, label = Variable(data[0]), Variable(data[1])
            label = label.type(torch.FloatTensor)
            if cuda:
                input = input.cuda()
                target = label.cuda()
            else:
                target = label

            output = NetS(input)
            output = F.sigmoid(output)
            output = output.detach()
            input_mask = input.clone()
            output_masked = input_mask * output
            if cuda:
                output_masked = output_masked.cuda()
            # print(output_masked.shape)
            result = NetC(output_masked)
            target_masked = input_mask * target

            if cuda:
                target_masked = target_masked.cuda()
            # print(target_masked.shape)
            target_D = NetC(target_masked)


            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(target_D, valid)
            fake_loss = adversarial_loss(result, fake)
            loss_D = (real_loss + fake_loss) / 2

            epoch_loss_D += loss_D

            optimizerD.zero_grad()
            loss_D.backward()
            optimizerD.step()
            # clip parameters in D
            for p in NetC.parameters():
                p.data.clamp_(-0.05, 0.05)
            # train G

            output = NetS(input)
            output = F.sigmoid(output)

            output_masked = input_mask * output

            if cuda:
                output_masked = output_masked.cuda()
            result = NetC(output_masked)
            target_masked = input_mask * target

            if cuda:
                target_masked = target_masked.cuda()
            target_G = NetC(target_masked)
            loss_dice = dice_loss(output, target)
            # Loss measures generator's ability to fool the discriminator
            loss_G = adversarial_loss(result, valid)

            loss_G_joint = loss_G + loss_dice
            epoch_loss += loss_G_joint

            optimizerG.zero_grad()
            loss_G_joint.backward()
            optimizerG.step()

            pbar0.update(1)

    print("===> Epoch[{}]({}/{}): Batch Dice: {:.4f}".format(epoch, i, len(dataloader), 1 - loss_dice))
    print("===> Epoch[{}]({}/{}): G_Loss: {:.4f}".format(epoch, i, len(dataloader), epoch_loss / i))
    print("===> Epoch[{}]({}/{}): D_Loss: {:.4f}".format(epoch, i, len(dataloader), epoch_loss_D / i))

    all_G_loss.append(epoch_loss.item() / i)

    if epoch % evel_every == 0:
        print('===> start the evaluation')
        NetS.eval()
        IoUs, dices = [], []

        total_sum = 0
        total_dices = 0
        with tqdm(total=len(dataloader_val)) as pbar1:
            for i, data in enumerate(dataloader_val, 1):
                input, gt = Variable(data[0]), Variable(data[1])
                gt = gt.type(torch.FloatTensor)

                if cuda:
                    input = input.cuda()
                    gt = gt.cuda()

                pred = NetS(input)

                pred = F.sigmoid(pred)

                pred[pred < 0.5] = 0
                pred[pred >= 0.5] = 1

                pred_np = pred.data.cpu().numpy()
                gt = gt.data.cpu().numpy()

                l = 0
                for slice in range(gt.shape[0]):
                    if np.sum(gt[slice]) > 0:
                        l += 1

                dice = 1 - dice_loss_numpy(pred_np, gt)

                total_dices += dice * input.size()[0]
                total_sum += l

                pbar1.update(1)

        NetS.train()

        dices = total_dices / total_sum

        print('Dice: {:.4f}'.format(dices))
        print('done with epoch {}, and learning rate={}'.format(epoch, current_learning_rate))
        all_eval.append(dices)
        print(all_eval)
        print(all_G_loss)

        is_best = False
        if dices > max_dice:
            is_best = True
            max_dice = dices

        state = {'epoch': epoch, 'state_dict': NetS.state_dict(),
                 'optimizer': optimizerG.state_dict(), 'max_dice': max_dice}
        torch.save(state, modelG_file)

        state = {'epoch': epoch, 'state_dict': NetC.state_dict(),
                 'optimizer': optimizerD.state_dict(), 'max_dice': max_dice}
        torch.save(state, modelD_file)

        save_checkpoint(modelG_file, is_best, modelG_file_best)

