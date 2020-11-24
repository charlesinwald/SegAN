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
from net import NetS, NetC
from LoadDataCovid import Dataset, loader
import math
import nibabel as nib
#from nilearn import plotting


# Training settings
parser = argparse.ArgumentParser(description='Example')
parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
parser.add_argument('--niter', type=int, default=10000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate. Default=0.02')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use, for now it only supports one GPU')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--decay', type=float, default=0.5, help='Learning rate decay. default=0.5')
parser.add_argument('--cuda', action='store_true', help='using GPU or not')
parser.add_argument('--seed', type=int, default=666, help='random seed to use. Default=1111')
parser.add_argument('--outpath', default='./outputs', help='folder to output images and model checkpoints')
opt = parser.parse_args()
opt.cuda = True
device = 'cuda:0'
#device = 'cpu'
print(opt)

argED = {
    'width': 32,
    'latent_width': 4,
    'depth': 16,
    'latent': 2,
    'colors': 1,
    'lr': 1e-4,
}

modelG_file = 'generator.pt'
modelD_file = 'discriminator.pt'

activation = nn.LeakyReLU


# authors use this initializer, but it doesn't seem essential
def Initializer(layers, slope=0.2):
    for layer in layers:
        if hasattr(layer, 'weight'):
            w = layer.weight.data
            std = 1 / np.sqrt((1 + slope ** 2) * np.prod(w.shape[:-1]))
            w.normal_(std=std)
        if hasattr(layer, 'bias'):
            layer.bias.data.zero_()


def Encoder(scales, depth, latent, colors):
    layers = []
    layers.append(nn.Conv2d(colors, depth, 1, padding=1))
    kp = depth
    for scale in range(scales):
        k = depth << scale
        layers.extend([nn.Conv2d(kp, k, 3, padding=1), activation()])
        layers.extend([nn.Conv2d(k, k, 3, padding=1), activation()])
        layers.append(nn.AvgPool2d(2))
        kp = k
    k = depth << scales
    layers.extend([nn.Conv2d(kp, k, 3, padding=1), activation()])
    layers.append(nn.Conv2d(k, latent, 3, padding=1))
    Initializer(layers)
    return nn.Sequential(*layers)


def Decoder(scales, depth, latent, colors):
    layers = []
    kp = latent
    for scale in range(scales - 1, -1, -1):
        k = depth << scale
        layers.extend([nn.Conv2d(kp, k, 3, padding=1), activation()])
        layers.extend([nn.Conv2d(k, k, 3, padding=1), activation()])
        layers.append(nn.Upsample(scale_factor=2))
        kp = k
    layers.extend([nn.Conv2d(kp, depth, 3, padding=1), activation()])
    layers.append(nn.Conv2d(depth, colors, 3, padding=1))
    Initializer(layers)
    return nn.Sequential(*layers)


def load_checkpoint_for_eval(model, max_iou, filename, device):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename, map_location=device)
        start_epoch = checkpoint['epoch']
        max_iou = checkpoint['max_dice']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(filename, checkpoint['epoch']))
        del checkpoint
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, start_epoch, max_iou


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
# load testing data
Testing_data = Dataset(data_path, 'Validation')
dataloader_val = loader(Testing_data, opt.batchSize, shuffle=False)

cudnn.benchmark = True
print('===> Building model')
NetS = NetS(ngpu=opt.ngpu)
# scales = int(round(math.log(argED['width'] // argED['latent_width'], 2)))
# encoder = Encoder(scales, argED['depth'], argED['latent'], argED['colors'])
# decoder = Decoder(scales, argED['depth'], argED['latent'], argED['colors'])
# NetS.apply(weights_init)
print(NetS)

if cuda:
    NetS = NetS.cuda()
    # encoder=encoder.cuda()
    # decoder=decoder.cuda()

    # criterion = criterion.cuda()


max_iou = 0
NetS, start_epoch, max_iou=load_checkpoint_for_eval(NetS, max_iou, modelG_file,device)
test_res=[]
total_sum=0
total_dices=0
for i, data in enumerate(dataloader_val, 1):
    input, gt = Variable(data[0]), Variable(data[1])
    gt = gt.type(torch.FloatTensor)

    if cuda:
        input = input.cuda()
        gt = gt.cuda()

    pred = NetS(input)
    pred = F.sigmoid(pred)
    # pred=pred.squeeze(1)
    # gt = gt.squeeze(1)
    # print(pred.shape)

    # encoded_input = encoder(input)
    # pred = decoder(encoded_input)
    pred[pred < 0.5] = 0
    pred[pred >= 0.5] = 1
    # pred = pred.type(torch.LongTensor)
    # if cuda:
    #     pred=pred.cuda()
    pred_np = pred.data.cpu().numpy()
    gt = gt.data.cpu().numpy()

    dice = 1 - dice_loss_numpy(pred_np, gt)
    # print(dice)

    total_dices += dice * input.size()[0]

    total_sum += input.size()[0]

    #print(pred.size())

    pred_np=pred.squeeze(1).permute(1, 2,0).data.cpu().numpy()

    #print(pred_np.shape)

    test_res.append(pred_np)

test_res = np.concatenate(test_res,axis=2)
step = 5
w=512
h=512
d=500
for index,(start, end) in enumerate(Testing_data.slice_indices):
    test_volume = test_res[:, :, start:end]
    # print(test_volume)
    # print(test_volume.shape)

    #image = nib.load(Testing_data.input_paths[index])
    #image=image.get_data()
    #image = image[:w, :h, :d]
    # image = self.img_transform(image)
    #row_size = image.shape[0]
    #col_size = image.shape[1]
    #image = image[int(col_size / 5):int(col_size / 5 * 4),
    #        int(row_size / 5):int(row_size / 5 * 4), :]  # Lung image
    #output_seg=image*test_volume

    masked_lung_img_nii = nib.Nifti1Image(test_volume, affine=np.eye(4))
    head, tail = os.path.split(Testing_data.input_paths[index])
    file_to_save=tail

    nib.save(masked_lung_img_nii, os.path.join('outputs_segmentation', file_to_save))

    # Visualize slices
    #plotting.plot_anat(masked_lung_img_nii, display_mode='z', cut_coords=range(0, masked_lung_img_nii.shape[2], step),
    #                   title='Slices')

    #plotting.show()
