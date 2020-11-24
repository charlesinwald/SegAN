import os
import numpy as np
from glob import glob
from PIL import Image
import torch
from torchvision.transforms import Compose, CenterCrop, Normalize, ToTensor,Resize
from transform import ReLabel, ToLabel, Scale, HorizontalFlip, VerticalFlip, ColorJitter
from torchvision import transforms
import random
import nibabel as nib
from tqdm import tqdm

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

class Dataset(torch.utils.data.Dataset):

    def __init__(self, root,folder_name):
        self.size = (180,135)
        self.root = root
        self.mean=0.5
        self.std=0.5
        #self.transform = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        self.transform = transforms.Compose([transforms.ToTensor()])
        if not os.path.exists(self.root):
            raise Exception("[!] {} not exists.".format(root))

        self.img_transform = Compose([

            ToTensor(),
            #Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        ])

        #sort file names
        self.input_paths = sorted(glob(os.path.join(self.root, '{}/*ct.nii.gz'.format(folder_name))))
        self.label_paths = sorted(glob(os.path.join(self.root, '{}/*seg.nii.gz'.format(folder_name))))
        self.name = os.path.basename(root)
        if len(self.input_paths) == 0 or len(self.label_paths) == 0:
            raise Exception("No images/labels are found in {}".format(self.root))

        train_data=[]
        w=512
        h=512
        d=500

        start=0
        self.slice_indices=[]

        with tqdm(total=len(self.input_paths)) as pbar0:

            for p in self.input_paths:
                image = nib.load(p)
                image = image.get_data()
                image=image[:w,:h,:d]
                end=image.shape[2]+start
                self.slice_indices.append((start,end))
                start=end
                #image = self.img_transform(image)
                row_size = image.shape[0]
                col_size = image.shape[1]
                #image = image[int(col_size / 5):int(col_size / 5 * 4),
                #             int(row_size / 5):int(row_size / 5 * 4),:]  # Lung image


                #image[image<=-1000]=-1000.0
                #image[image>=500]=500.0

                #image=(image+1000)/1500.0
                #image=image-0.224
                #image=image/0.306

                mean = np.mean(image)
                std = np.std(image)
                image = image - mean
                image = image / std

                # for i in range(image.shape[2]):
                #     img=image[:,:,i]
                #     row_size = img.shape[0]
                #     col_size = img.shape[1]
                #
                #     mean = np.mean(img)
                #     std = np.std(img)
                #     img = img - mean
                #     img = img / std
                #
                #     # Find the average pixel value near the lungs
                #     # to renormalize washed out images
                #     middle = img[int(col_size / 5):int(col_size / 5 * 4),
                #              int(row_size / 5):int(row_size / 5 * 4)]  # Lung image
                #     #im = self.transform(image[:,:,i])

                image = self.transform(image)
                image = image.type(torch.FloatTensor)
                train_data.append(image)
                pbar0.update(1)

        train_data=torch.cat(train_data)
        train_data=train_data.unsqueeze(1)
        self.train_data=train_data

        print(torch.mean(train_data))
        print(torch.std(train_data))

        train_labels = []
        with tqdm(total=len(self.input_paths)) as pbar1:
            for p in self.label_paths:
                label = nib.load(p)
                label = label.get_data()
                #label = label[:w,:h,:d]
                row_size = label.shape[0]
                col_size = label.shape[1]
                #label = label[int(col_size / 5):int(col_size / 5 * 4),
                #        int(row_size / 5):int(row_size / 5 * 4), :]  # Lung image
                label = label.astype(int)
                #label = self.label_transform(label)
                label = self.img_transform(label)
                train_labels.append(label)

                pbar1.update(1)

        train_labels = torch.cat(train_labels)
        train_labels = train_labels.unsqueeze(1)
        self.train_labels = train_labels


    def __getitem__(self, index):


        # image = nib.load(self.input_paths[index])
        # image = image.get_data()
        # label = nib.load(self.label_paths[index])
        # label = label.get_data()
        #
        # image = self.img_transform(image)
        # label = self.label_transform(label)

        #return image, label


        return self.train_data[index],self.train_labels[index]



    def __len__(self):
        #return len(self.input_paths)
        return len(self.train_data)




def loader(dataset, batch_size, num_workers=8, shuffle=True):

    input_images = dataset

    input_loader = torch.utils.data.DataLoader(dataset=input_images,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                num_workers=num_workers)

    return input_loader
