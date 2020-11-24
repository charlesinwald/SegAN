import os
from glob import glob
import numpy as np
import random
import shutil

root='/home/mohamedt/COVID-19-20/COVID-19-20_v2'
folder_name='Train'
valid_folder=os.path.join(root,'val')

input_paths = np.array(sorted(glob(os.path.join(root, '{}/*ct.nii.gz'.format(folder_name)))))
label_paths = np.array(sorted(glob(os.path.join(root, '{}/*seg.nii.gz'.format(folder_name)))))

indices=np.arange(len(input_paths))
random.shuffle(indices)
frac=0.8
nb_train=int(frac*len(input_paths))
input_paths=input_paths[indices]
label_paths=label_paths[indices]
valid_paths=input_paths[nb_train:]
valid_labels=label_paths[nb_train:]

for path in valid_paths:
    shutil.move(path,valid_folder)

for path in valid_labels:
    shutil.move(path,valid_folder)

