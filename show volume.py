from mayavi import mlab
import numpy as np
import nibabel as nib
import os

#data_path='/home/mohamed/PycharmProjects/COVID-19-20/COVID-19-20_v2/valid_small'
data_path_test='outputs_segmentation'
#data_path='/home/mohamed/PycharmProjects/inter_seggan/SampleSubmission'
#data_path='/home/mohamed/PycharmProjects/COVID-19-20/COVID-19-20_v2/Validation'
data_path='/home/mohamed/PycharmProjects/COVID-19-20/COVID-19-20_v2/Train'

#file_name=os.path.join(data_path,'volume-covid19-A-0003_ct.nii.gz')
seg_file=os.path.join(data_path,'volume-covid19-A-0255_seg.nii.gz')
#seg_file=os.path.join(data_path,'513_seg.nii.gz')
seg_file_test=os.path.join(data_path_test,'volume-covid19-A-0451_ct.nii.gz')

#seg_file=os.path.join(data_path,'volume-covid19-A-0164_seg.nii.gz')

image_test = nib.load(seg_file_test)
image_test = image_test.get_data()


image = nib.load(seg_file)
image = image.get_data()


mlab.figure(1)
image_test_ = mlab.pipeline.scalar_field(image_test.astype(np.float))
mlab.pipeline.iso_surface(image_test_, contours=[0.5], opacity=1, color=(.8, .2, .1))

mlab.figure(2)
image_ = mlab.pipeline.scalar_field(image.astype(np.float))
mlab.pipeline.iso_surface(image_, contours=[0.5], opacity=1, color=(.8, .2, .1))

mlab.show()