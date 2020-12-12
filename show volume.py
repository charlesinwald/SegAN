from mayavi import mlab
import numpy as np
import nibabel as nib
import os
from nilearn import plotting


#data_path='/home/mohamed/PycharmProjects/COVID-19-20/COVID-19-20_v2/valid_small'
data_path_test='outputs_segmentation_val'
#data_path='/home/mohamed/PycharmProjects/inter_seggan/SampleSubmission'
#data_path='/home/mohamed/PycharmProjects/COVID-19-20/COVID-19-20_v2/Validation'
#data_path='/home/mohamed/PycharmProjects/COVID-19-20/COVID-19-20_v2/Train'
data_path='val'

#file_name=os.path.join(data_path,'volume-covid19-A-0003_ct.nii.gz')
seg_file=os.path.join(data_path,'volume-covid19-A-0288_seg.nii.gz')
#seg_file=os.path.join(data_path,'513_seg.nii.gz')
seg_file_test=os.path.join(data_path_test,'volume-covid19-A-0288_ct.nii.gz')

#seg_file=os.path.join(data_path,'volume-covid19-A-0164_seg.nii.gz')

original_volume_file=os.path.join(data_path,'volume-covid19-A-0288_ct.nii.gz')

image_test = nib.load(seg_file_test)
image_test = image_test.get_data()


image = nib.load(seg_file)
image = image.get_data()

org_volume = nib.load(original_volume_file)
org_volume = org_volume.get_data()
org_volume=org_volume.astype(np.float)
image=image.astype(np.float)
#image_test=image_test.astype(np.float)

masked_image_gt=org_volume*image
masked_image_generated=org_volume*image_test

# Visualize slices

# plotting.plot_anat(nib.Nifti1Image(org_volume, affine=np.eye(4)), display_mode='z', cut_coords=range(0, masked_image_gt.shape[2], 1),
#                    title='Slices')
# plotting.plot_anat(nib.Nifti1Image(masked_image_gt, affine=np.eye(4)), display_mode='z', cut_coords=range(0, masked_image_gt.shape[2], 1),
#                    title='Slices')
#
# plotting.plot_anat(nib.Nifti1Image(masked_image_generated, affine=np.eye(4)), display_mode='z', cut_coords=range(0, masked_image_gt.shape[2], 1),
#                    title='Slices')
#
# plotting.plot_anat(nib.Nifti1Image(image, affine=np.eye(4)), display_mode='z', cut_coords=range(0, masked_image_gt.shape[2], 1),
#                    title='Slices')
# plotting.plot_anat(nib.Nifti1Image(image_test, affine=np.eye(4)), display_mode='z', cut_coords=range(0, masked_image_gt.shape[2], 1),
#                    title='Slices')



plotting.show()

mlab.figure(1)
org_volume_ = mlab.pipeline.scalar_field(org_volume.astype(np.float))
mlab.pipeline.iso_surface(org_volume_, contours=[0.5], opacity=1, color=(.8, .2, .1))


mlab.figure(2)
image_ = mlab.pipeline.scalar_field(image.astype(np.float))
mlab.pipeline.iso_surface(image_, contours=[0.5], opacity=1, color=(.8, .2, .1))

mlab.figure(3)
image_test_ = mlab.pipeline.scalar_field(image_test.astype(np.float))
mlab.pipeline.iso_surface(image_test_, contours=[0.5], opacity=1, color=(.8, .2, .1))

mlab.figure(4)
masked_image_gt_ = mlab.pipeline.scalar_field(masked_image_gt.astype(np.float))
mlab.pipeline.iso_surface(masked_image_gt_, contours=[0.5], opacity=0.5, color=(.8, .2, .1))

mlab.figure(5)
masked_image_generated_ = mlab.pipeline.scalar_field(masked_image_generated.astype(np.float))
mlab.pipeline.iso_surface(masked_image_generated_, contours=[0.5], opacity=1, color=(.8, .2, .1))

mlab.show()