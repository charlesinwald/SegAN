from mayavi import mlab
import numpy as np
import nibabel as nib
import os
from nilearn import plotting
import SimpleITK as sitk
from matplotlib import pyplot as plt
import time

# two different segmentation methods

#define six connectivity for region growing of volume data
def six_connectivity(center, visited_pixels, volume_dimensions):
    center_neighbors = []

    if (center[0] < volume_dimensions[0]-1) and not visited_pixels[center[0]+1, center[1], center[2]]:
        center_neighbors.append((center[0]+1, center[1], center[2]))
    if (center[1] < volume_dimensions[1]-1) and not visited_pixels[center[0], center[1]+1, center[2]]:
        center_neighbors.append((center[0], center[1]+1, center[2]))
    if (center[2] < volume_dimensions[2]-1) and not visited_pixels[center[0], center[1], center[2]+1]:
        center_neighbors.append((center[0], center[1], center[2]+1))

    if (center[0] > 0) and not visited_pixels[center[0]-1, center[1], center[2]]:
        center_neighbors.append((center[0]-1, center[1], center[2]))
    if (center[1] > 0) and not visited_pixels[center[0], center[1]-1, center[2]]:
        center_neighbors.append((center[0], center[1]-1, center[2]))
    if (center[2] > 0) and not visited_pixels[center[0], center[1], center[2]-1]:
        center_neighbors.append((center[0], center[1], center[2]-1))

    return center_neighbors

#region growing function for segmentation of 3D data. The inputs are the volume data, the seed point, and the global threshold
# for the growing criteria
def apply_region_grow(img_3D, seed,threshold):

    segmentation_result = np.zeros(img_3D.shape, dtype=np.bool)
    visited_pixels = np.zeros_like(segmentation_result)

    segmentation_result[seed] = True
    visited_pixels[seed] = True
    queue = six_connectivity(seed, visited_pixels, img_3D.shape)
    p_seed=img_3D[seed]

    while len(queue) > 0:
        current_pixel = queue.pop()

        if visited_pixels[current_pixel]:
            continue

        visited_pixels[current_pixel] = True

        if abs(img_3D[current_pixel] - p_seed) <= threshold:
            segmentation_result[current_pixel] = True
            queue += six_connectivity(current_pixel, visited_pixels, img_3D.shape)

    return segmentation_result


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

imgOriginal = sitk.GetImageFromArray(org_volume)
#plt.imshow(org_volume[:,:,50])
#plt.show()


# set the seed point
seed_point = (166, 308,30)
seed_point2 = (360, 280,30)
threshold=105

seed_airway = (256, 241,50)
threshold_airway = 105



print('please wait')

# set threshold of region growing criteria for airway


# call region_growing function for airway segmentation
segmented_result = apply_region_grow(org_volume, seed_point, threshold)
segmented_result2 = apply_region_grow(org_volume, seed_point2, threshold)
segmented_result_airway = apply_region_grow(org_volume, seed_airway, threshold_airway)


# show segmentation results of airway and lungs
mlab.figure(2)

airway_surface = mlab.pipeline.scalar_field(segmented_result_airway.astype(np.float))
lung1_surface = mlab.pipeline.scalar_field(segmented_result.astype(np.float))
lung2_surface = mlab.pipeline.scalar_field(segmented_result2.astype(np.float))
covid = mlab.pipeline.scalar_field(image.astype(np.float))

mlab.pipeline.iso_surface(airway_surface, contours=[0.5], opacity=1)
# set the same opacity and color for both lungs
mlab.pipeline.iso_surface(lung1_surface, contours=[0.5], opacity=0.05, color=(.5, .3, .1))
mlab.pipeline.iso_surface(lung2_surface, contours=[0.5], opacity=0.05, color=(.5, .3, .1))
mlab.pipeline.iso_surface(covid, contours=[0.5], opacity=1,color=(.5, .0, .0))

mlab.figure(3)
airway_surface = mlab.pipeline.scalar_field(segmented_result_airway.astype(np.float))
lung1_surface = mlab.pipeline.scalar_field(segmented_result.astype(np.float))
lung2_surface = mlab.pipeline.scalar_field(segmented_result2.astype(np.float))
covid_test = mlab.pipeline.scalar_field(image_test.astype(np.float))

mlab.pipeline.iso_surface(airway_surface, contours=[0.5], opacity=1)
# set the same opacity and color for both lungs
mlab.pipeline.iso_surface(lung1_surface, contours=[0.5], opacity=0.05, color=(.5, .3, .1))
mlab.pipeline.iso_surface(lung2_surface, contours=[0.5], opacity=0.05, color=(.5, .3, .1))
mlab.pipeline.iso_surface(covid_test, contours=[0.5], opacity=1,color=(.5, .0, .0))

mlab.show()