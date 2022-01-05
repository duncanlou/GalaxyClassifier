import numpy as np
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.visualization import LinearStretch, SinhStretch, ImageNormalize, ZScaleInterval, SqrtStretch, \
    SquaredStretch, PowerStretch, LogStretch, ContrastBiasStretch, MinMaxInterval, PercentileInterval, AsinhStretch
from ccdproc import CCDData
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from astropy.io import fits
import os
import matplotlib.pyplot as plt
from photutils import MedianBackground, Background2D

from utils import have_source_in_center
from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma
from photutils.segmentation import detect_sources, detect_threshold, make_source_mask

from utils import CatPSimgMinMax

import properimage.single_image as si
from properimage.operations import subtract

ROOT = '/Users/loukangzhi/PycharmProjects/GalaxyClassifier/data/sources/GALAXY/170.83644m3.0926838'

a = os.path.exists(ROOT)
files = [os.path.join(ROOT, f) for f in os.listdir(ROOT)]
files = sorted(files)
image_dat = []

image_cube_vmax = 0
fig = plt.figure()

for i in range(5):
    one_channel = files[i]
    image = fits.getdata(one_channel)
    norm = ImageNormalize(image, interval=ZScaleInterval(), stretch=LinearStretch())
    ax = fig.add_subplot(2, 3, i + 1)
    plt.imshow(image, origin='lower', cmap='Greys_r', norm=norm, interpolation='nearest')

plt.tight_layout()
plt.show()

fig = plt.figure()
for i in range(5):
    one_channel = files[i]
    image = fits.getdata(one_channel)
    center_region = image[100:140, 100:140]
    _, vmax = CatPSimgMinMax(center_region)
    single_channel_img_dat = np.where(image > vmax, vmax, image)
    if vmax > image_cube_vmax:
        image_cube_vmax = vmax
    image_dat.append(image)

image_cube = np.stack(image_dat, axis=2)
image_cube = np.transpose(image_cube, (2, 0, 1))
print(image_cube.shape)

image_cube = LinearStretch().__call__(ZScaleInterval().__call__(image_cube))


for i in range(5):
    single_channel_img_dat = image_cube[i]
    ax = fig.add_subplot(2, 3, i + 1)
    plt.imshow(single_channel_img_dat, origin='lower', cmap='Greys_r', interpolation='nearest')

plt.tight_layout()
plt.show()




# print("image cube center area's image_cube_vmax: ", image_cube_vmax)
#
# image_cube = np.stack(image_dat, axis=2)
# print(image_cube.shape)
# image_cube = np.transpose(image_cube, (2, 0, 1))
# print(image_cube.shape)
# max = np.max(image_cube)
# min = np.min(image_cube)
# print("cube_min:", min, " cube_max:", max)
#
# image_cube_vmin, kaka = CatPSimgMinMax(image_cube)
# print("image_cube_vmin: ", image_cube_vmin, "kaka: ", kaka)
#
# image_cube = (image_cube - min) / (image_cube_vmax - min)
#
#
# # strech = SinhStretch()
# #
# #
# # image_cube1 = strech.__call__(image_cube)
# print("--------------------------------------------------------")
#
#
# for i in range(5):
#     fig = plt.figure()
#     im = plt.imshow(image_cube[i], origin='lower')
#     fig.colorbar(im)
#     plt.show()
