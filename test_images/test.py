import numpy as np
from astropy.stats import sigma_clipped_stats
from astropy.visualization import LinearStretch, SinhStretch, ImageNormalize, ZScaleInterval, SqrtStretch, \
    SquaredStretch, PowerStretch, LogStretch, ContrastBiasStretch, MinMaxInterval, PercentileInterval, AsinhStretch
from ccdproc import CCDData
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from regions import CirclePixelRegion, make_example_dataset, PixCoord
from astropy.io import fits
import os
import matplotlib.pyplot as plt

from utils import have_source_in_center
from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma
from photutils.segmentation import detect_sources, detect_threshold

from utils import CatPSimgMinMax

norm = ImageNormalize(stretch=SqrtStretch())
ROOT = '/home/duncan/PycharmProjects/MyResearchProject_Duncan/data/sources/GALAXY/170.82828m3.3732091'

files = [os.path.join(ROOT, f) for f in os.listdir(ROOT)]
files = sorted(files)
image_dat = []

image_cube_vmax = 0

for i in range(5):
    one_channel = files[i]
    image = fits.getdata(one_channel)
    b = have_source_in_center(image)
    print(b)
    center_region = image[100:140, 100:140]
    vmin, vmax = CatPSimgMinMax(center_region)
    print(f"vmin: {vmin}, vmax: {vmax}")
    image = np.where(image > vmax, vmax, image)
    if vmax > image_cube_vmax:
        image_cube_vmax = vmax
    image_dat.append(image)

print("image cube center area's image_cube_vmax: ", image_cube_vmax)

image_cube = np.stack(image_dat, axis=2)
print(image_cube.shape)
image_cube = np.transpose(image_cube, (2, 0, 1))
print(image_cube.shape)
max = np.max(image_cube)
min = np.min(image_cube)
print("cube_min:", min, " cube_max:", max)

image_cube_vmin, kaka = CatPSimgMinMax(image_cube)
print("image_cube_vmin: ", image_cube_vmin, "kaka: ", kaka)

image_cube = (image_cube - min) / (image_cube_vmax - min)


strech = SinhStretch()


image_cube1 = strech.__call__(image_cube)
print("--------------------------------------------------------")


for i in range(5):
    fig = plt.figure()
    im = plt.imshow(image_cube[i], origin='lower', cmap='Greys')
    fig.colorbar(im)
    plt.show()
