import numpy as np
from astropy.stats import sigma_clipped_stats
from astropy.visualization import LinearStretch, SinhStretch, ImageNormalize, ZScaleInterval, SqrtStretch, \
    SquaredStretch, PowerStretch, LogStretch, ContrastBiasStretch, MinMaxInterval
from ccdproc import CCDData
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from regions import CirclePixelRegion, make_example_dataset, PixCoord
from astropy.io import fits
import os
import matplotlib.pyplot as plt

from photutils.centroids import centroid_com, centroid_quadratic, centroid_1dg, centroid_2dg
from photutils.detection import find_peaks

from utils import CatPSimgMinMax

ROOT = '/home/duncan/PycharmProjects/MyResearchProject_Duncan/data/sources/GALAXY/172.18648m1.4108745'

files = [os.path.join(ROOT, f) for f in os.listdir(ROOT)]
files = sorted(files)
image_dat = []

center_max = 0
for i in range(5):
    dat = fits.getdata(files[i])
    mean, median, std = sigma_clipped_stats(dat, sigma=3.0)
    print(median)
    threashold = median + (5. * std)
    tbl = find_peaks(dat, threashold, box_size=11)
    tbl['peak_value'].info.format = '%.8g'
    # print(tbl[:10])
    # dat -= median
    t = dat[117:123, 117:123]
    tmp = np.max(t)
    if tmp > center_max:
        center_max = tmp
    image_dat.append(dat)

print(f"center_max: {center_max}")

image_cube = np.stack(image_dat, axis=2)
print(image_cube.shape)
image_cube = np.transpose(image_cube, (2, 0, 1))
print(image_cube.shape)
max = np.max(image_cube)
min = np.min(image_cube)
print(min, max)

new_arr = []
for i in range(5):
    one_channel = image_cube[i]
    center = one_channel[117:123, 117:123]
    center_max = np.max(center)
    new_one = np.where(one_channel > center_max, center_max, one_channel)
    minf, maxf = CatPSimgMinMax(new_one)
    print("maxf: ", maxf)
    print("centermax:", center_max)
    print("minf:", minf)
    print("min", np.min(new_one))
    guiyi = (new_one - minf) / (center_max - minf)
    new_arr.append(guiyi)


# print(image_cube1)
# strech = SinhStretch()
# image_cube1 = strech.__call__(image_cube1)
# print("--------------------------------------------------------")
# print(image_cube1)

for i in range(5):
    fig = plt.figure()
    im = plt.imshow(new_arr[i], origin='lower')
    fig.colorbar(im)
    plt.show()
