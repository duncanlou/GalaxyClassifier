import numpy as np
from astropy.visualization import LinearStretch, SinhStretch, ImageNormalize, ZScaleInterval
from ccdproc import CCDData
from regions import CirclePixelRegion, make_example_dataset, PixCoord
from astropy.io import fits
import os
import matplotlib.pyplot as plt

from utils import CatPSimgMinMax

files = [os.path.join(f'{os.getcwd()}/113.742p31.567408', f) for f in os.listdir('113.742p31.567408')]
files = sorted(files)
image_dat = []
for i in range(5):
    dat = fits.getdata(files[i])
    image_dat.append(dat)

image_cube = np.stack(image_dat, axis=2)
print(image_cube.shape)
image_cube = np.transpose(image_cube, (2, 0, 1))
print(image_cube.shape)
max = np.max(image_cube)
min = np.min(image_cube)
print(min, max)
minf, maxf = CatPSimgMinMax(image_cube)
print(minf, maxf)

image_cube1 = (image_cube - minf) / (maxf - minf)
# print(image_cube)
# print(image_cube1)
stretch = LinearStretch(slope=0.5, intercept=0.5) + SinhStretch() + \
              LinearStretch(slope=2, intercept=-1)
value = stretch.__call__(image_cube1)
print(value)
# print(image_cube1)

for i in range(5):
    plt.imshow(value[i], origin='lower', cmap='gray')
    plt.colorbar()
    plt.show()

#
