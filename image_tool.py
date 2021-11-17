import os

import numpy as np
from astropy.io import fits
from astropy.convolution import interpolate_replace_nans, Gaussian2DKernel

root = os.getcwd() + os.path.sep + "images14"

kernel = Gaussian2DKernel(x_stddev=1)


raw_src_imgs = [os.path.join(root, a_src) for a_src in os.listdir(root)]

fixed_src_imgs = []
for i in range(len(raw_src_imgs)):
    fixed_src_imgs.append(raw_src_imgs[i])





def scan_and_fix_damaged_image(raw_images):
    for i in range(5):
        row, col = np.where(np.isnan(raw_images[i]))
        print(f"Bad pixel num: {len(row)}")
        if len(row) >= 240 * 240 * 0.05:  # if missing values pixels number takes up 5% of the image, discard this image
            # delete this source's related images
            fixed_src_imgs.remove(file_dir)
            print(file_dir + "is removed")







for i in range(len(raw_src_imgs)):
    a_source = raw_src_imgs[i]
    tmp = []
    print(f"正在scan第{i}个source")
    for fits_file in os.listdir(a_source):
        scan_and_fix_damaged_image(os.path.join(a_source, fits_file), a_source)


print(f"sources2's length is: {len(fixed_src_imgs)}")