import os

import numpy as np
from astropy.io import fits
from scipy import ndimage

kernel = np.ones((3, 3))
root = "data/images14"

sources = os.listdir(root)
sources = [root + os.path.sep + a_src for a_src in sources]
sources2 = []
for i in range(len(sources)):
    sources2.append(sources[i])


def conv_mapping(x):
    """
        When the fifth value (x[4]) of the filter array (the center of the window)
        is null, replace it with the mean of the surrounding values.
        """
    if np.isnan(x[4]) and not np.isnan(np.delete(x, 4)).all():
        return np.nanmean(np.delete(x, 4))
    else:
        return x[4]


def scan_and_fix_damaged_image(img_file, file_dir):
    dat = fits.getdata(img_file)
    result = ndimage.generic_filter(dat, function=conv_mapping, footprint=kernel, mode='constant', cval=np.NaN)

    while np.isnan(result).any():
        result = ndimage.generic_filter(result, function=conv_mapping, footprint=kernel, mode='constant',
                                        cval=np.NaN)
        row, col = np.where(np.isnan(result))
        # print(f"Bad pixel num: {len(row)}")
        if len(row) >= 240 * 240 * 0.05:  # if missing values pixels number takes up 5% of the image, discard this image
            # delete this source's related images
            sources2.remove(file_dir)
            print(file_dir + "is removed")


for i in range(len(sources)):
    a_source = sources[i]
    tmp = []
    print(f"正在scan第{i}个source")
    for fits_file in os.listdir(a_source):
        scan_and_fix_damaged_image(os.path.join(a_source, fits_file), a_source)


print(f"sources2's length is: {len(sources2)}")