import logging
import os
import numpy as np
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans

kernel = Gaussian2DKernel(x_stddev=1)


def rmnan(filename):
    image = fits.getdata(filename)
    header = fits.getheader(filename)
    for i in range(5):
        single_channel = image[i, :, :]
        x, y = np.where(np.isnan(single_channel))
        logging.warning(f"nan pixels: {x}")
        if len(x) > 50:  # drop this sources
            os.remove(filename)
            logging.warning(f"{filename} has been removed")
            return None
        elif len(x) == 0:
            continue
        else:  # filter this image
            fixed_img = interpolate_replace_nans(single_channel, kernel)
            image[i, :, :] = fixed_img
    fits.update(filename, image, header=header)
    return image


file = "data/sources/GALAXY/172.767730_+2.153035_cube.fits"
rmnan(file)
