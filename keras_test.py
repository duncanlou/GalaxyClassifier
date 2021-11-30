import numpy as np
import aplpy
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from astropy.io import fits
from astropy.visualization import make_lupton_rgb

g = "test_images/stack_g_ra172.831370_dec2.081550_arcsec60_skycell1365.052.fits"
r = "test_images/stack_r_ra172.831370_dec2.081550_arcsec60_skycell1365.052.fits"
i = "test_images/stack_i_ra172.831370_dec2.081550_arcsec60_skycell1365.052.fits"
z = "test_images/stack_z_ra172.831370_dec2.081550_arcsec60_skycell1365.052.fits"
y = "test_images/stack_y_ra172.831370_dec2.081550_arcsec60_skycell1365.052.fits"
lala = aplpy.make_rgb_cube([g, r, i, z, y], "test_images/5_channel_img.fits")
