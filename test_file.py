import time
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

image = fits.getdata("stack_g_ra145.160390_dec-0.655469_arcsec60_skycell1268.082.fits")
plt.imshow(image)
plt.show()