import os

import aplpy
from astropy.io import fits

root = "test_images/113.742p31.567408"

# arr = os.listdir(root)
# z = os.path.join(root, arr[0])
# i = os.path.join(root, arr[1])
# y = os.path.join(root, arr[2])
# r = os.path.join(root, arr[3])
# g = os.path.join(root, arr[4])
#
# aplpy.make_rgb_cube([g, r, i, z, y], 'test_images/5_dim_cube.fits')


dat = fits.getdata('test_images/5_dim_cube.fits')
print(dat)

