import logging
import os
import shutil

import numpy as np
from astropy.io import fits

from utils import have_source_in_center, remove_nan

galaxy_source_root = os.path.join(os.getcwd(), "data/sources/GALAXY")
star_source_root = os.path.join(os.getcwd(), "data/sources/STAR")
quasar_source_root = os.path.join(os.getcwd(), "data/sources/QSO")

too_many_nan_galaxy = "/home/duncan/PycharmProjects/MyResearchProject_Duncan/data/too_many_invalid_values/GALAXY"
too_many_nan_star = "/home/duncan/PycharmProjects/MyResearchProject_Duncan/data/too_many_invalid_values/STAR"
too_many_nan_QSO = "/home/duncan/PycharmProjects/MyResearchProject_Duncan/data/too_many_invalid_values/QSO"

no_source_in_center_galaxy = "/home/duncan/PycharmProjects/MyResearchProject_Duncan/data/no_source_in_center/GALAXY"
no_source_in_center_star = "/home/duncan/PycharmProjects/MyResearchProject_Duncan/data/no_source_in_center/STAR"
no_source_in_center_QSO = "/home/duncan/PycharmProjects/MyResearchProject_Duncan/data/no_source_in_center/QSO"

for root, dirs, fnames in sorted(os.walk(star_source_root, followlinks=True)):  # here
    for dir in sorted(dirs):
        src_path = os.path.join(root, dir)
        fits_img = [os.path.join(src_path, f) for f in os.listdir(src_path)]
        for f in fits_img:
            img_dat = fits.getdata(f)
            x, y = np.where(np.isnan(img_dat))
            if len(x) > 100:  # drop this sources
                shutil.move(src_path, too_many_nan_star)  # here
                print(f"{f} contains {len(x)} nan pixels, discard this source from dataset")
                break
            else:
                img_dat = remove_nan(img_dat)
            if not have_source_in_center(img_dat):
                logging.warning(f"{f}: No source has been detected in the image center, discard this "
                                f"source from dataset")
                shutil.move(src_path, no_source_in_center_star)  # here
                break
