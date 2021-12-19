import logging
import os
import shutil

import astropy.table
import numpy as np
import torch
import torchvision.utils
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from astropy.stats import gaussian_fwhm_to_sigma
from photutils import detect_threshold, detect_sources

kernel = Gaussian2DKernel(x_stddev=1)

import matplotlib.pyplot as plt


def check_if_is_five(source_dir="data/sources/QSO"):
    root_path = os.path.join(os.getcwd(), source_dir)
    sources = [os.path.join(root_path, source) for source in os.listdir(root_path)]
    for s in sources:
        if not os.path.isdir(s):
            continue
        img_arr = os.listdir(s)
        if len(img_arr) != 5:
            print(f"{s} don't have 5 fits files")
            raise ValueError

    print("check finished")


# check_if_is_five(source_dir='data/sources/GALAXY')



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


def remove_nan(image_dat):  # shape: [5, 240, 240]
    row, col = np.where(np.isnan(image_dat))
    kernel = Gaussian2DKernel(x_stddev=2)
    if len(row) > 0:
        # replace bad data with values interpolated from their neighbors
        image_dat = interpolate_replace_nans(image_dat, kernel)
    return image_dat


def classify_downloaded_sources(dir):
    T = astropy.table.Table.read("/home/duncan/PycharmProjects/MyResearchProject_Duncan/data/DuncanSDSSdata.tbl",
                                 format='ipac')
    ralist = list(T['ra'])
    declist = list(T['dec'])
    classes = list(T['class'])

    positions = []
    for i in range(len(ralist)):
        positions.append((ralist[i], declist[i]))

    source_dirs = os.listdir(dir)

    contained_count = 0
    uncontained_count = 0
    for src_dir in source_dirs:
        full_src_path = os.path.join("/mnt/DataDisk/Duncan/images4", src_dir)
        if os.path.isdir(full_src_path):
            if src_dir.__contains__('p'):
                ra, dec = src_dir.split('p')
                ra, dec = float(ra), float(dec)
            else:
                ra, dec = src_dir.split('m')
                dec = "-" + dec
                ra, dec = float(ra), float(dec)
            pos = (ra, dec)
            if positions.__contains__(pos):
                idx = positions.index(pos)
                label = classes[idx]
                print(label)
                contained_count += 1
                if label == 'GALAXY':
                    dest_folder = "/home/duncan/PycharmProjects/MyResearchProject_Duncan/data/sources/GALAXY"
                if label == 'QSO':
                    dest_folder = "/home/duncan/PycharmProjects/MyResearchProject_Duncan/data/sources/QSO"
                if label == 'STAR':
                    dest_folder = "/home/duncan/PycharmProjects/MyResearchProject_Duncan/data/sources/STAR"

                shutil.move(src=full_src_path, dst=dest_folder)

            else:
                print(f"can't find {pos} in table")
                uncontained_count += 1

    print(contained_count)
    print(uncontained_count)


def get_fits_dict(imgs, source_dir):
    if len(imgs) != 5:
        logging.error(f"{source_dir_name} must contain 5 fits files, current is {len(img_list)}")
        raise IOError

    fits_dict = {}

    for f in imgs:
        if f.startswith("stack_g"):
            fits_dict[0] = os.path.join(source_dir, f)
        elif f.startswith("stack_r"):
            fits_dict[1] = os.path.join(source_dir, f)
        elif f.startswith("stack_i"):
            fits_dict[2] = os.path.join(source_dir, f)
        elif f.startswith("stack_z"):
            fits_dict[3] = os.path.join(source_dir, f)
        elif f.startswith("stack_y"):
            fits_dict[4] = os.path.join(source_dir, f)
        else:
            raise ValueError

    return fits_dict


def CatPSimgMinMax(img_dat):
    medflux = np.median(img_dat)
    madflux = np.median(np.abs(img_dat - medflux))

    lomad, himad = (-2.0, 10.0)  # number of mads to the min and max

    minflux = medflux + lomad * madflux
    maxflux = medflux + himad * madflux

    # midflux = medflux  + 0.5 * (himad + lomad) * madflux
    # radflux = 0.5 * (himad - lomad) * madflux

    return minflux, maxflux


from astropy.visualization import ImageNormalize, MinMaxInterval, SqrtStretch


def showImages(img):
    g = img[0]  # blue
    i = img[1]  # red
    r = img[2]  # green

    color_fits = torch.stack((i, r, g), dim=0)
    vmin, vmax = CatPSimgMinMax(color_fits)
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SqrtStretch())

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(color_fits.numpy().transpose((1, 2, 0)), origin='lower', norm=norm)
    fig.colorbar(im)
    plt.show()


def have_source_in_center(image_dat):
    sigma = 3.0 * gaussian_fwhm_to_sigma  # FWHM = 3
    kenl = Gaussian2DKernel(sigma, x_size=3, y_size=3)
    kenl.normalize()

    threshold = detect_threshold(image_dat, nsigma=2.)
    segm = detect_sources(image_dat, threshold, npixels=5, kernel=kenl)
    if segm is None:
        return False
    else:
        return True
