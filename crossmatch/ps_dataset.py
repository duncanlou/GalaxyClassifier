import os
import warnings

import numpy as np
import pandas as pd
from astropy.utils.exceptions import AstropyWarning

warnings.simplefilter('ignore', category=AstropyWarning)
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D

from skimage.exposure import rescale_intensity
from torch.utils.data import Dataset
from utils import remove_nan, optimize_image, normalize, get_sigma_clip, cal_luptitude


class FitsImageSet(Dataset):
    def __init__(self, radio_root_dir, opt_root_dir, ps_positive_samples_csv, ps_negative_samples_csv,
                 radio_transform=None, opt_transform=None):
        self.opt_root_dir = opt_root_dir
        self.radio_root_dir = radio_root_dir
        self.ps_background_cutouts = os.listdir(opt_root_dir)
        self.radio_sources = os.listdir(radio_root_dir)
        assert len(self.ps_background_cutouts) == len(self.radio_sources)

        self.ps_p_samples = pd.read_csv(ps_positive_samples_csv)

        self.ps_n_samples = pd.read_csv(ps_negative_samples_csv)


        self.df_samples = pd.concat([self.ps_p_samples, self.ps_n_samples], ignore_index=True)

        self.df_samples = self.df_samples.sample(frac=1)
        self.sample_groups = self.df_samples.groupby("VLASS_component_name")
        self.radio_transform = radio_transform
        self.opt_transform = opt_transform

    def __len__(self):
        return len(self.sample_groups)

    def create_PS_cutout(self, opt_sample: pd.Series):
        radio_component_name = opt_sample["VLASS_component_name"]
        ps_background_cutout_path = os.path.join(self.opt_root_dir, radio_component_name)
        fits_imgs = os.listdir(ps_background_cutout_path)
        fits_imgs.sort()
        fits_path = [os.path.join(ps_background_cutout_path, img) for img in fits_imgs]
        five_band_img_cutouts = []

        ra, dec = opt_sample['Pan-STARRS_RAJ2000'], opt_sample['Pan-STARRS_DEJ2000']
        for fits_file in fits_path:
            image = fits.getdata(fits_file).squeeze()
            header = fits.getheader(fits_file)
            image[np.isnan(image)] = 0.0
            pos = SkyCoord(ra, dec, frame='icrs', unit=u.deg)
            cutout = Cutout2D(image, position=pos, size=240, wcs=WCS(header).celestial)
            five_band_img_cutouts.append(cutout)

        return five_band_img_cutouts

    def create_imgcube(self, img_data_list):
        img_cube = np.stack(img_data_list, axis=2)
        img_cube = optimize_image(img_cube)
        img_cube = normalize(img_cube)
        return img_cube

    def preprocess_radio_image(self, radio_component):
        radio_img_path = os.path.join(self.radio_root_dir, f"{radio_component}.fits")
        raw_image = fits.getdata(radio_img_path).squeeze()
        image = remove_nan(raw_image)
        image[np.isnan(image)] = 0.0
        image = get_sigma_clip(image)
        image = rescale_intensity(image, out_range=(0, 1))
        return image

    def __getitem__(self, idx):
        opt_sample: pd.Series = self.df_samples.iloc[idx]
        radio_component_name = opt_sample["VLASS_component_name"]

        radio_img_dat = self.preprocess_radio_image(radio_component_name)

        cutouts_for_one_PS_source = self.create_PS_cutout(opt_sample)
        ps_img_list = [cutout.data for cutout in cutouts_for_one_PS_source]
        w1flux = opt_sample['w1flux']
        w2flux = opt_sample['w2flux']
        m1, m2 = cal_luptitude(w1flux, w2flux)
        wise_magnitude_info = np.asarray([m1, m2])

        ps_imgcube = self.create_imgcube(ps_img_list)

        cutout_position_info = cutouts_for_one_PS_source[0].position_original
        (cutout_centerX, cutout_centerY) = cutouts_for_one_PS_source[0].center_original
        distant_r = (432 - cutout_centerX) ** 2 + (432 - cutout_centerY) ** 2
        distant_r = distant_r ** 0.5
        cutout_position_info = cutout_position_info + (distant_r,)
        cutout_position_info = np.asarray(cutout_position_info)

        raw_label = opt_sample["PS_class"]
        label = 1 if raw_label == 'P' else 0

        ps_id = opt_sample['Pan-STARRS_objID']
        ps_ra = opt_sample['Pan-STARRS_RAJ2000']
        ps_dec = opt_sample['Pan-STARRS_DEJ2000']

        if self.radio_transform:
            radio_img_dat = self.radio_transform(radio_img_dat)
        if self.opt_transform:
            ps_imgcube = self.opt_transform(ps_imgcube)

        return radio_img_dat, ps_imgcube, wise_magnitude_info, cutout_position_info, label, (ps_id, ps_ra, ps_dec)
