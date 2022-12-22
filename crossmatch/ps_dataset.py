import math
import os
import warnings

import numpy as np
import pandas as pd
from astropy.utils.exceptions import AstropyWarning
from numpy.linalg import norm

warnings.simplefilter('ignore', category=AstropyWarning)
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D

from sympy import Point, Line

from skimage.exposure import rescale_intensity
from torch.utils.data import Dataset
from utils import remove_nan, optimize_image, normalize, get_sigma_clip, cal_luptitude

df_grouped_components = pd.read_csv("catalogs/grouped_VLASS_components.csv")


def get_point2line_distance(x_opt, y_opt, pa):
    x_prime = x_opt - 432
    y_prime = y_opt - 432

    line_angle = pa - 90 if pa >= 90 else pa + 90

    if line_angle != 90:
        k = np.tan(line_angle)  # y = kx
        d_p2l = np.abs(k * x_prime - y_prime) / np.sqrt(1 + k * k)
    else:
        d_p2l = np.abs(x_prime)

    return d_p2l


def get_point2line_distance2(x1, y1, x2, y2, x3, y3):
    return norm(np.cross((x2 - x1, y2 - y1), (x1 - x3, y1 - y3))) / norm((x2 - x1, y2 - y1))


def get_line_intersection(x1, y1, k1, x2, y2, k2):
    b1 = y1 - k1 * x1
    b2 = y2 - k2 * x2
    x_intersection = (b2 - b1) / (k1 - k2)
    y_intersection = k1 * x_intersection + b1
    return x_intersection, y_intersection


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
        # self.ps_n_samples = self.ps_n_samples.groupby("VLASS_component_name").first().reset_index()

        self.df_samples = pd.concat([self.ps_p_samples, self.ps_n_samples], ignore_index=True)
        self.df_samples = self.df_samples.sample(frac=1)

        self.radio_transform = radio_transform
        self.opt_transform = opt_transform

    def __len__(self):
        return len(self.df_samples)

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
        from astropy.wcs import WCS
        radio_img_path = os.path.join(self.radio_root_dir, f"{radio_component}.fits")
        raw_image = fits.getdata(radio_img_path).squeeze()
        header = fits.getheader(radio_img_path)
        wcs = WCS(header).celestial
        image = remove_nan(raw_image)
        image[np.isnan(image)] = 0.0
        image = get_sigma_clip(image)
        image = rescale_intensity(image, out_range=(0, 1))
        return image, wcs

    def get_2_brightest_components_from_one_group(self, the_group):
        if len(the_group) > 2:  # There are at least 3 components in the scope
            # Choose the 2 brightest components
            flux_2_brightest = the_group["Total_flux_1"].nlargest(2)  # a pandas.series object
            indice_of_the_2_brightest_components = list(flux_2_brightest.index.values)
            df_2_brightest_components = the_group.loc[indice_of_the_2_brightest_components]
            return df_2_brightest_components
        else:  # There only 2 components in this scope
            return the_group

    def __getitem__(self, idx):
        opt_sample: pd.Series = self.df_samples.iloc[idx]
        radio_component_name = opt_sample["VLASS_component_name"]

        radio_img_dat, radio_wcs = self.preprocess_radio_image(radio_component_name)

        cutouts_for_one_PS_source = self.create_PS_cutout(opt_sample)
        ps_img_list = [cutout.data for cutout in cutouts_for_one_PS_source]
        w1flux = opt_sample['w1flux']
        w2flux = opt_sample['w2flux']
        m1, m2 = cal_luptitude(w1flux, w2flux)
        wise_magnitude_info = np.asarray([m1, m2])

        ps_imgcube = self.create_imgcube(ps_img_list)

        cutout_position_info = cutouts_for_one_PS_source[0].position_original
        x = cutout_position_info[0]
        y = cutout_position_info[1]

        radio_core_position_info = []

        df_target_component_row = df_grouped_components[
            df_grouped_components["Component_name_1"] == radio_component_name]

        if len(df_target_component_row) == 0:  # Single component
            origin = (radio_img_dat.shape[0] / 2, radio_img_dat.shape[1] / 2)
            r = math.dist((x, y), origin)
            if opt_sample['DC_Maj'] == 0 or opt_sample['DC_Min'] == 0:  # compact
                distance_x = 0
                distance_y = r
            else:  # extended
                # set radio component position as origin, then the optical candidate's position is:
                pa = opt_sample['DC_PA']
                line_angle = pa - 90 if pa >= 90 else pa + 90
                if line_angle != 90:
                    slope = math.tan(line_angle)
                    flux_center_line = Line(Point(*origin), slope=slope)
                    distance_x = float(flux_center_line.distance(Point(x, y)))
                    distance_y = math.sqrt(math.dist(origin, (x, y)) ** 2 - distance_x ** 2)
                else:
                    distance_x = abs(x - origin[0])
                    distance_y = abs(y - origin[1])

            radio_core_position_info.append(r)
            radio_core_position_info.append(distance_x)
            radio_core_position_info.append(distance_y)

        else:
            # multiple component
            group_id = df_target_component_row.loc[:, "GroupID"].iloc[0]
            the_group = df_grouped_components[df_grouped_components["GroupID"] == group_id]
            df_2_components = self.get_2_brightest_components_from_one_group(the_group)
            ras = list(df_2_components['RA_1'])
            decs = list(df_2_components['DEC_1'])
            component1_coord = SkyCoord(ra=ras[0], dec=decs[0], unit='deg')
            component2_coord = SkyCoord(ra=ras[1], dec=decs[1], unit='deg')
            x1_radio, y1_radio = radio_wcs.world_to_pixel(component1_coord)
            x2_radio, y2_radio = radio_wcs.world_to_pixel(component2_coord)
            x1_radio = x1_radio.item()
            x2_radio = x2_radio.item()
            y1_radio = y1_radio.item()
            y2_radio = y2_radio.item()
            pa1 = df_2_components.loc[:, 'DC_PA_1'].iloc[0]
            pa2 = df_2_components.loc[:, 'DC_PA_1'].iloc[1]
            line1_angle = pa1 - 90 if pa1 >= 90 else pa1 + 90
            line2_angle = pa2 - 90 if pa2 >= 90 else pa2 + 90

            start_point_PA_line1 = Point(x1_radio, y1_radio)
            end_point_PA_line1 = Point(x1_radio + math.cos(line1_angle), y1_radio + math.sin(line1_angle))
            PA_line1 = Line(start_point_PA_line1, end_point_PA_line1)

            start_point_PA_line2 = Point(x2_radio, y2_radio)
            end_point_PA_line2 = Point(x2_radio + math.cos(line2_angle), y2_radio + math.sin(line2_angle))
            PA_line2 = Line(start_point_PA_line2, end_point_PA_line2)

            assert ~PA_line1.is_parallel(PA_line2)
            x_intersect, y_intersect = (PA_line1.intersection(PA_line2))[0].coordinates

            (x_bar, y_bar) = ((x1_radio + x2_radio) / 2, (y1_radio + y2_radio) / 2)
            crossingPoint2FluxCenter_distance = math.dist((x_intersect, y_intersect), (x_bar, y_bar))

            p1 = Point(x1_radio, y1_radio)
            p2 = Point(x2_radio, y2_radio)
            flux_center_line = Line(p1, p2)
            crossingPoint2fluxCenterLine_perpendicular_distance = float(
                flux_center_line.distance(Point(x_intersect, y_intersect)))
            parallel_distance = math.sqrt(
                crossingPoint2FluxCenter_distance ** 2 - crossingPoint2fluxCenterLine_perpendicular_distance ** 2)
            distance_between_two_component_flux_centers = math.dist((x1_radio, y1_radio), (x2_radio, y2_radio))
            if parallel_distance > 0.5 * distance_between_two_component_flux_centers or crossingPoint2fluxCenterLine_perpendicular_distance > 60:
                # The crossing point is probably irrelevant
                # The line is the one travels through the 2 brightest radio component centers
                # Origin is the flux center
                origin = (x_bar, y_bar)
                r = math.dist((x, y), origin)
                distance_x = float(flux_center_line.distance(Point(x, y)))
                distance_y = math.sqrt(r ** 2 - distance_x ** 2)

            else:
                # The Origin is the average of the PA line crossing point and the group flux center
                # The line passes through the origin and is parallel to the flux center line
                origin = ((x_bar + x_intersect) / 2, (y_bar + y_intersect) / 2)
                r = math.dist(origin, (x, y))

                flux_center_parallelLine = flux_center_line.parallel_line(Point(*origin))
                distance_x = float(flux_center_parallelLine.distance(Point(x, y)))
                distance_y = math.sqrt(r ** 2 - distance_x ** 2)

            radio_core_position_info.append(r)
            radio_core_position_info.append(distance_x)
            radio_core_position_info.append(distance_y)

        cutout_position_info = np.asarray(radio_core_position_info)

        label = opt_sample["PS_class"]
        if label == 'P':
            label = 1
        else:
            label = 0

        ps_id = opt_sample['Pan-STARRS_objID']
        ps_ra = opt_sample['Pan-STARRS_RAJ2000']
        ps_dec = opt_sample['Pan-STARRS_DEJ2000']

        cname = radio_component_name[len('VLASS1QLCIR J'):]
        if cname.__contains__('-'):
            rra, rdec = cname.split('-')
            VLASS_name = (0, np.float(rra), np.float(rdec))
        else:
            rra, rdec = cname.split('+')
            VLASS_name = (1, np.float(rra), np.float(rdec))

        if self.radio_transform:
            radio_img_dat = self.radio_transform(radio_img_dat)
        if self.opt_transform:
            ps_imgcube = self.opt_transform(ps_imgcube)
        return radio_img_dat, ps_imgcube, wise_magnitude_info, cutout_position_info, label, (
            ps_id, ps_ra, ps_dec), VLASS_name
