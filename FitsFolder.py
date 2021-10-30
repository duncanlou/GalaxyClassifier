import os
from typing import Tuple, Dict, List, Optional, Callable, cast

import numpy as np
import torch

from astropy.io import fits
from astropy.table import Table

from scipy import ndimage

from torchvision.datasets import DatasetFolder, folder


class FitsFolder(DatasetFolder):
    EXTENSIONS = ('.fits',)

    def __init__(
            self,
            root: str,
            data_table,
            transform=None,
            target_transform=None,
            loader=None,
    ):
        self.T = data_table
        if loader is None:
            loader = self.__fits_loader
        super(FitsFolder, self).__init__(root, loader, self.EXTENSIONS,
                                         transform=transform,
                                         target_transform=target_transform)

    def find_classes(self, directory: str) -> Tuple[List[List[str]], Dict[str, int]]:
        print("find_classes() will be invoked how many times?")
        src_entries = self.T[130000:140000]
        self.types = src_entries['class']
        img_folders = os.listdir('data/images14')

        img_files = []
        for i in range(len(img_folders)):
            img_folder = img_folders[i]
            tmp = []
            for f_name in os.listdir(os.path.join('data/images14', img_folder)):
                img_path = os.path.join(os.path.join('data/images14', img_folder), f_name)
                tmp.append(img_path)
            img_files.append(tmp)

        class_type_mappings = {
            'GALAXY': 0,
            'QSO': 1,
            'STAR': 2
        }

        return img_files, class_type_mappings

    def make_dataset(self,
                     directory: str,
                     class_to_idx: Dict[str, int],
                     extensions: Optional[Tuple[str, ...]] = None,
                     is_valid_file: Optional[Callable[[str], bool]] = None,
                     ) -> list[tuple[list[str], int]]:

        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError(
                "The class_to_idx parameter cannot be None."
            )
        directory = os.path.expanduser(directory)

        src_files, _ = self.find_classes(directory)

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

        if extensions is not None:
            def is_valid_file(x: str) -> bool:
                return folder.has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        instances = []

        for idx, img_f in enumerate(src_files):
            item = img_f, class_to_idx[self.types[idx]]
            instances.append(item)

        return instances

    @staticmethod
    def __fits_loader(images):
        if len(images) != 5:
            print(images)
            raise IndexError
        g_dat = fits.getdata(images[0])
        i_dat = fits.getdata(images[1])
        r_dat = fits.getdata(images[2])
        y_dat = fits.getdata(images[3])
        z_dat = fits.getdata(images[4])

        # def conv_mapping(x):
        #     """
        #     When the fifth value (x[4]) of the filter array (the center of the window) is null, replace it with the mean
        #     of the surrounding values
        #     :param x:
        #     :return:
        #     """
        #     if np.isnan(x[4]):  # x中至少有一个不为NULL的值
        #         if not np.isnan(np.delete(x, 4)).all():
        #             return np.nanmean(np.delete(x, 4))
        #         else:
        #             print(x)
        #             raise IOError
        #     else:  # x中的所有值均为NULL
        #         return x[4]
        #
        # mask = np.ones((3, 3))
        #
        # g_dat = ndimage.generic_filter(g_dat, function=conv_mapping, footprint=mask, mode='constant', cval=np.NaN)
        # i_dat = ndimage.generic_filter(i_dat, function=conv_mapping, footprint=mask, mode='constant', cval=np.NaN)
        # r_dat = ndimage.generic_filter(r_dat, function=conv_mapping, footprint=mask, mode='constant', cval=np.NaN)
        # y_dat = ndimage.generic_filter(y_dat, function=conv_mapping, footprint=mask, mode='constant', cval=np.NaN)
        # z_dat = ndimage.generic_filter(z_dat, function=conv_mapping, footprint=mask, mode='constant', cval=np.NaN)

        def fits_normalization(img_dat):
            if img_dat.shape != (240, 240):
                raise TypeError
            vmax = np.max(img_dat)
            vmin = np.min(img_dat)

            img_dat[:][:] = (img_dat[:][:] - vmin) / (vmax - vmin)
            return img_dat

        def fits_std(img_dat):
            if img_dat.shape != (240, 240):
                raise TypeError
            mean = np.mean(img_dat)
            std = np.std(img_dat)
            img_dat[:][:] = (img_dat[:][:] - mean) / std
            return img_dat

        def dat_transform(raw_dat):
            std_dat = fits_std(raw_dat)
            norm_dat = fits_normalization(std_dat)
            return norm_dat

        g_dat = dat_transform(g_dat)
        i_dat = dat_transform(i_dat)
        r_dat = dat_transform(r_dat)
        y_dat = dat_transform(y_dat)
        z_dat = dat_transform(z_dat)

        dat = np.stack((g_dat, i_dat, r_dat, y_dat, z_dat), axis=2)

        return dat
