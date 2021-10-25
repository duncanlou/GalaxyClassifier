import os
from typing import Tuple, Dict, List, Optional, Callable, cast

import numpy as np

from astropy.io import fits

from scipy import ndimage

from torchvision.datasets import DatasetFolder, folder


class FitsFolder(DatasetFolder):
    EXTENSIONS = ('.fits',)

    def __init__(
            self,
            root: str,
            transform=None,
            target_transform=None,
            loader=None,
    ):
        if loader is None:
            loader = self.__fits_loader
        super(FitsFolder, self).__init__(root, loader, self.EXTENSIONS,
                                         transform=transform,
                                         target_transform=target_transform)

    def make_dataset(self,
                     directory: str,
                     class_to_idx: Dict[str, int],
                     extensions: Optional[Tuple[str, ...]] = None,
                     is_valid_file: Optional[Callable[[str], bool]] = None,
                     ) -> List[Tuple[Tuple[str, str, str, str, str], int]]:
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError(
                "The class_to_idx parameter cannot be None."
            )
        directory = os.path.expanduser(directory)

        if class_to_idx is None:
            _, class_to_idx = super().find_classes(directory)
        elif not class_to_idx:
            raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

        if extensions is not None:
            def is_valid_file(x: str) -> bool:
                return folder.has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        instances = []
        available_classes = set()
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue

            spectra = []
            for entry in os.scandir(target_dir):
                if os.path.isdir(entry.path):
                    spectra.append(os.path.join(target_dir, entry.name))

            spectra = sorted(spectra)
            g_fits = sorted([os.path.join(spectra[0], f) for f in os.listdir(spectra[0])])
            i_fits = sorted([os.path.join(spectra[1], f) for f in os.listdir(spectra[1])])
            r_fits = sorted([os.path.join(spectra[2], f) for f in os.listdir(spectra[2])])
            y_fits = sorted([os.path.join(spectra[3], f) for f in os.listdir(spectra[3])])
            z_fits = sorted([os.path.join(spectra[4], f) for f in os.listdir(spectra[4])])

            length = len(g_fits)
            if all(len(lst) == length for lst in [g_fits, i_fits, r_fits, y_fits, z_fits]):
                for i in range(length):
                    all_channel_img_path = g_fits[i], i_fits[i], r_fits[i], y_fits[i], z_fits[i]
                    item = all_channel_img_path, class_index
                    instances.append(item)

                    if target_class not in available_classes:
                        available_classes.add(target_class)

            else:
                msg = f"Channels have different lengths"
                raise FileNotFoundError(msg)

        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            if extensions is not None:
                msg += f"Supported extensions are: {', '.join(extensions)}"
            raise FileNotFoundError(msg)

        return instances

    @staticmethod
    def __fits_loader(all_channal_source_files):
        arr = all_channal_source_files
        g_dat = fits.getdata(arr[0])
        i_dat = fits.getdata(arr[1])
        r_dat = fits.getdata(arr[2])
        y_dat = fits.getdata(arr[3])
        z_dat = fits.getdata(arr[4])

        def conv_mapping(x):
            """
            When the fifth value (x[4]) of the filter array (the center of the window) is null, replace it with the mean
            of the surrounding values
            :param x:
            :return:
            """
            if np.isnan(x[4]) and not np.isnan(np.delete(x, 4)).all():
                return np.nanmean(np.delete(x, 4))
            else:
                return x[4]

        mask = np.ones((3, 3))
        g_dat = ndimage.generic_filter(g_dat, function=conv_mapping, footprint=mask, mode='constant', cval=np.NaN)
        i_dat = ndimage.generic_filter(i_dat, function=conv_mapping, footprint=mask, mode='constant', cval=np.NaN)
        r_dat = ndimage.generic_filter(r_dat, function=conv_mapping, footprint=mask, mode='constant', cval=np.NaN)
        y_dat = ndimage.generic_filter(y_dat, function=conv_mapping, footprint=mask, mode='constant', cval=np.NaN)
        z_dat = ndimage.generic_filter(z_dat, function=conv_mapping, footprint=mask, mode='constant', cval=np.NaN)

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
