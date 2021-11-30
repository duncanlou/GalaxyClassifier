import os
from typing import Tuple, Dict, List, Optional, Callable, cast, Any

import numpy as np
import torch

from astropy.io import fits
from astropy.convolution import interpolate_replace_nans, Gaussian2DKernel

from torchvision.datasets import DatasetFolder, folder

kernel = Gaussian2DKernel(x_stddev=1)


def filter_dataset(fits_path):
    path_levels_arr = fits_path.split(os.path.sep)
    try:
        single_channel_fits_dat = fits.getdata(fits_path)
        row, col = np.where(np.isnan(single_channel_fits_dat))
        fits_n = path_levels_arr[-3] + os.path.sep + path_levels_arr[-2] + os.path.sep + path_levels_arr[-1]
        print(f"{len(row)} NaN values are found in: {fits_n}")
        if len(row) >= 50:  # if missing values pixels number is larger than 100, skip this image
            print(f"{len(row)} NaN values are found in: {fits_n}")
            return None
        else:
            return interpolate_replace_nans(single_channel_fits_dat, kernel)
    except OSError as err:
        fits_n = path_levels_arr[-3] + os.path.sep + path_levels_arr[-2] + os.path.sep + path_levels_arr[-1]
        print(f"invalid fits file: {fits_n}, error: {err}")
        return None


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
                     ) -> List[Tuple[List[str], int]]:
        """Generates a list of samples of a form (path_to_sample, class).

            See :class:`DatasetFolder` for details.

            Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
            by default.
        """
        directory = os.path.join(os.getcwd(), directory)

        # Ensure that class_to_idx is neither none nor empty.
        if class_to_idx is None:
            _, class_to_idx = self.find_classes(directory)
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
            if not os.path.isdir(target_dir):  # if target_dir is a file, then skip it
                continue
            for root, dirnames, fnames in sorted(os.walk(target_dir,
                                                         followlinks=True)):  # I don't quite understand 'os.walk' this API, need to study it
                single_source_img_dat = []
                for fname in sorted(fnames):  # for...else...    (tomorrow to solve it)
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        # one_channal_img_dat = filter_dataset(path)
                        one_channal_img_dat = fits.getdata(path)
                        if one_channal_img_dat is None:  # drop all the 5 fits file
                            break
                        single_source_img_dat.append(one_channal_img_dat)
                        if target_class not in available_classes:
                            available_classes.add(target_class)
                if len(single_source_img_dat) == 5:
                    dat = np.stack(single_source_img_dat, axis=2)
                    item = dat, class_index
                    instances.append(item)

        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            if extensions is not None:
                msg += f"Supported extensions are: {', '.join(extensions)}"
            raise FileNotFoundError(msg)

        return instances

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        target = torch.tensor(target)
        return sample, target

    @staticmethod
    def __fits_loader(img_dat):

        vmax = np.max(img_dat)
        vmin = np.min(img_dat)
        img_dat[:, :, :] = (img_dat - vmin) / (vmax - vmin)
        return img_dat
