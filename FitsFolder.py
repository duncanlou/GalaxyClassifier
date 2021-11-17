import os
from typing import Tuple, Dict, List, Optional, Callable, cast

import numpy as np
import copy
from astropy.io import fits
from astropy.convolution import interpolate_replace_nans, Gaussian2DKernel

from torchvision.datasets import DatasetFolder, folder

IMG_ROOT = "/Users/loukangzhi/PycharmProjects/GalaxyClassifier/images14"


def filter_dataset(sources):
    sources_copy = copy.deepcopy(sources)

    for source in sources:
        for j in range(5):
            try:
                fits_dat = fits.getdata(source[j])
                row, col = np.where(np.isnan(fits_dat))
                if len(row) >= 100:  # if missing values pixels number is larger than 100, skip this image
                    path_n_arr = source[j].split(os.path.sep)
                    fits_n = path_n_arr[-3] + os.path.sep + path_n_arr[-2] + os.path.sep + path_n_arr[-1]
                    print(f"{len(row)} NaN values are found in: {fits_n}")
                    sources_copy.remove(source)
                    break
            except OSError as err:
                path_n_arr = source[j].split(os.path.sep)
                fits_n = path_n_arr[-3] + os.path.sep + path_n_arr[-2] + os.path.sep + path_n_arr[-1]
                print(f"invalid fits file: {fits_n}, error: {err}")
        else:
            pass

    diff = len(sources) - len(sources_copy)
    print(f"Altogether {diff} bad sources from total {len(sources)} sources have been removed")
    return sources_copy


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
        src_entries = self.T[130000:140000]
        self.types = src_entries['class']
        img_folders = os.listdir(IMG_ROOT)

        img_files = []
        for i in range(len(img_folders)):
            img_folder = img_folders[i]
            tmp = []
            for f_name in os.listdir(os.path.join(IMG_ROOT, img_folder)):
                img_path = os.path.join(os.path.join(IMG_ROOT, img_folder), f_name)
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
                     ) -> List[Tuple[List[str], int]]:

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

        src_files = filter_dataset(src_files)

        for idx, img_f in enumerate(src_files):
            item = img_f, class_to_idx[self.types[idx]]
            instances.append(item)

        return instances

    @staticmethod
    def __fits_loader(multichannel_fits):
        multichannel_fits.sort()
        kernel = Gaussian2DKernel(x_stddev=1)

        if len(multichannel_fits) != 5:
            print(multichannel_fits)
            raise IndexError

        def fits_normalization(img_dat):
            vmax = np.max(img_dat)
            vmin = np.min(img_dat)
            img_dat[:][:] = (img_dat[:][:] - vmin) / (vmax - vmin)
            return img_dat

        for i in range(5):
            img_dat = fits.getdata(multichannel_fits[i])
            multichannel_fits[i] = interpolate_replace_nans(img_dat, kernel)

        dat = np.stack(multichannel_fits, axis=2)
        dat = fits_normalization(dat)

        return dat
