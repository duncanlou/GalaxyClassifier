import os
from typing import Tuple, Dict, List, Optional, Callable, cast
import shutil

import numpy as np
import copy
from astropy.io import fits
from astropy.convolution import interpolate_replace_nans, Gaussian2DKernel

from torchvision.datasets import DatasetFolder, folder

kernel = Gaussian2DKernel(x_stddev=1)


def filter_dataset(sources):
    sources_copy = copy.deepcopy(sources)

    for source in sources:
        for j in range(5):
            path_levels_arr = source[j].split(os.path.sep)
            try:
                fits_dat = fits.getdata(source[j])
                row, col = np.where(np.isnan(fits_dat))
                if len(row) >= 50:  # if missing values pixels number is larger than 100, skip this image
                    fits_n = path_levels_arr[-3] + os.path.sep + path_levels_arr[-2] + os.path.sep + path_levels_arr[-1]
                    print(f"{len(row)} NaN values are found in: {fits_n}")
                    sources_copy.remove(source)
                    shutil.rmtree(path_levels_arr[-3] + os.path.sep + path_levels_arr[-2])
                    break
            except OSError as err:
                fits_n = path_levels_arr[-3] + os.path.sep + path_levels_arr[-2] + os.path.sep + path_levels_arr[-1]
                print(f"invalid fits file: {fits_n}, error: {err}")
                sources_copy.remove(source)
                shutil.rmtree(path_levels_arr[-3] + os.path.sep + path_levels_arr[-2])
                break
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
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                if len(fnames) == 5:
                    fits_in_a_source = []
                    for fname in sorted(fnames):
                        path = os.path.join(root, fname)
                        if is_valid_file(path):
                            fits_in_a_source.append(path)
                            if target_class not in available_classes:
                                available_classes.add(target_class)
                    item = fits_in_a_source, class_index
                    instances.append(item)

        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            if extensions is not None:
                msg += f"Supported extensions are: {', '.join(extensions)}"
            raise FileNotFoundError(msg)

        return instances






    @staticmethod
    def __fits_loader(multichannel_fits):
        multichannel_fits.sort()

        if len(multichannel_fits) != 5:
            print(multichannel_fits)
            raise IndexError

        def fits_normalization(img_dat):
            # img_dat.shape：(240, 240, 5)
            vmax = np.max(img_dat)
            vmin = np.min(img_dat)
            img_dat[:, :, :] = (img_dat - vmin) / (vmax - vmin)

        for i in range(5):
            img_dat = fits.getdata(multichannel_fits[i])
            multichannel_fits[i] = interpolate_replace_nans(img_dat, kernel)

        dat = np.stack(multichannel_fits, axis=2)
        fits_normalization(dat)

        return dat
