import os
from typing import Dict, Optional, Callable, List, Tuple

import numpy as np
from astropy.io import fits
from astropy.table import Table
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import find_classes

from utils import remove_nan, cal_luptitude, CatPSimgMinMax

T = Table.read('data/SDSSxWISE_cat.tbl', format='ipac')
source_id_col = list(T['source_id'])
w1flux = T['w1flux']
w2flux = T['w2flux']


def make_dataset(
        directory: str,
        class_to_idx: Optional[Dict[str, int]] = None,
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class).

    See :class:`DatasetFolder` for details.

    Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
    by default.
    """
    directory = os.path.expanduser(directory)

    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)
    elif not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

    instances = []
    available_classes = set()

    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, dirs, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for idx, dir in enumerate(sorted(dirs)):
                path = os.path.join(root, dir)
                if os.path.isdir(path):
                    item = path, class_index
                    instances.append(item)
                    if target_class not in available_classes:
                        available_classes.add(target_class)

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        if extensions is not None:
            msg += f"Supported extensions are: {', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return instances


class FitsImageFolder(DatasetFolder):
    EXTENSIONS = ('.fits',)

    def __init__(self, root, transform=None, target_transform=None,
                 loader=None):
        if loader is None:
            loader = self.__fits_loader

        super(FitsImageFolder, self).__init__(root, loader, self.EXTENSIONS,
                                              transform=transform,
                                              target_transform=target_transform)

    @staticmethod
    def make_dataset(
            directory: str,
            class_to_idx: Dict[str, int],
            extensions: Optional[Tuple[str, ...]] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        This can be overridden to e.g. read files from a compressed zip file instead of from the disk.

        Args:
            directory (str): root dataset directory, corresponding to ``self.root``.
            class_to_idx (Dict[str, int]): Dictionary mapping class name to class index.
            extensions (optional): A list of allowed extensions.
                Either extensions or is_valid_file should be passed. Defaults to None.
            is_valid_file (optional): A function that takes path of a file
                and checks if the file is a valid file
                (used to check of corrupt files) both extensions and
                is_valid_file should not be passed. Defaults to None.

        Raises:
            ValueError: In case ``class_to_idx`` is empty.
            ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
            FileNotFoundError: In case no valid file was found for any class.

        Returns:
            List[Tuple[str, int]]: samples of a form (path_to_sample, class)
        """
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError(
                "The class_to_idx parameter cannot be None."
            )
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)

    @staticmethod
    def __fits_loader(fits_name):
        src_dir_contents = os.listdir(fits_name)
        src_dir_contents.sort()

        paths = fits_name.split(os.path.sep)
        short_name = paths[-1]
        idx = source_id_col.index(short_name)
        w1flux_single_source = w1flux[idx]
        w2flux_single_source = w2flux[idx]
        m1, m2 = cal_luptitude(w1flux_single_source, w2flux_single_source)
        wise_magnitude_info = np.asarray([m1, m2])
        img_list = []
        # for i in range(5):
        #     fits_image = src_dir_contents[i]
        #     img_path = os.path.join(fits_name, fits_image)
        #     raw_data = fits.getdata(img_path)
        #     single_band_img_dat = remove_nan(raw_data)
        #     img_list.append(single_band_img_dat)

        # img_dat = np.stack(img_list, axis=2)
        # img_dat = optimize_image(img_dat)
        # img_dat = normalize(img_dat)

        # image_cube_vmax = 0

        for i in range(5):
            fits_f = src_dir_contents[i]
            single_channel_img_dat = fits.getdata(os.path.join(fits_name, fits_f)).squeeze()

            ################ Step 1. remove nan or np.inf by interplating new value ######################
            single_channel_img_dat = remove_nan(single_channel_img_dat)

            ################ Step 2. CatPSMinMax ######################
            scaleVmin, scaleVmax = CatPSimgMinMax(single_channel_img_dat)
            single_channel_img_dat = np.where(single_channel_img_dat > scaleVmax, scaleVmax, single_channel_img_dat)

            ################ Step 3. Min-Max Scaling ######################
            vmin = np.min(single_channel_img_dat)
            vmax = np.max(single_channel_img_dat)
            single_channel_img_dat = (single_channel_img_dat - vmin) / (vmax - vmin)

            # if vmax > image_cube_vmax:
            #     image_cube_vmax = vmax

            img_list.append(single_channel_img_dat)

        img_dat = np.stack(img_list, axis=2)

        return img_dat, wise_magnitude_info
