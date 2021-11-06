import numpy as np
from astropy.io import fits
from scipy import ndimage



class ImageFixingTool:

    def __init__(self):
        return

    def conv_mapping(x):
        """
            When the fifth value (x[4]) of the filter array (the center of the window)
            is null, replace it with the mean of the surrounding values.
            """
        if np.isnan(x[4]) and not np.isnan(np.delete(x, 4)).all():
            return np.nanmean(np.delete(x, 4))
        else:
            return x[4]

    def fix_damaged_image(dat):
        mask = np.ones((3, 3))
        result = ndimage.generic_filter(dat, function=conv_mapping, footprint=mask, mode='constant', cval=np.NaN)

        while np.isnan(result).any():
            result = ndimage.generic_filter(result, function=conv_mapping, footprint=mask, mode='constant',
                                            cval=np.NaN)
            row, col = np.where(np.isnan(result))
            print(f"Bad pixel num: {len(row)}")
        print(f"Image:  {images} has been fixed completely, congratulations! ")




