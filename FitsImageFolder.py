from astropy.io import fits
from torchvision.datasets import DatasetFolder


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
    def __fits_loader(filename):
        data = fits.getdata(filename)
        return data
