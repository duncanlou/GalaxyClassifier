from astropy.table import Table
from torch.utils.data import Dataset

from FitsFolder import FitsFolder


class CelebDataset(Dataset):
    def __init__(self, root_dir, source_table, transform=None, target_transform=None):
        print("CelebDataset.__init__() will be invoked how many times?")
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.T = source_table
        self.fits_folder = FitsFolder(root=self.root_dir, data_table=self.T)

    def __len__(self):
        return len(self.fits_folder)

    def __getitem__(self, idx):
        img_dat, label = self.fits_folder[idx]

        if self.transform:
            img_dat = self.transform(img_dat)
        if self.target_transform:
            label = self.target_transform(label)
        return img_dat, label
