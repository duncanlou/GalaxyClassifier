from torch.utils.data import Dataset

from FitsFolder import FitsFolder


class CelebDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.fits_folder = FitsFolder(root=self.root_dir)

    def __len__(self):
        return len(self.fits_folder)

    def __getitem__(self, idx):
        img_dat, label = self.fits_folder[idx]
        if self.transform:
            img_dat = self.transform(img_dat)
        if self.target_transform:
            label = self.target_transform(label)
        return img_dat, label


