from __future__ import print_function, division

from torch.utils.data import Dataset


class XMatchDataset(Dataset):
    def __init__(self, dataset, radio_transform=None, opt_transform=None):
        self.dataset = dataset
        self.radio_transforms = radio_transform
        self.opt_transforms = opt_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):  # radio_img_dat, ps_imgcube, wise_magnitude_info, cutout_position_info, label
        sample = self.dataset[idx]
        radio_image_dat = sample[0]
        ps_imgcube = sample[1]
        wise_mag_info = sample[2]
        ps_cutout_pos_info = sample[3]
        label = sample[4]
        ps_source_identity = sample[5]
        if self.radio_transforms:
            radio_image_dat = self.radio_transforms(radio_image_dat)
        if self.opt_transforms:
            ps_imgcube = self.opt_transforms(ps_imgcube)

        return radio_image_dat, ps_imgcube, wise_mag_info, ps_cutout_pos_info, label, ps_source_identity
        # return radio_image_dat, ps_imgcube, wise_mag_info, ps_cutout_pos_info, label
