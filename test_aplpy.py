import os

import aplpy
from astropy.io import fits

root = "data/sources/galaxy"

sources = os.listdir(root)


def get_fits_dict(imgs, source_dir):
    if len(imgs) != 5:
        raise IOError

    fits_dict = {}

    for f in imgs:
        if f.startswith("stack_g"):
            fits_dict[0] = os.path.join(root, source_dir, f)
        elif f.startswith("stack_r"):
            fits_dict[1] = os.path.join(root, source_dir, f)
        elif f.startswith("stack_i"):
            fits_dict[2] = os.path.join(root, source_dir, f)
        elif f.startswith("stack_z"):
            fits_dict[3] = os.path.join(root, source_dir, f)
        else:
            fits_dict[4] = os.path.join(root, source_dir, f)

    return fits_dict


new_dir_name = "data/sources/new_gal"

for s in sources:
    full_s = os.path.join(root, s)
    if not os.path.isdir(full_s):
        continue

    s_path = os.path.join(root, s)
    imgs = os.listdir(s_path)
    fits_dict = get_fits_dict(imgs, s)
    fits_list = [fits_dict[0], fits_dict[1], fits_dict[2], fits_dict[3], fits_dict[4]]

    path_arr = imgs[0].split("_")
    ra_s = path_arr[2][2:]
    dec_s = path_arr[3][3:]
    if float(dec_s) > 0:
        dec_s = f"+{dec_s}"

    dest = f"data/sources/new_gal/{ra_s}_{dec_s}_cube.fits"

    s_dest = f"{ra_s}_{dec_s}_cube.fits"

    if not s_dest in os.listdir(new_dir_name):
        print(s_dest)
        aplpy.make_rgb_cube(fits_list, dest)
    else:
        print(s_dest, "已存在")
        continue
