import os.path

from astropy.io import fits

# already checked: iamges1, images2, images3, images4
root = "/media/duncan/PANSTARRS_image_/PS_big_cutouts/North_ecliptic_region/images5"

sources = os.listdir(root)
bad_sources = []
count = 0

for s in sources:
    count += 1
    print(count)
    s_path = os.path.join(root, s)
    fits_imgs = os.listdir(s_path)
    # if len(fits_imgs) != 5:
    #     bad_sources.append(s_path)
    #     print(s)
    for f in fits_imgs:
        try:
            img = fits.getdata(os.path.join(s_path, f))
        except IOError as ioe:
            print(ioe)
            print(f"{s} is a bad source")
            bad_sources.append(s_path)
        except ValueError as vle:
            print(vle)
            print(f"{s} is a bad source")
            bad_sources.append(s_path)
        except TypeError as tpe:
            print(tpe)
            print(f"{s} is a bad source")
            bad_sources.append(s_path)

# redownload_path = "/home/duncan/PycharmProjects/bulk_download_sources/data/redownloaded"
# names = os.listdir(redownload_path)
# for name in names:
#     source_path = os.path.join(redownload_path, name)
#     dst = os.path.join(root, name)
#     shutil.copytree(source_path, dst)
