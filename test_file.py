from astropy.table import Table
import os, shutil

T = Table.read("data/DuncanSDSSdata.tbl", format="ipac")
classes = list(T['class'])
ras = list(T['ra'])
decs = list(T['dec'])

count_star = 0
count_quasar = 0
count_galaxy = 0

root = "/mnt/DataDisk/Duncan/images1"

for fits_f in os.listdir(root):
    origin_name = fits_f
    paths = origin_name.split("_")
    channel = paths[1]
    ra = paths[2].removeprefix("ra")
    dec = paths[3].removeprefix("dec")
    if float(dec) > 0:
        dec = "+" + str(dec)
    new_name = f"t{ra}{dec}.{channel}.fits"
    original_full_name = os.path.join(root, origin_name)
    new_full_name = os.path.join(root, new_name)
    os.rename(original_full_name, new_full_name)



