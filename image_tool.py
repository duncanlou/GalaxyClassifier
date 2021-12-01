import os
import numpy as np
from astropy.io import fits
from astropy.table import Table

T = Table.read("data/DuncanSDSSdata.tbl", format="ipac")
ras = T["ra"]
decs = T["dec"]

positions = []
for i in range(len(ras)):
    ra = round(ras[i], 6)
    dec = round(decs[i], 6)
    positions.append((ra, dec))

classes = T["class"]

files = os.listdir("data/sources/STAR")

for f in files:
    name_arr = f.split("_")
    ra_f = np.float(name_arr[0])
    ra_f = round(ra_f, 6)
    dec_f = np.float(name_arr[1])
    dec_f = round(dec_f, 6)
    pos = ra_f, dec_f
    if positions.__contains__(pos):
        idx = positions.index(pos)
        if classes[idx] != "STAR":
            print("分类分错了", pos, "真实的类", classes[idx])
            raise IOError
    else:
        i = list(ras).index(ra_f)
        table_dec = decs[i]
        if abs(table_dec - dec_f) > 1e-4:
            print(f"没在table中找到{pos}")








