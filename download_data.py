import os

from astropy.table import Table
import numpy as np
from astropy.logger import Logger
from panstamps.downloader import downloader


def getFits(ra, dec):
    fitsPaths, jpegPaths, colorPath = downloader(
        log=Logger(name="duncan's research"),
        fits=True,
        jpeg=False,
        ra=ra,
        dec=dec,
        color=False,
        imageType='stack',
        filterSet='grizy'
    ).get()
    return fitsPaths


IMG_ROOT = "./images14"
T = Table.read("data/DuncanSDSSdata.tbl", format="ascii.ipac")
src_table = T[130000:140000]
ra_tab = list(src_table['ra'])
dec_tab = list(src_table['dec'])

source_folders = os.listdir(IMG_ROOT)
print(len(source_folders))

position = []

for i in range(len(source_folders)):
    source_dir = source_folders[i]

    if source_dir.__contains__('p'):
        ra, dec = source_dir.split('p')
    elif source_dir.__contains__('m'):
        ra, dec = source_dir.split('m')
        dec = "-" + dec
    else:
        raise IOError
    ra = float(ra)
    dec = float(dec)

    if ra_tab.__contains__(ra) and dec_tab.__contains__(dec):  # 表格中的源存在于目录中
        position.append((ra, dec))
    else:
        pass

# for i in range(tab_len):
#     fitsPath = getFits(T['ra'][i], T['dec'][i])
#     print(i)
