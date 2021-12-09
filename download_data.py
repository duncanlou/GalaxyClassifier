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


fitsPath = getFits(ra=187.14697, dec=-2.126080)
print(fitsPath)
# for i in range(tab_len):
#     fitsPath = getFits(T['ra'][i], T['dec'][i])
#     print(i)
