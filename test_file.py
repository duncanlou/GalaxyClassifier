import os
import shutil
import numpy as np
from astropy.io import fits

from astropy.table import Table


T = Table.read('data/DuncanSDSSdata.tbl', format='ipac')
ras = list(T['ra'])
decs = list(T['dec'])

position = []
for i in range(len(ras)):
    position.append((ras[i], decs[i]))


root = "data/images1"
fits_f = os.listdir(root)
gfits = []
rfits = []
ifits = []
zfits = []
yfits = []
for f in fits_f:
    if f.endswith('g.fits'):
        gfits.append(f.removesuffix('.g.fits').removeprefix('t'))
    elif f.endswith('r.fits'):
        rfits.append(f.removesuffix('.r.fits').removeprefix('t'))
    elif f.endswith('i.fits'):
        ifits.append(f.removesuffix('.i.fits').removeprefix('t'))
    elif f.endswith('z.fits'):
        zfits.append(f.removesuffix('.z.fits').removeprefix('t'))
    else:
        yfits.append(f.removesuffix('.y.fits').removeprefix('t'))

print(len(gfits))
print(len(rfits))
print(len(ifits))
print(len(zfits))
print(len(yfits))

delta1 = set(rfits) - set(gfits)
print(delta1)

delta2 = set(rfits) - set(zfits)
print(delta2)

for i in gfits:
    if i.__contains__('+'):
        ra, dec = i.split('+')
        dec = '+' + dec
    if i.__contains__('-'):
        ra, dec = i.split('-')
        dec = '-' + dec
    ra = float(ra)
    dec = float(dec)
    if not position.__contains__((ra, dec)):
        print(ra, dec)






