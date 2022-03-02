import os
import random

import pandas as pd

BASE_PATH = "data/sources"
folder1 = 'GALAXY'
folder2 = 'QSO'
folder3 = 'STAR'

path1 = os.path.join(BASE_PATH, folder1)
path2 = os.path.join(BASE_PATH, folder2)
path3 = os.path.join(BASE_PATH, folder3)

subdirs1 = os.listdir(path1)
subdirs2 = os.listdir(path2)
subdirs3 = os.listdir(path3)


def getSourceCoord(subdirs):
    coordinates = []
    for dir in subdirs:
        if dir.__contains__('m'):
            ra, dec = dir.split('m')
            dec = "-" + dec
        elif dir.__contains__('p'):
            ra, dec = dir.split('p')
        else:
            continue
        coordinates.append((float(ra), float(dec)))
    return coordinates


galaxy_coord = getSourceCoord(subdirs1)
QSO_coord = getSourceCoord(subdirs2)
star_coord = getSourceCoord(subdirs3)

coord = galaxy_coord + QSO_coord + star_coord
random.shuffle(coord)

positions = pd.DataFrame(coord, columns=['ra', 'dec'])
