from astropy.table import Table
import os, shutil

T = Table.read("data/DuncanSDSSdata.tbl", format="ipac")
classes = list(T['class'])
ras = list(T['ra'])
decs = list(T['dec'])

count_star = 0
count_quasar = 0
count_galaxy = 0

for dir in os.listdir("data/sources/star"):
    if os.path.isdir(os.path.join("data/sources/star", dir)) and not dir.__contains__("p"):
        print("不正确的源目录： ", dir)
        raise IOError
    if dir.__contains__("p"):
        ra, dec = dir.split('p')
        ra = float(ra)

        dec = float(dec)
        table_index = -1
        idx1 = ras.index(ra)
        idx2 = decs.index(dec)
        if idx1 != idx2:
            print(ra, dec)
            if T[idx1]['dec'] == dec:
                table_index = idx1
            elif T[idx2]['ra'] == ra:
                table_index = idx2
            else:
                raise ValueError
        else:
            table_index = idx1
        which_type = classes[table_index]
        print(which_type)
        if which_type != 'STAR':
            src = os.path.join(os.getcwd() + "/data/sources/star", dir)
            if which_type == 'QSO':
                dest = os.path.join(os.getcwd() + "/data/sources/quasar")
                count_quasar += 1
            elif which_type == 'GALAXY':
                dest = os.path.join(os.getcwd() + "/data/sources/galaxy")
                count_galaxy += 1
            else:
                raise ValueError
            shutil.move(src, dest)
