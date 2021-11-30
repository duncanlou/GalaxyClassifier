import os
import shutil
from astropy.table import Table


src_root_path = os.path.join(os.getcwd(), "data/images14")
T = Table.read("data/DuncanSDSSdata.tbl", format="ascii.ipac")
src_table = T[130000:150000]
ra_tab = list(src_table['ra'])
dec_tab = list(src_table['dec'])
sourceObjs = []
source_folders = os.listdir(src_root_path)

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
        idx = ra_tab.index(ra)
        entry = T[idx]
        source_dir_full_path = os.path.join(src_root_path, source_dir)
        classType = entry['class']
        if classType == 'GALAXY':
            shutil.move(source_dir_full_path, "data/sources/galaxy")
        elif classType == 'QSO':
            shutil.move(source_dir_full_path, "data/sources/quasar")
        elif classType == 'STAR':
            shutil.move(source_dir_full_path, "data/sources/star")
        else:
            raise IOError
        print(f"{i}----{source_dir_full_path} has been moved")
        # single_source_fits = []
        # for fits_files in os.listdir(source_dir_full_path):
        #     fits_file_path = os.path.join(source_dir_full_path, fits_files)
        #     single_source_fits.append(os.path.join(os.getcwd(), fits_file_path))
        # sourceID = entry['specObjID']
        # ra = entry['ra']
        # dec = entry['dec']
        # classTypes = entry['class']
        # sourceObjs.append(
        #     PSsourcePt(srcID=sourceID, ra=ra, dec=dec, srcType=classTypes, fits_img=single_source_fits))
