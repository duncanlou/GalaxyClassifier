from astropy.io import ascii

# img_QSO = os.listdir("test_QSO")
# img_QSO.sort()
# QSO_dat_list = []
# for image in img_QSO:
#     dat = fits.getdata(os.path.join(os.getcwd(), "test_QSO", image))
#     QSO_dat_list.append(dat)
#
# img_star = os.listdir("test_star")
# img_star.sort()
# star_dat_list = []
# for image in img_star:
#     dat = fits.getdata(os.path.join(os.getcwd(), "test_star", image))
#     star_dat_list.append(dat)
#
# position = (120, 120)
# aperture = CircularAperture(position, r=3.)
#
# for data in QSO_dat_list:
#     phot_table = aperture_photometry(data, aperture, method='exact')
#     print(phot_table[0]['aperture_sum'])
#
# print("++++++++++++++++++++++++++++")
#
# for data in star_dat_list:
#     phot_table = aperture_photometry(data, aperture, method='exact')
#     print(phot_table[0]['aperture_sum'])

dat = ascii.read("/home/duncan/Desktop/DataSDSSmerged.tbl")
print(dat)
