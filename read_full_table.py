from astropy.table import Table

sdss_cat = Table.read('SDSS_33col.tbl', format='ipac')
wise_cat = Table.read('wise_catlog.tbl', format='ipac')
