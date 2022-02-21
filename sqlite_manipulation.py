import sqlite3

from astropy.table import Table

connection = sqlite3.connect("SDSS_source_dat.db")
SDSS_dat = Table.read("SDSS_33col.tbl", format="ipac")

cursor = connection.cursor()
cursor.execute("CREATE TABLE SDSS_sources (specObjID INTEGER,  ra REAL, dec REAL, class varchar(20), )")
