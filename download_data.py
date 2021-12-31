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


fitsPath = getFits(ra=140.34763, dec=26.525795)
print(fitsPath)
