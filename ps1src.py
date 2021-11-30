class PSSource:
    __slots__ = ['objID', 'ra', 'dec', 'label']

    def __init__(self, objID, ra, dec, label):
        self.objID = objID,
        self.ra = ra,
        self.dec = dec,
        self.label = label

    def __str__(self):
        pass
