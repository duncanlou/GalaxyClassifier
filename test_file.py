import os
import shutil

root = "data/sources/new_STAR"

cubes = os.listdir(root)

for f in cubes:
    if f.endswith("cube_2d.fits"):
        f_path = os.path.join(root, f)
        dest = "data/cube2d_STAR"
        shutil.move(f_path, dest)


