import os

PS_IMAGE_ROOT = "/mnt/DataDisk/Duncan/sources_for_Sean"

classes = os.listdir(PS_IMAGE_ROOT)

count = 0
for clz in classes:
    path = os.path.join(PS_IMAGE_ROOT, clz)
    source_dirs = os.listdir(path)
    for source_dir in source_dirs:
        source_path = os.path.join(path, source_dir)
        imgs = os.listdir(source_path)
        if len(imgs) != 5:
            count += 1
            print(source_path)
