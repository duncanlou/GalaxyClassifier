import os.path
import shutil

if __name__ == '__main__':
    VLASS_IMAGE_ROOT = "/mnt/DataDisk/Duncan/Pan-STARRS_Big_Cutouts/VLASS_training_data"
    # df = pd.read_csv("../data/preprocessed_cat/PS_p_RGZ_samples.csv")
    # T = Table.from_pandas(df)
    # component_names = list(T["VLASS_component_name"])
    files = os.listdir(VLASS_IMAGE_ROOT)

    count = 0
    for name in files:
        count += 1
        print(count)
        file_name = os.path.join(VLASS_IMAGE_ROOT, f"{name}")
        shutil.copy(src=file_name, dst=f"../RGZ_fits_files/{name}")

    # bad_files = []
    #
    # count = 0
    #
    # for s in sources:
    #     count += 1
    #     print(count)
    #     s_path = os.path.join(root, s)
    #     fits_fs = os.listdir(s_path)
    #     for f in fits_fs:
    #         f_path = os.path.join(s_path, f)
    #         try:
    #             image = fits.getdata(f_path)
    #         except OSError as e:
    #             print(f"{s} --- {f} is a bad file: {e}")
    #             bad_files.append(s)
    #         except TypeError as te:
    #             print(f"{s} --- {f} is a bad file: {te}")
    #             bad_files.append((s))
