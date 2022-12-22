import os.path
import shutil

if __name__ == '__main__':
    VLASS_IMAGE_ROOT = "/mnt/DataDisk/Duncan/Pan-STARRS_Big_Cutouts/PS_training_data"
    radio_components = ['VLASS1QLCIR J033537.02-273432.5',
                        'VLASS1QLCIR J033113.95-275519.3',
                        'VLASS1QLCIR J032746.89-271742.6',
                        'VLASS1QLCIR J033125.03-281810.9',
                        'VLASS1QLCIR J032636.78-280752.0',
                        'VLASS1QLCIR J033457.89-272637.4',
                        'VLASS1QLCIR J032916.31-272340.3',
                        'VLASS1QLCIR J032718.99-284640.2',
                        'VLASS1QLCIR J033241.99-273818.6',
                        'VLASS1QLCIR J033226.96-274106.7',
                        'VLASS1QLCIR J033111.67-273143.7',
                        'VLASS1QLCIR J033210.72-272634.9',
                        'VLASS1QLCIR J032910.90-272717.2',
                        'VLASS1QLCIR J032642.35-280805.0',
                        'VLASS1QLCIR J033009.38-281849.4',
                        'VLASS1QLCIR J033438.54-272721.0',
                        'VLASS1QLCIR J033449.00-281144.5',
                        'VLASS1QLCIR J032933.82-284139.4',
                        'VLASS1QLCIR J032934.26-273431.5',
                        'VLASS1QLCIR J033129.72-281817.8']
    # df = pd.read_csv("../data/preprocessed_cat/PS_p_RGZ_samples.csv")
    # T = Table.from_pandas(df)
    # component_names = list(T["VLASS_component_name"])

    for name in radio_components:
        dir_name = os.path.join(VLASS_IMAGE_ROOT, name)
        shutil.copytree(src=dir_name, dst=f"wrongly_predicted_Norris_ps/{dir_name}")

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
