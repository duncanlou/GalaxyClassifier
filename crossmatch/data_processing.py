import pandas as pd

if __name__ == '__main__':
    df_p_Norris = pd.read_csv("../data/preprocessed_cat/PS_p_Norris06_samples.csv")

    df_n = pd.read_csv("../data/preprocessed_cat/PS_n_samples.csv")

    df_n_Norris_all = df_n[df_n.VLASS_component_name.isin(df_p_Norris.VLASS_component_name)].reset_index(drop=True)

    # df_n_2 = df_n_1.groupby("VLASS_component_name").apply(lambda x: x.sample(n=1))
    df_n_Norris_all.to_csv("../data/preprocessed_cat/PS_n_samples_Norris_all.csv", index=False)
    # df_p.to_csv("../data/preprocessed_cat/PS_p_samples_RGZ_ROGUE1000.csv", index=False)

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
