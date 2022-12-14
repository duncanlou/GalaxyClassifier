from utils import append_new_line

if __name__ == '__main__':
    append_new_line("training_notes/test_notes.txt",
                    "Training_epoch_loss, Validation_epoch_loss, Training_epoch_accuracy, Validation_epoch_accuracy")
    append_new_line("training_notes/test_notes.txt",
                    "Training_epoch_loss, Validation_epoch_loss, Training_epoch_accuracy, Validation_epoch_accuracy")
    append_new_line("training_notes/test_notes.txt",
                    "Training_epoch_loss, Validation_epoch_loss, Training_epoch_accuracy, Validation_epoch_accuracy")
    # df_p = pd.read_csv("/home/duncan/PycharmProjects/MyResearchProject_Duncan/data/preprocessed_cat/PS_p_samples.csv")
    # df_n = pd.read_csv("/home/duncan/PycharmProjects/MyResearchProject_Duncan/data/preprocessed_cat/PS_n_samples.csv")
    #
    # df_p_ROGUE = df_p[df_p["Catalog_From"] == 'Norris06']
    # df_n_ROGUE = df_n[df_n["Catalog_From"] == 'Norris06']
    #
    # df_p_ROGUE.to_csv("../data/preprocessed_cat/PS_p_Norris06_samples.csv", index=False)
    # df_n_ROGUE.to_csv("../data/preprocessed_cat/PS_n_Norris06_samples.csv", index=False)

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
