import matplotlib.pyplot as plt


def matplotlib_imshow(img, one_channel=True):
    if one_channel:
        img = img.mean(dim=0)
    # img = img / 2 + 0.5
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap='Greys')
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()



def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)