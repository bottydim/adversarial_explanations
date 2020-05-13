import numpy as np
from matplotlib import pyplot as plt


def boundary_line(kernel, models, X_train, X_test, Y_train, Y_test, plot_points=True):
    '''

    :param plot_points:
    :param kernel: “linear” | “poly” | “rbf” | “sigmoid” | “cosine” |
    :param models:
    :param X_train:
    :param X_test:
    :param Y_train:
    :param Y_test:
    :return:
    '''

    #     kernel = "linear"
    X1, X2, X_set, y_set, X_mesh = pca_transform(kernel, X_train, X_test, Y_train, Y_test)
    Zs = pca_predict(X_mesh, models)

    plot_boundary(X1, X2, X_set, y_set, Zs, plot_points)


def plot_boundary(X1, X2, X_set, y_set, Zs, plot_points):
    from matplotlib.colors import ListedColormap
    models_str = ["Original", "Modified"]
    if len(Zs) > 2:
        models_str = ["M-{}".format(i) for i in range(len(Zs))]
    f, axes = plt.subplots(1, len(Zs), sharey=True)
    for i, Z in enumerate(Zs):
        ax = axes[i]
        ax.set_title(models_str[i])
        ax.contourf(X1, X2, Z.reshape(X1.shape),
                    alpha=0.5, cmap=ListedColormap(('red', 'green')))
        if plot_points:
            for k, j in enumerate(np.unique(y_set)):
                x_idx = np.where(y_set == j)[0]
                ax.scatter(X_set[x_idx, 0], X_set[x_idx, 1],
                           c=ListedColormap(('red', 'green'))(k), label=j)

            # plt.title('{} Boundary Line with {} PCA' .format(algo_name, kernel))
            # plt.xlabel('Component 1')
            # plt.ylabel('Component 2')
            # plt.legend()
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=3)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=3)


def pca_predict(X_mesh, models):
    Zs = []
    for i, model in enumerate(models):
        classifier = model
        Z = classifier.predict(X_mesh)
        Z = np.argmax(Z, axis=1)
        Zs.append(Z)
    return Zs


# noinspection PyUnusedLocal,PyUnusedLocal,PyUnusedLocal
def lda_tranform(kernel, X_train, X_test, Y_train, Y_test):
    #???? TODO!
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    clf = LinearDiscriminantAnalysis(n_components=2)
    X, y = X_train, np.argmax(Y_train, axis=1)

    clf.fit(X, y)
    LinearDiscriminantAnalysis(n_components=2, priors=None, shrinkage=None,
                               solver='svd', store_covariance=False, tol=0.0001)
    X_train_reduced = clf.predict(X)
    X_test_reduced = clf.predict(X_test)

    y_train = np.argmax(Y_train, axis=1)
    y_test = np.argmax(Y_test, axis=1)

    X_set, y_set = np.concatenate([X_train_reduced, X_test_reduced], axis=0), np.concatenate([y_train, y_test], axis=0)
    num = 1e2
    X1, X2 = np.meshgrid(np.linspace(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, num=num),
                         np.linspace(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, num=num))

    return X_train_reduced, X_test_reduced, clf, y_train, y_test


def tsne_transform():
    pass
    # X_embedded = TSNE(n_components=2).fit_transform(X)


def pca_transform(kernel, X_train, X_test, Y_train, Y_test):
    X_test_reduced, X_train_reduced, reduction, y_test, y_train = pca_reduce(X_test, X_train, Y_test, Y_train, kernel)

    # Boundary Line
    X_set, y_set = np.concatenate([X_train_reduced, X_test_reduced], axis=0), np.concatenate([y_train, y_test], axis=0)
    num = 1e2
    X1, X2 = np.meshgrid(np.linspace(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, num=num),
                         np.linspace(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, num=num))
    X_mesh = reduction.inverse_transform(np.array([X1.ravel(), X2.ravel()]).T)
    return X1, X2, X_set, y_set, X_mesh


def pca_reduce(X_test, X_train, Y_test, Y_train, kernel):
    from sklearn.decomposition import KernelPCA
    reduction = KernelPCA(n_components=2, kernel=kernel, fit_inverse_transform=True, n_jobs=7)
    X_train_reduced = reduction.fit_transform(X_train)
    X_test_reduced = reduction.transform(X_test)
    y_train = np.argmax(Y_train, axis=1)
    y_test = np.argmax(Y_test, axis=1)
    return X_test_reduced, X_train_reduced, reduction, y_test, y_train