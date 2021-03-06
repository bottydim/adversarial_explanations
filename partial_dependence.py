import random
from collections import defaultdict

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from scipy.stats.mstats import mquantiles
from sklearn.exceptions import NotFittedError
# noinspection PyProtectedMember,PyProtectedMember
from sklearn.inspection.partial_dependence import _grid_from_X
from sklearn.utils.extmath import cartesian

from datasets import feature_name_dict, prep_data


def _grid_from_X(X, percentiles=(0.05, 0.95), grid_resolution=100, use_unique=True):
    """Generate a grid of points based on the ``percentiles of ``X``.
    The grid is generated by placing ``grid_resolution`` equally
    spaced points between the ``percentiles`` of each column
    of ``X``.
    Parameters
    ----------
    X : ndarray
        The data
    percentiles : tuple of floats
        The percentiles which are used to construct the extreme
        values of the grid axes.
    grid_resolution : int
        The number of equally spaced points that are placed
        on the grid.
    Returns
    -------
    grid : ndarray
        All data points on the grid; ``grid.shape[1] == X.shape[1]``
        and ``grid.shape[0] == grid_resolution * X.shape[1]``.
    axes : seq of ndarray
        The axes with which the grid has been created.
    """
    if len(percentiles) != 2:
        raise ValueError('percentile must be tuple of len 2')
    if not all(0. <= x <= 1. for x in percentiles):
        raise ValueError('percentile values must be in [0, 1]')

    axes = []
    endpoints = []
    emp_percentiles = mquantiles(X, prob=percentiles, axis=0)
    for col in range(X.shape[1]):
        uniques = np.unique(X[:, col])
        if (uniques.shape[0] < grid_resolution) and use_unique:
            # feature has low resolution use unique vals
            axis = uniques
        else:
            # create axis based on percentiles and grid resolution
            axis = np.linspace(emp_percentiles[0, col] - 0.05,
                               emp_percentiles[1, col] + 0.05,
                               num=grid_resolution, endpoint=True)
        axes.append(axis)
        endpoints.append((emp_percentiles[0, col], emp_percentiles[1, col]))

    return cartesian(axes), axes


def get_endpoints(X, percentiles=(0.05, 0.95)):
    """Generate a grid of points based on the ``percentiles of ``X``.
    The grid is generated by placing ``grid_resolution`` equally
    spaced points between the ``percentiles`` of each column
    of ``X``.
    Parameters
    ----------
    X : ndarray
        The data
    percentiles : tuple of floats
        The percentiles which are used to construct the extreme
        values of the grid axes.
    grid_resolution : int
        The number of equally spaced points that are placed
        on the grid.
    Returns
    -------
    grid : ndarray
        All data points on the grid; ``grid.shape[1] == X.shape[1]``
        and ``grid.shape[0] == grid_resolution * X.shape[1]``.
    axes : seq of ndarray
        The axes with which the grid has been created.
    """
    if len(percentiles) != 2:
        raise ValueError('percentile must be tuple of len 2')
    if not all(0. <= x <= 1. for x in percentiles):
        raise ValueError('percentile values must be in [0, 1]')

    axes = []
    endpoints = []
    emp_percentiles = mquantiles(X, prob=percentiles, axis=0)

    for col in range(X.shape[1]):
        endpoints.append((emp_percentiles[0, col], emp_percentiles[1, col]))
    return endpoints


def partial_dependce_compute(model, X, features, percentiles=[0, 1], grid_resolution=100, use_unique=True):
    grid, values = _grid_from_X(X[:, features], percentiles,
                                grid_resolution, use_unique=use_unique)
    prediction_method = lambda X: model.predict_proba(X)

    averaged_predictions = get_averaged_predictions(prediction_method, X, grid, features)
    averaged_predictions = averaged_predictions.reshape(
        -1, *[val.shape[0] for val in values])
    averaged_predictions = averaged_predictions.T
    return averaged_predictions, values


def get_averaged_predictions(prediction_method, X, grid, features):
    averaged_predictions = []
    for new_values in grid:
        X_eval = X.copy()
        for i, variable in enumerate(features):
            X_eval[:, variable] = new_values[i]

        try:
            predictions = prediction_method(X_eval)
        except NotFittedError:
            raise ValueError(
                "'estimator' parameter must be a fitted estimator")

        # Note: predictions is of shape
        # (n_points,) for non-multioutput regressors
        # (n_points, n_tasks) for multioutput regressors
        # (n_points, 1) for the regressors in cross_decomposition (I think)
        # (n_points, 2)  for binary classifaction
        # (n_points, n_classes) for multiclass classification

        # average over samples
        averaged_predictions.append(np.mean(predictions, axis=0))

    # reshape to (n_targets, n_points) where n_targets is:
    # - 1 for non-multioutput regression and binary classification (shape is
    #   already correct in those cases)
    # - n_tasks for multi-output regression
    # - n_classes for multiclass classification.
    averaged_predictions = np.array(averaged_predictions).T
    if averaged_predictions.ndim == 1:
        # non-multioutput regression, shape is (n_points,)
        averaged_predictions = averaged_predictions.reshape(1, -1)
    elif averaged_predictions.shape[0] == 2:
        # Binary classification, shape is (2, n_points).
        # we output the effect of **positive** class
        averaged_predictions = averaged_predictions[1]
        averaged_predictions = averaged_predictions.reshape(1, -1)
    return averaged_predictions


def flip_X(X, feature_id, grid_resolution=10):
    col = feature_id
    uniques = np.unique(X[:, col])
    X_copy = X.copy()
    if (uniques.shape[0] > grid_resolution):
        # continious case
        raise NotImplementedError("Continious case not implemented")
    else:
        if uniques.shape[0] > 2:
            raise NotImplementedError("Non-binary case not implemented")
        for u in uniques:
            # if you want to do it one at a time
            # X_copy = X.copy()
            row_ids = np.where(X[:, col] != u)[0]
            X_copy[row_ids, col] = u

    return X_copy


def flip_x_i(models_all, model_lists_names, data_list, n_models, n_seeds):
    from evaluate import evaluate_metrics_across
    from datasets import dataset_names
    acc_list_dict_flip_dict = defaultdict(list)
    acc_list_dict_orig_dict = defaultdict(list)
    f_binary_list = [[12], [9], [], [0, 2]]

    # compute metrics
    for num_layers in range(n_models):
        print(10 * "#")
        print("num_layers {}".format(num_layers))
        print(10 * "#")
        for seed in range(n_seeds):
            print("M#{}-seed#{}".format(num_layers, seed))
            model_lists = models_all[num_layers][seed]
            acc_list_dict, loss_list_dict = evaluate_metrics_across(model_lists, f_binary_list,
                                                                    model_lists_names,
                                                                    data_list=data_list,
                                                                    use_train=True,
                                                                    do_flip=True)
            acc_list_dict_flip_dict[num_layers].append(acc_list_dict)
            acc_list_dict, loss_list_dict = evaluate_metrics_across(model_lists, f_binary_list,
                                                                    model_lists_names,
                                                                    data_list=data_list,
                                                                    use_train=True,
                                                                    do_flip=False)
            acc_list_dict_orig_dict[num_layers].append(acc_list_dict)

    # compute absolute difference
    acc_diff_dict = defaultdict(list)
    for num_layers in range(n_models):
        print(10 * "#")
        print("num_layers {}".format(num_layers))
        print(10 * "#")
        for seed in range(n_seeds):
            print("M#{}-seed#{}".format(num_layers, seed))
            acc_diff = {}
            for k, v in acc_list_dict_flip_dict[num_layers][seed].items():
                #             print(k)
                v_flip = np.array(acc_list_dict_flip_dict[num_layers][seed][k])
                v_orig = np.array(acc_list_dict_orig_dict[num_layers][seed][k])
                acc_diff[k] = np.abs(v_flip - v_orig)
            acc_diff_dict[num_layers].append(acc_diff)

    # average across seeds
    acc_diff_dict_avg_seeds = {}
    for k, v in acc_diff_dict.items():
        df_ = pd.DataFrame(v)
        # summarise accross seeds to get
        # columns models - modified, constant, original
        # rows - features - german-age, adult-gender ..
        df_avg = df_.apply(np.mean, axis=0)
        acc_diff_dict_avg_seeds[k] = df_avg

    # transform into pandas DF
    df_complexity_models = pd.DataFrame.from_dict({(i, j): acc_diff_dict_avg_seeds[i][j]
                                                   for i in acc_diff_dict_avg_seeds.keys()
                                                   for j in acc_diff_dict_avg_seeds[i].keys()},
                                                  orient='index')

    # label columns
    columns = []
    for i, features in enumerate(f_binary_list):
        for f in features:
            columns.append("{}-{}".format(dataset_names[i], feature_name_dict[dataset_names[i]][f]))
    df_complexity_models.columns = columns

    # plot
    fig, axes = plt.subplots(1, len(df_complexity_models.columns), figsize=(20, 5))
    for i, c in enumerate(df_complexity_models.columns):
        ax = axes[i]
        df_complexity_models.loc[:, c].unstack(level=1).plot(kind='line', ax=ax)
        ax.set_title(c)
        ax.set_xlabel("hidden layers")
        ax.set_ylabel("$\\vert f(x_\{i-flipped\}) - f(x_i)\\vert $")
    plt.tight_layout()
    return fig


def compute_pdp_plots_2(variables, use_unique=True, downsample=None, **kwargs):
    pdps = compute_pdps(variables, use_unique=use_unique, downsample=downsample)
    fig = plot_pdp(pdps, variables, kind="same")
    return pdps, fig


def compute_pdps(variables, use_unique=True, downsample=None, ):
    '''

    :param variables:
    :param use_unique: ??? =>> TODO rename to categorical_varaibles
    :param downsample: number of training points to estimate for
    :return:
    '''
    data_list, model_lists, f_nb_list, dataset_names, model_lists_names, x_labels = variables

    # from datasets import x_labels
    pdps = defaultdict(list)
    for i, data in enumerate(data_list):
        for j, _ in enumerate(model_lists[0][i]):
            sensitive_feature_id = f_nb_list[i][j]
            dataset_name = dataset_names[i]
            f_name = feature_name_dict[dataset_name][sensitive_feature_id]

            X_test, X_train, Y_test, Y_train = data

            for k, model_list in enumerate(model_lists):
                model_list_name = model_lists_names[k]
                print("{}:{}:{}".format(dataset_name, model_list_name, f_name))
                X, _ = X_train, Y_train
                if downsample is not None:
                    random.seed(30)
                    n_samples = X.shape[0]
                    ix_sample = random.sample(range(n_samples), min(downsample, n_samples))
                    X = X[ix_sample, :]
                # distinguish b/t a list of model which vary by feature and
                # the original model (does not vary by feature)
                if type(model_list[i]) is list:
                    model = model_list[i][j]
                else:
                    model = model_list[i]
                # model.layers[-1].activation = tf.keras.activations.softmax
                features = [sensitive_feature_id]
                percentiles = [0, 1]
                grid_resolution = 100
                averaged_predictions, values = partial_dependce_compute(model, X, features, percentiles=percentiles,
                                                                        grid_resolution=grid_resolution,
                                                                        use_unique=use_unique)
                pdps[model_list_name].append((averaged_predictions, values))
    return pdps


def compute_pdp_plots(variables, use_unique=True):
    dataset_fs, model_lists, f_nb_list, dataset_names, model_lists_names, x_labels = variables

    pdps = defaultdict(list)
    for i, f in enumerate(dataset_fs):
        for j, _ in enumerate(model_lists[0][i]):
            sensitive_feature_id = f_nb_list[i][j]
            dataset_name = dataset_names[i]
            f_name = feature_name_dict[dataset_name][sensitive_feature_id]

            Xtr, Xts, ytr, yts, Ztr, Zts = f(f_name, remove_z=False)
            X_test, X_train, Y_test, Y_train = prep_data(Xtr, Xts, ytr, yts)

            for k, model_list in enumerate(model_lists):
                model_list_name = model_lists_names[k]
                print("{}:{}:{}".format(dataset_name, model_list_name, f_name))
                X, Y = X_train, Y_train
                # distinguish b/t a list of model which vary by feature and
                # the original model (does not vary by feature)
                if type(model_list[i]) is list:
                    model = model_list[i][j]
                else:
                    model = model_list[i]
                model.layers[-1].activation = tf.keras.activations.softmax
                features = [sensitive_feature_id]
                percentiles = [0, 1]
                grid_resolution = 100
                averaged_predictions, values = partial_dependce_compute(model, X, features, percentiles=percentiles,
                                                                        grid_resolution=grid_resolution,
                                                                        use_unique=use_unique)
                pdps[model_list_name].append((averaged_predictions, values))
    fig = plot_pdp(pdps, variables, kind="same")
    return pdps, fig


def pdp_plot(ax, avg_pred, values, title=None, label=None):
    '''
    single pdp plot c.f. to plot_pdp, which plots all pdps!
    :param ax:
    :param avg_pred:
    :param values:
    :param title:
    :param label:
    :return:
    '''
    ax.plot(values, avg_pred)
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("x values")
    ax.set_ylabel("f(x)")


def plot_pdp_new(pdps, model_lists_names, *args, kind="same"):
    from datasets import x_labels, dataset_names, f_sensitive_list
    f_nb_list = f_sensitive_list

    n_cols = len(x_labels)
    n_rows = len(pdps.keys())
    if kind == "same":

        # f = plt.figure()
        # gs0 = gridspec.GridSpec(n_rows, n_cols, figure=f)
        fig, axes = plt.subplots(nrows=1, ncols=n_cols, figsize=(20, 5))

        cnt = 0
        key = list(pdps.keys())[0]
        for i, f in enumerate(dataset_names):
            for j in range(len(f_nb_list[i])):
                sensitive_feature_id = f_nb_list[i][j]
                f_name = feature_name_dict[dataset_names[i]][sensitive_feature_id]
                ax = axes[cnt]
                for k, _ in enumerate(model_lists_names):
                    model_list_name = model_lists_names[k]
                    # TODO change on else statement & try to merge both!
                    avg, values = pdps[model_list_name][cnt]
                    if type(values) is list:
                        values = np.array(values).transpose()
                    ax.plot(values, avg, label=model_list_name)
                title = x_labels[cnt]
                ax.set_title(title)
                ax.set_xlabel("x values")
                ax.set_ylabel("$p_{model}(y_1)$")
                #             pdp_plot(ax,avg,vals,x_labels[cnt])
                cnt += 1

        # cols = x_labels
        # rows = model_lists_names
        # for ax, col in zip(axes[0], cols):
        #     ax.set_title(col)

        # for ax, row in zip(axes[:,0], rows):
        #     ax.set_ylabel(row, rotation=0, size='large')
        plt.tight_layout()
        plt.legend()
    else:
        raise NotImplementedError()
        # f = plt.figure()
        # gs0 = gridspec.GridSpec(n_rows, n_cols, figure=f)
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 10))

        cnt = 0
        for i, f in enumerate(dataset_fs):
            for j, _ in enumerate(model_lists[0][i]):
                sensitive_feature_id = f_nb_list[i][j]
                f_name = feature_name_dict[dataset_names[i]][sensitive_feature_id]
                for k, model_list in enumerate(model_lists):
                    model_list_name = model_lists_names[k]
                    ax = axes[k][cnt]
                    avg, vals = pdps[model_list_name][cnt]
                    pdp_plot(ax, avg, vals, x_labels[cnt])
                cnt += 1

        cols = x_labels
        rows = model_lists_names
        for ax, col in zip(axes[0], cols):
            ax.set_title(col)

        for ax, row in zip(axes[:, 0], rows):
            ax.set_ylabel(row, rotation=0, size='large')
        plt.tight_layout()
    return fig


def plot_pdp(pdps, variables, kind="same"):
    dataset_fs, model_lists, f_nb_list, dataset_names, model_lists_names, x_labels = variables

    n_cols = len(x_labels)
    n_rows = len(model_lists)
    if kind == "same":

        # f = plt.figure()
        # gs0 = gridspec.GridSpec(n_rows, n_cols, figure=f)
        fig, axes = plt.subplots(nrows=1, ncols=n_cols, figsize=(20, 5))

        cnt = 0
        for i, f in enumerate(dataset_fs):
            for j, _ in enumerate(model_lists[0][i]):
                sensitive_feature_id = f_nb_list[i][j]
                f_name = feature_name_dict[dataset_names[i]][sensitive_feature_id]
                ax = axes[cnt]
                for k, model_list in enumerate(model_lists):
                    model_list_name = model_lists_names[k]
                    # TODO change on else statement & try to merge both!
                    avg, values = pdps[model_list_name][cnt]
                    ax.plot(values, avg, label=model_list_name)
                title = x_labels[cnt]
                ax.set_title(title)
                ax.set_xlabel("x values")
                ax.set_ylabel("$p_{model}(y_1)$")
                #             pdp_plot(ax,avg,vals,x_labels[cnt])
                cnt += 1

        # cols = x_labels
        # rows = model_lists_names
        # for ax, col in zip(axes[0], cols):
        #     ax.set_title(col)

        # for ax, row in zip(axes[:,0], rows):
        #     ax.set_ylabel(row, rotation=0, size='large')
        plt.tight_layout()
        plt.legend()
    else:

        # f = plt.figure()
        # gs0 = gridspec.GridSpec(n_rows, n_cols, figure=f)
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 10))

        cnt = 0
        for i, f in enumerate(dataset_fs):
            for j, _ in enumerate(model_lists[0][i]):
                sensitive_feature_id = f_nb_list[i][j]
                f_name = feature_name_dict[dataset_names[i]][sensitive_feature_id]
                for k, model_list in enumerate(model_lists):
                    model_list_name = model_lists_names[k]
                    ax = axes[k][cnt]
                    avg, vals = pdps[model_list_name][cnt]
                    pdp_plot(ax, avg, vals, x_labels[cnt])
                cnt += 1

        cols = x_labels
        rows = model_lists_names
        for ax, col in zip(axes[0], cols):
            ax.set_title(col)

        for ax, row in zip(axes[:, 0], rows):
            ax.set_ylabel(row, rotation=0, size='large')
        plt.tight_layout()
    return fig
