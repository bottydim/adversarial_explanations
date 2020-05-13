import time
from collections import defaultdict

import dill
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import *

MODEL_ORIGINAL = "Original"
MODEL_CONSTANT = "x_i Constant"
MODEL_MODIFIED = "Modified"


def distplot(data, f=None, ax=None):
    if ax is None:
        f, ax = plt.subplots()
    elif f is not None:
        raise NotImplemented()
    sns.distplot(data, kde=False, ax=ax)
    ax2 = ax.twinx()
    ax2.set_ylim(0, 3)
    ax2.yaxis.set_ticks([])
    sns.kdeplot(data, ax=ax2)
    ax.set_xlabel('x var')
    ax.set_ylabel('Counts')
    return ax


def save_model_lists(model_lists, model_lists_names, alpha,
                     file_dir="/homes/btd26/xai-research/xai-notebooks/Adversarial Explanations/temp_store/robust_fairness"):
    f_nb_list = f_sensitive_list
    for i, f in enumerate(dataset_fs):
        for j, _ in enumerate(model_lists[0][i]):
            sensitive_feature_id = f_nb_list[i][j]
            dataset_name = dataset_names[i]
            f_name = feature_name_dict[dataset_name][sensitive_feature_id]

            Xtr, Xts, ytr, yts, Ztr, Zts = f(f_name, remove_z=False)
            X_test, X_train, Y_test, Y_train = prep_data(Xtr, Xts, ytr, yts)

            for k, model_list in enumerate(model_lists):
                model_list_name = model_lists_names[k]
                file_name = "-".join(map(lambda x: "_".join(x.split(" ")),
                                         [dataset_name, model_list_name, f_name, "alpha_{}".format(round(alpha, 0))]))
                X, Y = X_train, Y_train
                # distinguish b/t a list of model which vary by feature and
                # the original model (does not vary by feature)
                if type(model_list[i]) is list:
                    file_name + "-feature_" + str(j)
                    file_path = os.path.join(file_dir, file_name)
                    model = model_list[i][j]
                    model.save_weights(file_path)
                #                 print(file_path)
                else:
                    file_path = os.path.join(file_dir, file_name)
                    if j == 0:
                        model = model_list[i]
                        model.save_weights(file_path)


def load_model_lists(model_lists_names, file_dir, num_layers, seed, alpha=15.0):
    '''

    :param model_lists_names:
    :param file_dir:
    :param num_layers:
    :param seed:
    :param alpha:
    :return: model_list[model][dataset]
    '''
    from train_utils import build_model
    import tensorflow as tf
    file_dir = os.path.join(file_dir, "model_{}".format(num_layers), "seed_{}".format(seed))
    model_dict = defaultdict(list)
    #     for i,m_name in enumerate(model_lists_names):
    #         model_dict[m_name] = []
    f_nb_list = f_sensitive_list

    for i, f in enumerate(dataset_fs):
        optimizer = tf.keras.optimizers.Adam(lr=0.01)
        n_features = n_features_list[i]
        for j in range(len(f_nb_list[i])):
            sensitive_feature_id = f_nb_list[i][j]
            dataset_name = dataset_names[i]
            f_name = feature_name_dict[dataset_name][sensitive_feature_id]
            for k, model_list_name in enumerate(model_lists_names):
                file_name = "-".join(map(lambda x: "_".join(x.split(" ")),
                                         [dataset_name, model_list_name, f_name, "alpha_{}".format(round(alpha, 0))]))

                # distinguish b/t a list of model which vary by feature and
                # the original model (does not vary by feature)
                if model_list_name != MODEL_ORIGINAL:
                    file_name + "-feature_" + str(j)
                    file_path = os.path.join(file_dir, file_name)
                    model = build_model(input_shape=(n_features), num_layers=num_layers, optimizer=optimizer)
                    model.load_weights(file_path)
                    l = model_dict[model_list_name]
                    if l == []:
                        [l.append([]) for _ in range(len(dataset_fs))]
                    model_dict[model_list_name][i].append(model)
                #                 print(file_path)
                else:
                    file_path = os.path.join(file_dir, file_name)
                    if j == 0:
                        model = build_model(input_shape=(n_features), num_layers=num_layers, optimizer=optimizer)
                        model.load_weights(file_path)
                        model_dict[model_list_name].append(model)

    model_lists = [model_dict[m_name] for m_name in model_lists_names]
    return model_lists


def plot_unfairness_diff(x, f_diff, ax, color):
    ax.plot(x, f_diff, marker="o", color=color)


def plot_dataset_unfairness(x, f_o, f_p, f_c, ax, color):
    ax.plot(x, f_o, marker="o", color=color)
    ax.plot(x, f_p, marker="v", color=color)
    ax.plot(x, f_c, marker="x", color=color)
    if (np.abs(f_p) - np.abs(f_o)) > 0:
        color = "r"
    else:
        color = "b"
    ax.plot((x, x), (f_o, f_p), '-', color=color)


def add_sidelines(axes, text_middle="Bais", text_top="Unprivileged", text_bottom="Privileged"):
    y_top = axes[0].get_ylim()[1]
    y_bottom = axes[0].get_ylim()[0]
    line_top = y_top - 0.4 * y_top
    line_bottom = y_bottom - 0.4 * y_bottom
    # x,y = np.array([[-.25,-.25], [0,line_top]])
    # line = lines.Line2D(x, y, lw=2., color='r', alpha=1)
    # line.set_clip_on(False)
    # axes[0].add_line(line)
    # axes[0].set(ylabel="biased towards")
    # axes[0].annotate("",
    #             xy=(-.25, line_top), xycoords='data',
    #             xytext=(-.25, line_top), textcoords='data',
    #             arrowprops=dict(arrowstyle="->",
    #                             connectionstyle="arc3"),
    #             )
    x_pos = -0.15
    axes[0].annotate("", xy=(x_pos, 1), xycoords=("axes fraction", "axes fraction"),  # end
                     xytext=(x_pos, 0.6), textcoords=("axes fraction", "axes fraction"),  # start
                     arrowprops=dict(facecolor='red', arrowstyle="->",
                                     connectionstyle="arc3"))
    axes[0].annotate("", xy=(x_pos, 0), xycoords=("axes fraction", "axes fraction"),  # end
                     xytext=(x_pos, 0.4), textcoords=("axes fraction", "axes fraction"),  # start
                     arrowprops=dict(facecolor='red', arrowstyle="->",
                                     connectionstyle="arc3"))
    x_pos_text = x_pos - 0.05
    t = axes[0].text(x_pos_text, 0.5, text_middle, ha="center", va="center", rotation=90,
                     size=15, transform=axes[0].transAxes)

    # t = axes[0].text(-0.1, line_top, text_top, ha="center", va="center", rotation=90,
    #                  size=15, transform=axes[0].transAxes)
    # t = axes[0].text(x_pos-0.20, line_bottom, text_bottom, ha="center", va="center", rotation=90,
    #                  size=15, )
    t = axes[0].text(x_pos_text, 0.8, text_top, ha="center", va="center", rotation=90,
                     size=15, transform=axes[0].transAxes)
    t = axes[0].text(x_pos_text, 0.2, text_bottom, ha="center", va="center", rotation=90,
                     size=15, transform=axes[0].transAxes)


def generata_fairness_arrows_plot(f_dict, f_dict_FC, f_nb_list=None, axes=None):
    if f_nb_list is None:
        f_nb_list = f_sensitive_list
    from datasets import x_labels
    # fairness_metrics = ["EQ","DP"]
    # fairness_metrics_names = ["Equal Opportunity","Demographic Parity"]
    from fair_measures import fairness_metrics, fairness_metrics_names
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
              "#17becf"]
    if axes is None:
        fig, axes = plt.subplots(1, len(fairness_metrics), figsize=(20, 5))
    cnt = 1
    for i, f in enumerate(dataset_fs):
        for j, _ in enumerate(f_nb_list[i]):
            df_ = f_dict[dataset_names[i]][j]
            for k, metric in enumerate(fairness_metrics):
                # f_dict_FC is model constant
                df_["{} (C)".format(metric)] = f_dict_FC[dataset_names[i]][j].loc["{} (M)".format(metric)]
                df_["{} DIFF".format(metric)] = np.abs(df_["{} (M)".format(metric)]) - np.abs(
                    df_["{} (O)".format(metric)])

                x = cnt
                f_o = df_["{} (O)".format(metric)]
                f_p = df_["{} (M)".format(metric)]
                f_c = df_["{} (C)".format(metric)]
                ax = axes[k]
                plot_dataset_unfairness(x, f_o, f_p, f_c, ax, color=colors[cnt - 1])
                ax.set_title(fairness_metrics_names[k])
                ax.set_xticks(range(1, len(x_labels) + 1))
                ax.set_xticklabels(x_labels, rotation=45)

            cnt += 1
    plt.tight_layout()
    # for i in range(len(fairness_metrics)):
    ax = axes[-1]
    ax.scatter([], [], marker="o", label=MODEL_ORIGINAL)
    ax.scatter([], [], marker="v", label=MODEL_MODIFIED)
    ax.scatter([], [], marker="x", label="$x_i$ constant", )
    ax.scatter([], [], marker="_", color="b", label="Original unfairness", )
    ax.scatter([], [], marker="_", color="r", label="Modified unfairness", )
    add_sidelines(axes)
    plt.legend(loc='upper right')
    return fig


def plot_unfairness_dots_across(f_dict_list):
    from fair_measures import fairness_metrics, fairness_metrics_names
    from datasets import x_labels
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
              "#17becf"]
    fig, axes = plt.subplots(1, len(fairness_metrics), figsize=(20, 5))
    for l, f_dict in enumerate(f_dict_list):
        cnt = 1
        for i, f in enumerate(dataset_fs):
            for j in range(len(f_dict[dataset_names[i]])):
                df_ = f_dict[dataset_names[i]][j]
                # print(type(df_))
                # display(df_)
                # print(df_.index)
                for k, metric in enumerate(fairness_metrics):
                    # f_dict_FC is model constant
                    f_diff = df_["{} DIFF".format(metric)]
                    x = cnt
                    ax = axes[k]
                    plot_unfairness_diff(x, f_diff, ax, color=colors[cnt - 1])
                    ax.set_title(fairness_metrics_names[k])
                    #                 ax.set_xticks(range(len(f_dict_list)))
                    #                 ax.set_xticklabels(range(len(f_dict_list)),rotation=45)
                    ax.set_xticks(range(1, len(x_labels) + 1))
                    ax.set_xticklabels(x_labels, rotation=45)
                    ax.axhline(y=0, linewidth=1, color='r')

                cnt += 1
    add_sidelines(axes, text_middle="Unfair (M)", text_top="More", text_bottom="Less")
    return fig


def save_fig(figure, fname, fig_dir=None, f_end=".png"):
    if fig_dir is not None:
        if not os.path.exists(fig_dir):
            os.mkdir(fig_dir)
    figure
    # plt.show(block=False)
    plt.savefig(os.path.join(fig_dir, fname + f_end), dpi=150)
    with open(os.path.join(fig_dir, fname + ".fig"), "wb") as f:
        dill.dump(figure, f)


default_save_dir = "/home/btd26/xai-research/xai-notebooks/Adversarial Explanations/temp_store"


def save_file(obj, path, default=False):
    if default:
        if path[0] =="/":
            path = path[1:]
        path = os.path.join(default_save_dir,path)
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
    with open(path, "wb") as f:
        dill.dump(obj, f)
        return True


def load_file(path, default=False):
    if default:
        if path[0] =="/":
            path = path[1:]
        path = os.path.join(default_save_dir,path)
    with open(path, "rb") as f:
        obj = dill.load(f)
        return obj


def f_dict2table(f_dict):
    from datasets import dataset_names
    f_dict_flat = []
    for i, _ in enumerate(dataset_names):
        for j, m in enumerate(f_dict[dataset_names[i]]):
            df_ = f_dict[dataset_names[i]][j]
            f_dict_flat.append(df_)
    f_dict_table = pd.concat(f_dict_flat, axis=1)
    return f_dict_table.transpose()


class Timer:
    '''
    # Start timer
  my_timer = Timer()

  # ... do something

  # Get time string:
  time_hhmmss = my_timer.get_time_hhmmss()
  print("Time elapsed: %s" % time_hhmmss )

  # ... use the timer again
  my_timer.restart()

  # ... do something

  # Get time:
  time_hhmmss = my_timer.get_time_hhmmss()

  # ... etc
    '''

    def __init__(self):
        self.start = time.time()

    def restart(self):
        self.start = time.time()

    def get_time_hhmmss(self):
        end = time.time()
        m, s = divmod(end - self.start, 60)
        h, m = divmod(m, 60)
        time_str = "%02d:%02d:%02d" % (h, m, s)
        return time_str
