# rc('text', usetex=True)

from collections import defaultdict

from matplotlib import pyplot as plt

from datasets import feature_name_dict, dataset_names, f_sensitive_list, sample_data
from datasets import get_data_list
from robust import *
from train_utils import *
from utils import save_fig


# -----------------------------------------------------------------------------
# PLOTS
# -----------------------------------------------------------------------------
# TODO move!
def plot_ranking_histograms(models, inputs, z_idx, ys=None, num_f=-1, title="",
                            attribution_methods=attribution_methods):
    '''

    :param models:
    :param inputs:
    :param z_idx:
    :param ys:
    :param f_num: # of features -1 = max z_idx
    :param title:
    :param attribution_methods: which explanation methods to use
    :return:
    '''
    #     models = [model_orig,model]
    models_str = ["Original", "Modified"]

    # attribution_methods = [saliency, gradient_x_input, integrated_grads, guided_backprop]
    # att_method_str = ["Gradients", "Gradient*Input", "Integrated Gradients", "Guided-Backprop"]
    n_rows = len(attribution_methods)
    f, axes = plt.subplots(n_rows, len(models), sharey=True, figsize=(10, 6))
    for model in models:
        model.activation = tf.keras.activations.linear
    for j, attribution in enumerate(attribution_methods):
        for i, m in enumerate(models):
            ax = axes[j, i]
            g_ = attribution(m, inputs, ys=ys)
            ax.hist(get_ranking(g_, z_idx), range=[0, num_f])
            ax.set_xlabel("rank")
            ax.set_ylabel("# samples")
            ax.set_title("{} {} Model".format(att_method_str[j], models_str[i]))
    y_title_pos = axes[0][0].get_position().get_points()[1][1] + (1 / n_rows) * 0.2
    f.suptitle(title, fontsize=16, y=y_title_pos)
    plt.tight_layout()
    for model in models:
        model.activation = tf.keras.activations.softmax

    return f


def generate_attribution_list(models, inputs, z_idx, ys=None,attribution_methods=attribution_methods):
    attribution_list = defaultdict(list)
    models_str = ["Original", "Modified"]
    for model in models:
        model.activation = tf.keras.activations.linear
    for j, attribution in enumerate(attribution_methods):
        for i, m in enumerate(models):
            g_ = attribution(m, inputs, ys=ys)
            attribution_list[models_str[i]].append(g_)
    for model in models:
        model.activation = tf.keras.activations.softmax
    return attribution_list

def plot_ranking_histograms_att(attribution_list, att_method_str, num_f, z_idx, models_str=["Original", "Modified"],
                                title="", ):
    n_rows = len(attribution_list[models_str[0]])
    f, axes = plt.subplots(n_rows, len(models_str), sharey=True, figsize=(10, 6))
    for j in range(n_rows):
        for i, model_name in enumerate(models_str):
            ax = axes[j, i]
            g_ = attribution_list[model_name][j]
            # num_f is not g_.shape since
            # TODO move get_ranking out of this fx!
            ax.hist(get_ranking(g_, z_idx), range=[0, num_f])
            ax.set_xlabel("rank")
            ax.set_ylabel("# samples")
            ax.set_title("{} {} Model".format(att_method_str[j], model_name))
    y_title_pos = axes[0][0].get_position().get_points()[1][1] + (1 / n_rows) * 0.2
    f.suptitle(title, fontsize=16, y=y_title_pos)
    plt.tight_layout()
    return f


def compute_rankings(model_orig, model, inputs, outputs=None, attribution_methods=attribution_methods):
    models_str = ["Original", "Modified"]
    models = [model_orig, model]
    for model in models:
        model.activation = tf.keras.activations.linear
    attribution_dict = defaultdict(list)
    for j, attribution in enumerate(attribution_methods):
        for i, m in enumerate(models):
            print(i, j)
            g_ = attribution(m, inputs, ys=outputs)
            attribution_dict[models_str[i]].append(g_)
    for model in models:
        model.activation = tf.keras.activations.softmax
    return attribution_dict


# -----------------------------------------------------------------------------
# EVAL
# -----------------------------------------------------------------------------


def evaluate_config(X_test, X_train, Y_test, Y_train, acc_list, e_loss, p_loss, model_orig, inputs, outputs, model,
                    z_idx, title, fig_dir=None, use_LIME=True):
    fig, (ax_1, ax_2) = plt.subplots(1, 2, figsize=(8, 4))
    ax = ax_1
    color = 'tab:orange'
    ax.plot(np.arange(len(e_loss)), e_loss, label="Explanation Loss", color=color)
    ax.set_ylabel("loss")
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:green'
    ax2.plot(np.arange(len(acc_list)), np.array(acc_list)[:, 1], label="Accuracy", color=color)
    ax.set_xlabel("epochs")
    ax2.set_ylabel("accuracy")
    # second ax
    ax = ax_2
    color = 'tab:orange'
    ax.plot(np.arange(len(e_loss)), e_loss, color=color)
    ax.set_ylabel("explanation loss")
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.plot(np.arange(len(p_loss)), p_loss, label="Categorical Cross-Entropy", color=color)
    ax2.set_ylabel("performance loss")
    ax.set_xlabel("epochs")
    fig.legend(loc='upper left')
    # ax2.legend()
    plt.tight_layout()
    if fig_dir is not None:
        save_fig(fig, "loss_cf", fig_dir)

    df, df_test = evaluate_config_2(X_test, X_train, Y_test, Y_train, fig_dir, inputs, model, model_orig, outputs,
                                    title, use_LIME, z_idx)

    return df, df_test


def evaluate_config_2(X_test, X_train, Y_test, Y_train, fig_dir, inputs, model, model_orig, outputs, title, use_LIME,
                      z_idx):
    # In[66]:
    loss_tr, acc_tr = model.evaluate(X_train, Y_train)
    loss_ts, acc_ts = model.evaluate(X_test, Y_test)
    loss_tr_o, acc_tr_o = model_orig.evaluate(X_train, Y_train)
    loss_ts_o, acc_ts_o = model_orig.evaluate(X_test, Y_test)
    # In[81]:
    round_err = 4
    loss_diff_tr = np.round(np.abs(loss_tr - loss_tr_o), round_err)
    loss_diff_ts = np.round(np.abs(loss_ts - loss_ts_o), round_err)
    print("Loss Diff: train - {} ---- test - {}".format(loss_diff_tr, loss_diff_ts))
    acc_diff_tr = np.round(np.abs(acc_tr - acc_tr_o), round_err)
    acc_diff_ts = np.round(np.abs(acc_ts - acc_ts_o), round_err)
    print("Acc Diff: train - {:.2f}% ---- test - {:.2f}%".format(acc_diff_tr * 100, acc_diff_ts * 100))
    explanation_loss_tr = compute_explanation_loss(inputs, outputs, model, z_idx)
    # if explanation_loss_tr != e_loss[-1]:
    #     print("Explanation loss not computed correctly! compute:{} e_loss:{}".format(
    #         float(explanation_loss_tr), float(e_loss[-1])))
    print("Explanation loss - {:.4f}".format(explanation_loss_tr))
    # 0 because suddenly non of the features are important!
    # In[68]:
    # In[74]:
    num_p, percent = analyze_mismatch(model_orig, model, inputs)
    print("Prediction Mismatch (Train):", num_p, str(round(percent, 3)) + "%")
    # Test Set Mismatch
    inputs_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    outputs_test = tf.convert_to_tensor(Y_test, dtype=tf.float32)
    num_p, percent = analyze_mismatch(model_orig, model, inputs_test)
    print("Prediction Mismatch (Test):", num_p, str(round(percent, 3)) + "%")
    # In[76]:
    from evaluate import plot_ranking_histograms
    from functools import partial
    # In[78]:
    models = [model_orig, model]
    att_methods = attribution_methods
    if not use_LIME:
        att_methods = att_methods[:-1]
    # TODO the attribution methods are evaluate 2x -> ones in plot_ranking & ones in analyze DF! there is no reason!
    plot_ranking_histograms_ = partial(plot_ranking_histograms, models=models, z_idx=z_idx, num_f=X_train.shape[1] - 1,
                                       attribution_methods=att_methods)
    fig_train = plot_ranking_histograms_(inputs=inputs, ys=outputs, title="{} Train".format(title))
    save_fig(fig_train, fname="ranking_histograms_train", fig_dir=fig_dir)
    # In[80]:
    # with Test
    fig_test = plot_ranking_histograms_(inputs=inputs_test, ys=outputs_test, title="{} Test".format(title))
    save_fig(fig_test, fname="ranking_histograms_test", fig_dir=fig_dir)
    # ### Explain ranking
    # In[84]:
    print("Computing Summary Table Train")
    df = get_summary_table(models=models, inputs=inputs, outputs=outputs, feature=z_idx,
                           attribution_methods=att_methods)
    diff_mean_tr, df_top_diff_tr, shifts_sum_tr, shifts_mean_tr = analyze_summary_table(df, title=title)
    print("Computing Summary Table Test")
    df_test = get_summary_table(models=models, inputs=inputs_test, outputs=outputs_test, feature=z_idx,
                                attribution_methods=att_methods)
    diff_mean_ts, df_top_diff_ts, shifts_sum_ts, shifts_mean_ts = analyze_summary_table(df_test, title)
    return df, df_test


def evaluate_attribution():
    raise NotImplementedError
    n_models = 6
    n_seeds = 10
    attribution_methods = defaultdict(list)
    for num_layers in range(n_models):
        print(10 * "#")
        print("num_layers {}".format(num_layers))
        print(10 * "#")
        f_dict_list = []
        f_dict_list_FC = []
        for seed in range(n_seeds):
            print("M#{}-seed#{}".format(num_layers, seed))
            models_list_p, models_nulify, model_orig_list = models_all[num_layers][seed]
            get_m_o = lambda i, j: model_orig_list[i]
            df_attack = generate_table_attack_success(models_list_p, get_original_model=get_m_o,
                                                      data_list=data_list, f_nb_list=None, use_train=True,
                                                      verbose=1)
            df = get_summary_table(models=models, inputs=inputs, outputs=outputs, feature=z_idx,
                                   attribution_methods=att_methods)
            #         diff_mean_tr, df_top_diff_tr, shifts_sum_tr, shifts_mean_tr = analyze_summary_table(df, title=title)
            print("Computing Summary Table Test")
            df_test = get_summary_table(models=models, inputs=inputs_test, outputs=outputs_test, feature=z_idx,
                                        attribution_methods=att_methods)
            attribution_methods[num_layers].append((df, df_test))


def summarise_attack_suspetibility(models, inputs, outputs, z_idx, use_LIME=False):
    att_methods = attribution_methods
    if not use_LIME:
        att_methods = att_methods[:-1]
    df = get_summary_table(models=models, inputs=inputs, outputs=outputs, feature=z_idx,
                           attribution_methods=att_methods)
    return df


def evaluate_fairness_model(model, model_name, X_train, Ytr, Ztr, verbose=0):
    '''

    :param model:
    :param X_train: we need this to make the predictions
    :param ytr: y_true has shape: {}, expected (n,)
    :param Ztr:
    :return:
    '''
    from fair_measures import fairness_IBM
    m_o = model
    y_o = m_o.predict(X_train)
    y_o = np.argmax(y_o, axis=1)
    # y_o = np.expand_dims(y_o, -1)
    y_pred = y_o

    ytr = np.argmax(Ytr, axis=1)
    results = fairness_IBM(y_pred, Ztr, ytr, verbose)
    # results = analyze_fairness(y_pred, Ztr, ytr, verbose)
    results.columns = [x + " " + model_name for x in results.columns]
    return results


def evaluate_fairness(model_p, model_orig, X_train, ytr, Ztr, verbose=False):
    if verbose:
        print("#################################################################")
        print("-------------------------FAIRNESS--------------------------------")
        print("-------------------------TRAIN-----------------------------------")
    m_p = model_p
    m_o = model_orig
    models = [m_p, m_o]
    model_names = ["(M)", "(O)"]
    df_list = []
    for i, model in enumerate(models):
        df_list.append(evaluate_fairness_model(model, model_names[i], X_train, ytr, Ztr, verbose=0))
    results = pd.concat(df_list, axis=1)
    from fair_measures import fairness_metrics
    for i, fairness_metric in enumerate(fairness_metrics):
        gap = results["{} {}".format(fairness_metric, model_names[0])] - results[
            "{} {}".format(fairness_metric, model_names[1])]
        results["{} Diff".format(fairness_metric)] = gap
    return results


def analyze_fairness(y_pred, Ztr, ytr, verbose):
    raise NotImplemented
    results = {}
    from fair_measures import fairness_metrics, fairness_functions
    for i, fairness_metric in enumerate(fairness_metrics):
        cls_0, cls_1, gap = fairness_functions[i](y_pred, Ztr, ytr)
        results["{} val".format(fairness_metric)] = [[cls_0, cls_1]]
        results["{}".format(fairness_metric)] = gap
        if verbose:
            print(fairness_metric)

    if verbose:
        print("Equal Opportunity ")
        print(cls_0, cls_1, gap)
    results = pd.DataFrame(results)
    return results


# -----------------------------------------------------------------------------
# ANALYSIS UTILS
# -----------------------------------------------------------------------------
def overview_table(result_list, decimals_cols=1, decimals_loss=3):
    '''

    :param result_list: [df_results_ga,df_results_gg,df_results_ba,df_results_ag,df_results_ar]
    :return:
    '''
    df_results_present = overview_table_(result_list)
    num_cols = len(df_results_present.columns)  # %%
    decimals = np.ones(num_cols, dtype=np.int) + decimals_cols
    decimals[list(df_results_present.columns).index("Loss diff.")] = decimals_loss  # %%
    decimals[list(df_results_present.columns).index("Expl. loss")] = 4
    decimals = pd.Series(decimals, index=df_results_present.columns)  # %%
    df_results_present = df_results_present.round(decimals)
    index = pd.MultiIndex.from_tuples(zip(list(df_results_present["Dataset"]), list(df_results_present["Feature"])),
                                      names=['Dataset', 'Feature'])
    df_results_present = df_results_present.drop(columns=['Dataset', 'Feature'])
    return df_results_present


def overview_fairness(fair_list):
    df_fair_all = overview_table_(fair_list)
    num_cols = []
    for c in df_fair_all.columns:
        if df_fair_all[c].dtype is not np.dtype('O'):
            num_cols.append(c)
    df_fair_all.loc[:, num_cols] *= 100
    df_fair_all.columns = [c.replace("GAP", "Diff") for c in df_fair_all.columns]
    return df_fair_all


def overview_table_(result_list):
    df_results_all = pd.concat(result_list, axis=1)  # %%
    df_results_present = df_results_all.transpose().infer_objects()  # %%
    return df_results_present


def overall_fairness(data_set_name, feature_name, model_p, model_orig, X_train, ytr, Ztr_b, verbose=False):
    fair_results = evaluate_fairness(model_p, model_orig, X_train, ytr, Ztr_b, verbose=verbose)
    fair_results.insert(0, "Dataset", data_set_name, True)
    fair_results.insert(1, "Feature", feature_name, True)
    # results = pd.Series(fair_results)
    # results.index = fair_results.columns
    # TODO change to dataframe not to loose types!
    return pd.Series(fair_results.T.iloc[:, 0])


# def process_fairness_(data_set_name, feature_name, fair_results):
#     demo_ag = fair_results["Demo"]
#     df_fair_ = pd.Series(
#         [data_set_name, feature_name, fair_results["EQ (O)"], fair_results["EQ (P)"], fair_results["GAP"]])
#     df_fair = pd.concat([df_fair_, demo_ag])
#     df_fair.index = ["Dataset", "Feature", "EQ (O)", "EQ (P)", "EQ Diff", "DP (O)", "DP (P)", "DP Diff"]
#     return df_fair
#
#
# def process_fairness_2(data_set_name, feature_name, fair_results):
#     demo_ag = fair_results["Demo"]
#     df_fair_ = pd.Series(
#         [data_set_name, feature_name, fair_results["EQ val (O)"], fair_results["EQ val (P)"],
#          fair_results["EQ (O)"], fair_results["EQ (P)"], fair_results["GAP"]])
#     df_fair = pd.concat([df_fair_, demo_ag])
#     df_fair.index = ["Dataset", "Feature", "EQ val (O)", "EQ val (P)", "EQ (O)", "EQ (P)", "EQ Diff", "DP (O)",
#                      "DP (P)", "DP Diff"]
#     return df_fair


# which results?
# Table 1
def process_results(data_set_name, feature_name, X, Y, model, model_orig, z_idx, normalise_eloss=True, use_LIME=True):
    models = [model_orig, model]
    inputs = tf.convert_to_tensor(X, dtype=tf.float32)
    outputs = tf.convert_to_tensor(Y, dtype=tf.float32)
    att_methods = attribution_methods
    if not use_LIME:
        att_methods = att_methods[:-1]
    df = get_summary_table(models=models, inputs=inputs, outputs=outputs, feature=z_idx,
                           attribution_methods=att_methods)
    diff_mean_tr, top_o_diff_percent, shifts_sum_tr, shifts_mean_tr = analyze_summary_table(df, title="")
    explanation_loss_tr = compute_explanation_loss(inputs, outputs, model, z_idx, normalise=normalise_eloss)

    # loss diff
    # In[66]:
    loss_tr, acc_tr = model.evaluate(X, Y)

    loss_tr_o, acc_tr_o = model_orig.evaluate(X, Y)
    # In[81]:
    round_err = 4
    loss_diff_tr = np.round(np.abs(loss_tr - loss_tr_o), round_err)
    acc_diff_tr = np.round(np.abs(acc_tr - acc_tr_o), round_err) * 100
    num_p, percent = analyze_mismatch(model_orig, model, inputs)
    mismatch = round(percent, 3)

    columns = ["Dataset", "Feature", "Expl. loss", "Avg. #shifts", "Mean diff.", "Top_O (P) (%)", "Loss diff.",
               "Acc. diff.",
               "Mismatch (%)"]

    df = pd.Series([data_set_name, feature_name, explanation_loss_tr.numpy(),
                    shifts_mean_tr, diff_mean_tr, round(top_o_diff_percent, 2), loss_diff_tr, acc_diff_tr, mismatch],
                   index=columns)
    return df


def analyze_summary_table(df, title=None):
    # In[91]:
    if title is not None:
        print(title)
    diff__mean = df["Mean Diff"].mean()
    print("Mean diff", diff__mean)
    # In[92]:
    top_o_diff_percent = df["#Top_O (P)"].sum() / df["#Top_O (O)"].sum() * 100
    print("Top (O) diff", top_o_diff_percent)
    # In[93]:
    shifts__sum = df["# shifts"].sum()
    print("Total # shifts", shifts__sum)
    shifts__mean = df["# shifts"].mean()
    print("Avg # shifts", shifts__mean)
    return diff__mean, top_o_diff_percent, shifts__sum, shifts__mean


def analyze_summary_table_legacy(inputs, models, outputs, z_idx, title=None):
    from evaluate import get_summary_table
    from functools import partial
    # In[86]:
    get_summary_table_ = partial(get_summary_table, models=models)
    # In[90]:
    df = get_summary_table_(inputs=inputs, outputs=outputs, feature=z_idx)
    from IPython.display import display
    display(df)
    # In[91]:
    if title is not None and title is not "":
        print(title)
    diff__mean = df["Mean Diff"].mean()
    print("Mean diff", diff__mean)
    # In[92]:
    df_top_diff = df["#Top_O (O)"] - df["#Top_O (P)"]
    print("Top (O) diff", df_top_diff.sum())
    # In[93]:
    shifts__sum = df["# shifts"].sum()
    print("Total # shifts", shifts__sum)
    shifts__mean = df["# shifts"].mean()
    print("Avg # shifts", shifts__mean)
    return df


def analyze_mismatch(model_orig, model, inputs, verify_layers=True):
    if verify_layers:
        model_orig.layers[-1].activation = tf.keras.activations.softmax
        model.layers[-1].activation = tf.keras.activations.softmax
    mismatch = mismatch_ids(inputs, model, model_orig)
    if verify_layers:
        model_orig.layers[-1].activation = tf.keras.activations.softmax
        model.layers[-1].activation = tf.keras.activations.softmax
    return len(mismatch), (len(mismatch) / int(inputs.shape[0])) * 100


def mismatch_ids(inputs, model, model_orig):
    y_ = model_orig(inputs)
    y_o = np.argmax(y_, axis=1)
    y_p = np.argmax(model(inputs), axis=1)
    mismatch = np.where(y_o != y_p)[0]
    #     print(len(mismatch),str(len(mismatch)/int(y_p.shape[0])*100)+"%")
    return mismatch


def get_summary_table_att(attribution_list, att_method_str, models_str, feature,
                          attribution_methods=attribution_methods):
    results_y_pred = []
    n_rows = len(attribution_list[models_str[0]])
    for j in range(n_rows):
        g_o = attribution_list[models_str[0]][j]
        g_p = attribution_list[models_str[1]][j]
        r_o, r_p = get_ranking(g_o, feature=feature), get_ranking(g_p, feature=feature)
        m_o, m_p = get_mode_(r_o, feature=feature), get_mode_(r_p, feature=feature)
        min_o, max_o, mean_o, skew_o, kurtosis_o = get_stats(r_o)
        min_p, max_p, mean_p, skew_p, kurtosis_p = get_stats(r_p)
        results = [m_o, m_p, m_p - m_o, mean_o, mean_p, mean_p - mean_o, min_o, min_p]
        column_strs = []

        for k in [r_o.min() + 1, 5, 1]:
            n_k_o, n_k_p = num_points_top_k(r_o, k), num_points_top_k(r_p, k)
            results.extend([n_k_o, n_k_p])
            # TODO Unnecessarily repeated over attribution methods
            column_strs.append("Top-{} (O)".format(k))
            column_strs.append("Top-{} (P)".format(k))
        column_strs[0] = "#Top_O (O)"
        column_strs[1] = "#Top_O (P)"
        results_y_pred.append(results)

    columns = ["Mode (O)", "Mode (P)", "# shifts", "Mean (O)", "Mean (P)", "Mean Diff", "Top Rank(O)", "Top Rank(P)"]
    columns.extend(column_strs)
    # compute len & if <0, trim
    len_att_str = len(attribution_methods) - len(att_method_str)
    if len_att_str < 0:
        df_index = att_method_str[:len_att_str]
    else:
        df_index = att_method_str
    df = pd.DataFrame(results_y_pred, columns=columns, index=df_index)
    return df


def get_summary_table(models, inputs, outputs, feature, attribution_methods=attribution_methods):
    results_y_pred = []
    mean_diffs = []
    model_orig, model = models
    for f in attribution_methods:
        g_p = f(model, inputs, ys=outputs)
        g_o = f(model_orig, inputs, ys=outputs)
        r_o, r_p = get_ranking(g_o, feature=feature), get_ranking(g_p, feature=feature)
        m_o, m_p = get_mode_(r_o, feature=feature), get_mode_(r_p, feature=feature)
        min_o, max_o, mean_o, skew_o, kurtosis_o = get_stats(r_o)
        min_p, max_p, mean_p, skew_p, kurtosis_p = get_stats(r_p)
        results = [m_o, m_p, m_p - m_o, mean_o, mean_p, mean_p - mean_o, min_o, min_p]
        column_strs = []

        for k in [r_o.min() + 1, 5, 1]:
            n_k_o, n_k_p = num_points_top_k(r_o, k), num_points_top_k(r_p, k)
            results.extend([n_k_o, n_k_p])
            # TODO Unnecessarily repeated over attribution methods
            column_strs.append("Top-{} (O)".format(k))
            column_strs.append("Top-{} (P)".format(k))
        column_strs[0] = "#Top_O (O)"
        column_strs[1] = "#Top_O (P)"
        results_y_pred.append(results)

    columns = ["Mode (O)", "Mode (P)", "# shifts", "Mean (O)", "Mean (P)", "Mean Diff", "Top Rank(O)", "Top Rank(P)"]
    columns.extend(column_strs)
    # compute len & if <0, trim
    len_att_str = len(attribution_methods) - len(att_method_str)
    if len_att_str < 0:
        df_index = att_method_str[:len_att_str]
    else:
        df_index = att_method_str
    df = pd.DataFrame(results_y_pred, columns=columns, index=df_index)
    return df


def binarize_sen(Ztr, target_id=0, assign_val=1):
    white_val = np.unique(Ztr)[target_id]
    white_idx = np.where(Ztr == white_val)[0]
    Ztr_bin = np.array(Ztr)
    Ztr_bin[white_idx] = assign_val
    not_white_idx = np.where(Ztr != white_val)[0]
    # inverse
    assign_val = (assign_val - 1) * -1
    Ztr_bin[not_white_idx] = assign_val
    return Ztr_bin


def binarize_sen_dict(Ztr, targets=None, assign_vals=[0, 1]):
    '''

    :param Ztr:
    :param targets: [[0], [1]]
    :param assign_vals: [0, 1]
    :return:
    '''
    Ztr_bin = np.copy(Ztr)
    if targets is None:
        targets = list(zip(np.unique(Ztr)))
    elif callable(targets):
        mask = targets(Ztr)
        Ztr_bin[np.where(mask)[0]] = assign_vals[1]
        Ztr_bin[np.where(~mask)[0]] = assign_vals[0]
        return Ztr_bin
    n_categories = len(targets)
    # check if the assign values are less then the given targets
    if len(assign_vals) < n_categories:
        assign_vals = range(n_categories)

    # iterate over the groups and for all the values in the group assign the corresponding value

    for k in range(n_categories):

        for i, v in enumerate(targets[k]):
            Ztr_bin = np.where(Ztr == v, assign_vals[k], Ztr_bin)
    return Ztr_bin


def plot_shift_accross_features(mi, models, inputs, outputs, att_methods, annotate=False):
    avg_mode_list = []
    avg_mean_list = []
    for f_idx in reversed(np.argsort(mi[:])):
        print(f_idx)
        df = get_summary_table(models=models, inputs=inputs, outputs=outputs, feature=f_idx,
                               attribution_methods=att_methods)
        avg_mode_list.append(df["# shifts"].mean())
        avg_mean_list.append(df["Mean Diff"].mean())
    #     display(df)
    fig, ax = plt.subplots()
    sort_arg = list(reversed(np.argsort(mi[:])))
    x = np.arange(len(avg_mode_list))
    y = avg_mode_list
    ax.plot(x, y)
    if annotate:
        for i, j in zip(x, y):
            ax.annotate(str(j), xy=(i, j + 0.5))
    ax2 = ax.twinx()
    color = "tab:orange"
    ax2.plot(np.arange(len(avg_mode_list)), avg_mean_list, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax.set_xticks(np.arange(mi.shape[0]))
    ax.set_xticklabels(sort_arg)
    ax.set_xlabel("feature")
    ax.set_ylabel("Average Mode Diff")
    ax2.set_ylabel("Average Mean Diff")
    return fig


def feature_importance_remove(X_train, Y_train, X_test, Y_test):
    from train_utils import build_model, get_dataset, fit_model
    models = []
    n_features = X_train.shape[-1]
    feature_nb_train = []
    feature_nb_test = []

    for i in range(n_features):
        print(i)
        ix = np.delete(np.arange(n_features), i)
        model = build_model(input_shape=(n_features - 1,), num_layers=3)
        batch_size = 30000
        adult_train = get_dataset(X_train[:, ix], Y_train, batch_size=batch_size)
        adult_test = get_dataset(X_test[:, ix], Y_test)
        EPOCHS = 1000
        history = fit_model(model, X_train[:, ix], Y_train, EPOCHS, batch_size=batch_size, verbose=0)
        #     file_name="../temp_store/models/adult-{}.h5".format(i)
        #     model.load_weights(file_name)
        models.append(model)
        loss, acc = model.evaluate(adult_train)
        feature_nb_train.append(acc)
        loss, acc = model.evaluate(adult_test)
        feature_nb_test.append(acc)
    return models, feature_nb_train, feature_nb_test


def feature_importance_nulify(X_train, Y_train, X_test, Y_test, feature_idx=None, seed=49, num_layers=3):
    from train_utils import build_model, get_dataset, fit_model
    from datasets import nulify_feature
    models = []
    n_features = X_train.shape[-1]
    if feature_idx is None:
        feature_idx = range(n_features)
    feature_nb_train = []
    feature_nb_test = []

    for i in feature_idx:
        print(i)
        #     ix = np.delete(np.arange(n_features), i)
        model = build_model(input_shape=(n_features,), num_layers=num_layers, seed=seed)
        batch_size = 10000
        x, y = nulify_feature(X_train, Y_train, i)
        adult_train = get_dataset(x, y, batch_size=batch_size)

        EPOCHS = 1000
        history = fit_model(model, x, y, EPOCHS=EPOCHS, batch_size=batch_size, verbose=0)
        x, y = nulify_feature(X_test, Y_test, i)
        adult_test = get_dataset(x, y)
        #     file_name="../temp_store/models/adult-{}.h5".format(i)
        #     model.load_weights(file_name)
        models.append(model)
        loss, acc = model.evaluate(adult_train)
        feature_nb_train.append(acc)
        loss, acc = model.evaluate(adult_test)
        feature_nb_test.append(acc)
    return models, feature_nb_train, feature_nb_test


def evaluate_pertrubed_models(X_train, Y_train, X_test, Y_test, e_alpha=0.25, feature_set=None, attack=no_attack,
                              train_robust=False, model_orig=None, EPOCHS=1000, R_EPOCHS=100, batch_size=30000,
                              seed=49, num_layers=3):
    n_features = X_train.shape[-1]

    adult_train = get_dataset(X_train, Y_train, batch_size=batch_size)
    adult_test = get_dataset(X_test, Y_test)
    optimizer = tf.keras.optimizers.Adam(lr=0.01)
    model_full = build_model(input_shape=(n_features), num_layers=num_layers, optimizer=optimizer, seed=seed)

    if train_robust:
        # TODO
        optimizer = tf.keras.optimizers.Adam(lr=0.01)
        for t in range(R_EPOCHS):
            adv_err, adv_loss = epoch_adversarial(adult_train, model_full, attack, epsilon=0.25, alpha=0.08,
                                                  num_iter=30, optimizer=optimizer)
    else:
        if model_orig is None:
            print("WARNING")
            print("TRAINING MODEL FROM SCRATCH")
            history = fit_model(model_full, X_train, Y_train, EPOCHS, batch_size=batch_size, verbose=0)
        else:
            model_full.set_weights(model_orig.get_weights())
    models_p = []
    adv_lis = []
    val_names = ("train_err", "test_err", "adv_err", "adv_err_fgsm", "e_loss", "e_loss_train")
    print(*("{}".format(i) for i in val_names), sep="\t")
    if feature_set is None:
        feature_set = range(n_features)
    for i in feature_set:
        print(i)
        z_idx = i
        model_explain = clone_model(model_full)
        optimizer = tf.keras.optimizers.Adam(lr=0.01)
        for t in range(50):
            train_err, train_loss = epoch_explanation(adult_train, model_explain, attack,
                                                      sensitive_feature_id=z_idx, e_alpha=e_alpha,
                                                      epsilon=0.25, alpha=0.08, num_iter=30, optimizer=optimizer)
        models_p.append(model_explain)
        adv_err, adv_err_f, e_loss, e_loss_train, test_err = epoch_eval(adult_train, adult_test, model_explain, z_idx)
        r = (train_err, test_err, adv_err, adv_err_f, e_loss, e_loss_train)
        adv_lis.append(r)
        print(*("{:.6f}".format(i) for i in r), sep="\t")

    return models_p, adv_lis


def visualise_feature_importance(feature_nb_train):
    n_features = len(feature_nb_train)
    fig, ax = plt.subplots()
    f_nb = feature_nb_train - np.min(feature_nb_train)
    f_nb *= 100
    ax.bar(np.arange(n_features), f_nb)
    ax.set_xlabel("feature")
    ax.set_ylabel("% acc_f - acc_min")
    print(np.argsort(feature_nb_train)[::-1])
    return fig


def evaluate_feature_importance_over_datasets(dataset_fs=None):
    from evaluate import feature_importance_nulify
    from datasets import prep_data
    models_list = []
    feature_nb_train_list = []
    feature_nb_test_list = []
    if dataset_fs is None:
        from datasets import dataset_fs
    for f in dataset_fs:
        Xtr, Xts, ytr, yts, Ztr, Zts = f(0, remove_z=False)
        X_test, X_train, Y_test, Y_train = prep_data(Xtr, Xts, ytr, yts)
        models, feature_nb_train, feature_nb_test = feature_importance_nulify(X_train, Y_train, X_test, Y_test)
        models_list.append(models)
        feature_nb_train_list.append(feature_nb_train)
        feature_nb_test_list.append(feature_nb_test)
    return models_list, feature_nb_train_list, feature_nb_test_list


# TODO optimise such that it accepts a list of models and a list of get_fxs!
# TODO also optimise so that it can compute for one list
def evaluate_fairness_across(models_list_p_r, get_model, data_list=None, f_nb_list=None, use_train=True, verbose=1):
    '''

    :param models_list_p_r:
    :param get_model: lambda i,j:models_nulify[i][j] or lambda i,j:model_orig_list[i]
    :return:
    '''
    from datasets import binarise_dict, get_data_list
    if f_nb_list is None:
        f_nb_list = f_sensitive_list
    if data_list is None:
        data_list = get_data_list()
    fairnes_dict = defaultdict(dict)
    eq_diffs_FC = []
    dp_diffs_FC = []
    for i, data in enumerate(data_list):

        X_test, X_train, Y_test, Y_train = data
        if use_train:
            X = X_train
            Y = Y_train
        else:
            X = X_test
            Y = Y_test
        print("models #{}".format(len((models_list_p_r[i]))))
        for j, m in enumerate(models_list_p_r[i]):
            f_name = feature_name_dict[dataset_names[i]][f_nb_list[i][j]]
            f_id = f_nb_list[i][j]

            Ztr = X[:, f_id]
            Ztr_b = binarize_sen_dict(Ztr, **binarise_dict[dataset_names[i]][f_name])

            model_p = m
            model_orig = get_model(i, j)
            data_set_name = dataset_names[i]
            feature_name = feature_name_dict[data_set_name][f_nb_list[i][j]]
            df_fair_gg = overall_fairness(data_set_name, feature_name, model_p, model_orig, X, Y, Ztr_b)
            if verbose > 0:
                from IPython.display import display
                display(df_fair_gg)
            fairnes_dict[data_set_name][j] = df_fair_gg
            eq_diffs_FC.append(df_fair_gg["EQ Diff"])
            dp_diffs_FC.append(df_fair_gg["DP Diff"])
    return fairnes_dict, eq_diffs_FC, dp_diffs_FC


def generate_table_attack_success(model_lists, get_original_model, data_list=None, f_nb_list=None, use_train=True,
                                  verbose=1):
    '''

    :param model_lists:
    :param get_original_model:
    :param data_list:
    :param f_nb_list:
    :param use_train:
    :param verbose:
    :return: evalautes, explantion loss, accuracy & mismatch %
    '''
    if f_nb_list is None:
        f_nb_list = f_sensitive_list
    if data_list is None:
        data_list = get_data_list()

    metrics = []
    table_list = []
    for i, data in enumerate(data_list):

        X_test, X_train, Y_test, Y_train = data
        if use_train:
            inputs = tf.convert_to_tensor(X_train, dtype=tf.float32)
            outputs = tf.convert_to_tensor(Y_train, dtype=tf.float32)
        else:
            inputs = tf.convert_to_tensor(X_test, dtype=tf.float32)
            outputs = tf.convert_to_tensor(Y_test, dtype=tf.float32)
        print("models #{}".format(len((model_lists[i]))))
        for j, model in enumerate(model_lists[i]):
            f_name = feature_name_dict[dataset_names[i]][f_nb_list[i][j]]
            model_orig = get_original_model(i, j)
            data_set_name = dataset_names[i]
            feature_name = feature_name_dict[data_set_name][f_nb_list[i][j]]
            sensitive_feature_id = f_nb_list[i][j]
            e_loss = compute_explanation_loss(inputs, outputs, model, sensitive_feature_id,
                                              norm=1,
                                              normalise=True,
                                              loss=None,
                                              )

            acc_m = compute_metric(model, inputs, outputs)
            acc_o = compute_metric(model_orig, inputs, outputs)
            acc_change = acc_m - acc_o
            acc_change *= 100
            print(acc_m, acc_o, acc_change)
            # print(model.evaluate(inputs,outputs))
            # print(model_orig.evaluate(inputs, outputs))
            num_p, percent = analyze_mismatch(model_orig, model, inputs)
            s = pd.Series()
            s["Dataset"] = data_set_name
            s["Feature"] = feature_name
            s["E Loss"] = e_loss.numpy()
            s["Acc $\Delta$"] = acc_change
            s["Mismatch (%)"] = percent
            # display(s)
            table_list.append(s)
    df = pd.concat(table_list, axis=1).T
    # df.set_index(["Dataset", "Feature"],inplace=True)
    return df


def generate_table_low_attribution(model_lists, get_original_model, data_list=None, f_nb_list=None,
                                   use_train=True,
                                   sample=None,
                                   verbose=1):
    if f_nb_list is None:
        f_nb_list = f_sensitive_list
    if data_list is None:
        data_list = get_data_list()

    metrics = []
    table_dict = defaultdict(list)
    for i, data in enumerate(data_list):

        X, Y = sample_data(data, sample, use_train)
        inputs = tf.convert_to_tensor(X, dtype=tf.float32)
        outputs = tf.convert_to_tensor(Y, dtype=tf.float32)

        print("models #{}".format(len((model_lists[i]))))
        for j, model in enumerate(model_lists[i]):
            f_name = feature_name_dict[dataset_names[i]][f_nb_list[i][j]]
            model_orig = get_original_model(i, j)
            data_set_name = dataset_names[i]
            feature_name = feature_name_dict[data_set_name][f_nb_list[i][j]]
            sensitive_feature_id = f_nb_list[i][j]
            models = [model_orig, model]
            df = summarise_attack_suspetibility(models, inputs, outputs, sensitive_feature_id, use_LIME=True)
            table_dict[data_set_name].append(df)
    # df.set_index(["Dataset", "Feature"],inplace=True)
    return table_dict


generate_table_methods = generate_table_low_attribution


def compute_metric(model, X, Y, metric=tf.keras.metrics.CategoricalAccuracy()):
    y_pred = model(X)
    metric.reset_states()
    metric.update_state(Y, y_pred)
    return metric.result().numpy()


def evaluate_metrics_across(model_lists, f_nb_list,
                            model_lists_names,
                            data_list=None,
                            use_train=True,
                            do_flip=False,
                            constant_test=False,
                            ):
    '''
    computes the loss & accuracy!
    :param model_lists:
    :param f_nb_list:
    :param model_lists_names: ["Modified", "Modified (R)", "x_i Constant", "Original", ]
    :return:
    '''

    from partial_dependence import flip_X
    from utils import MODEL_CONSTANT
    if f_nb_list is None:
        f_nb_list = f_sensitive_list
    if data_list is None:
        data_list = get_data_list()

    metric_list_dict = defaultdict(list)
    # model_lists = [models_list_p, models_list_p_r, models_nulify, model_orig_list, ]

    m_loss = tf.keras.metrics.CategoricalCrossentropy()
    m_acc = tf.keras.metrics.CategoricalAccuracy()
    metrics = [m_acc, m_loss]
    for i, data in enumerate(data_list):

        X_test, X_train, Y_test, Y_train = data
        if use_train:
            X = X_train
            Y = Y_train
        else:
            X = X_test
            Y = Y_test

        inputs = tf.convert_to_tensor(X, dtype=tf.float32)
        outputs = tf.convert_to_tensor(Y, dtype=tf.float32)
        # for j, _ in enumerate(model_lists[0][i]):
        for j, _ in enumerate(f_nb_list[i]):
            f_id = f_nb_list[i][j]
            f_name = feature_name_dict[dataset_names[i]][f_id]
            if do_flip:
                try:
                    X = flip_X(X, f_id)
                    inputs = tf.convert_to_tensor(X, dtype=tf.float32)
                except Exception as e:
                    print(str(e))
                    print("{}-{}".format(dataset_names[i], f_name))
                    continue
            if constant_test:
                # c.f. nulify_feature()
                X_constant = X.copy()
                X_constant[:, f_id] = 0
                inputs_constant = tf.convert_to_tensor(X_constant, dtype=tf.float32)
            # iterate over Modified, Nulified, Original, etc.
            for k, model_list in enumerate(model_lists):
                model_list_name = model_lists_names[k]

                # distinguish b/t a list of model which vary by feature and
                # the original model (does not vary by feature)
                if type(model_list[i]) is list:
                    m = model_list[i][j]
                else:
                    m = model_list[i]
                # manual evaluation because model.evaluate does not work for adverarial explanation models
                if constant_test and model_list_name == MODEL_CONSTANT:
                    y_pred = m(inputs_constant)
                    print("Constant Inputs")
                else:
                    y_pred = m(inputs)
                results = []
                for m in metrics:
                    m.update_state(outputs, y_pred)
                    results.append(m.result().numpy())
                    m.reset_states()
                metric_list_dict[model_list_name].append(results)
    acc_list_dict = {k: list(zip(*v))[0] for k, v in metric_list_dict.items()}
    loss_list_dict = {k: list(zip(*v))[1] for k, v in metric_list_dict.items()}
    return acc_list_dict, loss_list_dict


def model_evaluate(model, X, Y,
                   metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.CategoricalCrossentropy()]):
    y_pred = model(X)
    results = []
    for m in metrics:
        m.update_state(Y, y_pred)
        results.append(m.result().numpy())
        m.reset_states()
    return results
