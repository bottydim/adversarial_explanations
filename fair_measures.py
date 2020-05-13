from collections import defaultdict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def fairness_IBM(y_pred, Ztr, ytr, verbose=0):
    from aif360.datasets import BinaryLabelDataset
    from aif360.metrics import ClassificationMetric

    assert np.array_equal(np.unique(Ztr), np.array([0, 1])), "Z must contain either 0 or 1"
    # if len(ytr.shape) == 1:
    # ytr = np.expand_dims(ytr, -1)

    Ztr = np.squeeze(Ztr)
    if verbose:
        print(ytr.shape)
        print(Ztr.shape)
    unprivileged_groups = [{"zs": [0]}]
    privileged_groups = [{"zs": [1]}]
    metric_arrs = defaultdict(list)
    dict_ = {"y_true": ytr,
             "zs": Ztr}
    df = pd.DataFrame(dict_)
    dataset = BinaryLabelDataset(df=df, label_names=["y_true"], protected_attribute_names=["zs"],
                                 unprivileged_protected_attributes=[[0]],
                                 privileged_protected_attributes=[[1]]
                                 )

    dataset_pred = dataset.copy()
    dataset_pred.labels = y_pred
    metric = ClassificationMetric(
        dataset, dataset_pred,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups)

    # metric_arrs['bal_acc'].append((metric.true_positive_rate()
    #                              + metric.true_negative_rate()) / 2)
    metric_arrs["EA"].append(metric.accuracy(privileged=False) - metric.accuracy(privileged=True))
    # ASSUMING ALL OTHER METRICS RETURN U - P
    metric_arrs['EO'].append(metric.average_odds_difference())
    # The ideal value of this metric is 1.0
    # A value < 1 implies higher benefit for the privileged group
    # and a value >1 implies a higher
    metric_arrs['DI'].append(metric.disparate_impact() - 1)
    metric_arrs['DP'].append(metric.statistical_parity_difference())
    metric_arrs['EQ'].append(metric.equal_opportunity_difference())
    metric_arrs['TH'].append(metric.between_group_theil_index() * 10)
    results = pd.DataFrame(metric_arrs)
    return results


def demongraphic_parity_legacy(Ztr, ytr, y_p, y_o):
    assert len(ytr.shape) == 1
    assert len(y_p.shape) == 2
    assert len(y_o.shape) == 2
    assert len(Ztr.shape) == 2
    # demographic parity - correlation of sensitive property to outcome!
    df_demographic_parity = pd.DataFrame(np.hstack((Ztr, np.expand_dims(ytr, -1), y_p, y_o)),
                                         columns=["Sensitive Property", "Outcome", "Predicted (P)", "Predicted (O)"])
    return df_demographic_parity.corr()


def demongraphic_parity(Ztr, y_pred, ytr, sen_val_id):
    assert len(np.unique(y_pred)) < 3, "y_true is not binary"
    assert len(np.unique(ytr)) < 3, "y_pred is not binary"
    sen_vals = np.unique(Ztr)
    sen_val = sen_vals[sen_val_id]
    sen_idx = np.where(Ztr == sen_val)[0]
    y_true = ytr[sen_idx]
    y_pred = np.squeeze(y_pred)[sen_idx]
    total = y_true.shape[0]

    n_positive = len(np.where(y_pred == 1)[0])

    return n_positive / total, (total - n_positive) / total


def fairness_equal_accuracy(Ztr, y_pred, ytr, sen_val_id):
    sen_vals = np.unique(Ztr)
    sen_val = sen_vals[sen_val_id]
    sen_idx = np.where(Ztr == sen_val)[0]
    y_true = ytr[sen_idx]
    y_pred = np.squeeze(y_pred)[sen_idx]
    total = y_true.shape[0]
    from sklearn.metrics import confusion_matrix
    if len(y_true) == 0 or len(y_pred) == 0:
        print("0 samples")
        return None
    elif len(np.unique(y_pred)) < 2:
        tp = confusion_matrix(y_true, y_pred).ravel()[0]
        print("Same predictions")
        tn = 0
    else:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return (tp + tn) / total, None


def get_tpr(y_pred, y_true):
    from sklearn.metrics import confusion_matrix
    if len(y_true) == 0 or len(y_pred) == 0:
        print("0 samples")
    elif len(np.unique(y_pred)) < 2:
        tp = confusion_matrix(y_true, y_pred).ravel()[0]
        tp /= len(y_pred)
        print("Same predictions")
        return 1 - tp, tp
    else:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        #     print(tn, fp, fn, tp)
        # tpr = True Positive / Positive = tp + False Negative - Positive
        tpr = tp / (tp + fn)
        # tnr
        if (tn + fp) > 0:
            tnr = tn / (tn + fp)
        else:
            tnr = None
        return tnr, tpr


def fairness_equal_opportunity(Ztr, y_pred, ytr, sen_val_id):
    '''

    :param Ztr: must contail all different values of Ztr,
    :param y_pred:
    :param ytr:
    :param sen_val_id:
    :return:
    '''
    sen_vals = np.unique(Ztr)
    sen_val = sen_vals[sen_val_id]
    sen_idx = np.where(Ztr == sen_val)[0]
    y_true = ytr[sen_idx]
    y_pred = np.squeeze(y_pred)[sen_idx]

    assert len(y_true.shape) < 2, "y_true has shape: {}, expected (n,)".format(y_true.shape)
    assert len(y_pred.shape) < 2, "y_pred has shape: {}, expected (n,)".format(y_pred.shape)
    # equal_opportunity
    y_1_idx = np.where(y_true == 1)[0]
    y_true = y_true[y_1_idx]
    y_pred = y_pred[y_1_idx]
    total = y_true.shape[0]
    if total == 0 or y_pred.shape[0] == 0:
        print("0 samples: total# {}, pred#{}".format(total, y_pred.shape[0]))
        return np.nan, np.nan
    tnr, tpr = get_tpr(y_pred, y_true)
    return tpr, tnr


# TODO generalise OUTPUT for non-binary case by iterating over np.unique(Ztr)
def get_equal_fx(y_pred, Ztr, ytr, fairness_equal_fx, fx_module=lambda x: x):
    '''

    :param y_pred:
    :param Ztr:
    :param ytr:
    :param fairness_equal_fx:
    :param fx_module: change to np.abs if absolute fairness is necessary
    :return:
    '''
    odds_list = []
    assert len(np.unique(Ztr)) == 2, "more than 2 classes"
    for u_val in range(len(np.unique(Ztr))):
        odds_0, _ = fairness_equal_fx(Ztr, y_pred, ytr, sen_val_id=u_val)
        odds_list.append(odds_0)

    # TODO output generalise to non-binary!
    if len(odds_list) > 2:
        print("WARNING >2 classes")
        odds_0_p, odds_1_p = odds_list[0], odds_list[1]
    else:
        odds_0_p, odds_1_p = odds_list
    # follow the ideology
    # 0 - unprivilleged
    # 1 - privilleged
    # U- P
    gap_p = fx_module(odds_0_p - odds_1_p)
    return odds_0_p, odds_1_p, gap_p


# return types?
def get_demographic_parity(y_pred, Ztr, ytr):
    return get_equal_fx(y_pred, Ztr, ytr, demongraphic_parity)


def get_equal_opportunity(y_pred, Ztr, ytr):
    return get_equal_fx(y_pred, Ztr, ytr, fairness_equal_opportunity)


def get_equal_accuracy(y_pred, Ztr, ytr):
    return get_equal_fx(y_pred, Ztr, ytr, fairness_equal_accuracy)


def print_odds(odds_0, odds_1):
    print("Odds cls_0: {},cls_1: {}".format(odds_0, odds_1))
    print("Unfair GAP: {}".format(np.abs(odds_0 - odds_1)))


def plot_roc(fpr, tpr, roc_auc, cls):
    plt.figure()
    lw = 2
    plt.plot(fpr[cls], tpr[cls], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[cls])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


# a metric is a standardised measure => change names
fairness_metrics = ["EQ", "DP", "EA", "EO", "DI", "TH"]
fairness_metrics_names = ["Equal Opportunity", "Demographic Parity", "Equal Accuracy", "Equal Odds", "Disparate Impact",
                          "Theil Index"]
# fairness_functions = [get_equal_opportunity, get_demographic_parity, get_equal_accuracy]
