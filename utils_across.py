import tensorflow as tf

# this should be only in the call module, all other modules should not have it!!!
# best keep it in the main fx!
if __name__ == '__main__':
    tf.enable_eager_execution()
config = tf.ConfigProto()
# config.gpu_options.visible_device_list = str('1')
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)

from datasets import dataset_fs, prep_data, f_sensitive_list
from evaluate import feature_importance_nulify, evaluate_pertrubed_models
from train_utils import build_model, fit_model


def get_original_models(seed, num_layers=3):
    model_orig_list = []
    for i, f in enumerate(dataset_fs):
        Xtr, Xts, ytr, yts, Ztr, Zts = f(0, remove_z=False)
        X_test, X_train, Y_test, Y_train = prep_data(Xtr, Xts, ytr, yts)
        optimizer = tf.keras.optimizers.Adam(lr=0.01)
        n_features = X_train.shape[-1]
        batch_size = 30000
        model_full = build_model(input_shape=(n_features), num_layers=num_layers, optimizer=optimizer, seed=seed)
        EPOCHS = 1000
        history = fit_model(model_full, X_train, Y_train, EPOCHS, batch_size=batch_size, verbose=0)
        model_orig_list.append(model_full)
    return model_orig_list


def get_nulify_models(seed, num_layers=3):
    f_nb_list = f_sensitive_list
    models_nulify = []
    for i, f in enumerate(dataset_fs):
        Xtr, Xts, ytr, yts, Ztr, Zts = f(0, remove_z=False)
        X_test, X_train, Y_test, Y_train = prep_data(Xtr, Xts, ytr, yts)
        models_nf, _, _ = feature_importance_nulify(X_train, Y_train, X_test, Y_test, feature_idx=f_nb_list[i],
                                                    seed=seed, num_layers=num_layers)
        models_nulify.append(models_nf)
    return models_nulify


def get_modified_models(model_orig_list, alpha, seed, sample_size=1000, num_layers=3):
    import random
    import time
    f_nb_list = f_sensitive_list

    start = time.time()

    # choose b/t min if undersampling or max for all

    sample_fx = min
    # sample_fx = max

    adv_lis_list = []
    models_list_p = []

    # robust
    adv_lis_list_r = []
    models_list_p_r = []

    # robust train
    adv_lis_list_R = []
    models_list_p_R = []

    for i, f in enumerate(dataset_fs):
        Xtr, Xts, ytr, yts, Ztr, Zts = f(0, remove_z=False)
        X_test, X_train, Y_test, Y_train = prep_data(Xtr, Xts, ytr, yts)
        random.seed(30)
        n_samples = X_train.shape[0]
        ix_sample = random.sample(range(n_samples), sample_fx(sample_size, n_samples))
        X_train = X_train[ix_sample, :]
        Y_train = Y_train[ix_sample, :]
        #         f_sort = np.argsort(f_nb_list[i])[::-1]
        #         feature_set = [f_sort[0],f_sort[int(len(f_sort)/2)],f_sort[-1]]
        feature_set = f_nb_list[i]
        print("ds {}, alpha {}, features {}".format(f, alpha, feature_set))
        # explain without robust
        models_p, adv_lis = evaluate_pertrubed_models(X_train, Y_train, X_test, Y_test, e_alpha=alpha,
                                                      feature_set=feature_set, model_orig=model_orig_list[i], seed=seed,
                                                      num_layers=num_layers)
        adv_lis_list.append(adv_lis)
        models_list_p.append(models_p)

        # models_p_r, adv_lis_r = evaluate_pertrubed_models(X_train, Y_train, X_test, Y_test,
        #                                                   e_alpha=alpha, attack=pgd_linf, train_robust=False,
        #                                                   feature_set=feature_set, seed=seed, num_layers=num_layers)
        # adv_lis_list_r.append(adv_lis_r)
        # models_list_p_r.append(models_p_r)
        end = time.time()
        print("TIME: {}".format(end - start))

        # models_p_R, adv_lis_R = evaluate_pertrubed_models(X_train, Y_train, X_test, Y_test,
        #                                                   e_alpha=alpha, attack=pgd_linf, train_robust=True,
        #                                                   feature_set=feature_set, seed=seed, num_layers=num_layers)

        # adv_lis_list_R.append(adv_lis_R)
        # models_list_p_R.append(models_p_R)

        end = time.time()
        print("TIME: {}".format(end - start))
    return models_list_p, models_list_p_r, models_list_p_R