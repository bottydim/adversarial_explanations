# this should be only in the call module, all other modules should not have it!!!
# best keep it in the main fx!
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
#### use with keras
# from keras.backend.tensorflow_backend import set_session
# set_session(sess)


print(tf.__version__)

from evaluate import *
from utils import *
from utils_across import *
from partial_dependence import *


def o():
    # read models!
    data_list = get_data_list()

    n_models = 6
    n_seeds = 10
    directory = "/homes/btd26/xai-research/xai-notebooks/Adversarial Explanations/temp_store/"
    models_all = load_models(n_models, n_seeds,directory=os.path.join(directory,"across_models/"))

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
            df = generate_table_methods(models_list_p, get_original_model=get_m_o,
                                        data_list=data_list, f_nb_list=None, use_train=True,
                                        verbose=1)
            df_name = "df_attribution_train_num_layers{}_seed{}.pandas"
            df.to_pickle(os.path.join(directory,"tables/{}"))
            #         diff_mean_tr, df_top_diff_tr, shifts_sum_tr, shifts_mean_tr = analyze_summary_table(df, title=title)
            print("Computing Summary Table Test")
            df_test = generate_table_methods(models_list_p, get_original_model=get_m_o,
                                             data_list=data_list, f_nb_list=None, use_train=False,
                                             verbose=1)
            attribution_methods[num_layers].append((df, df_test))


def load_models(n_models, n_seeds, alpha=3.0,
                directory="/homes/btd26/xai-research/xai-notebooks/Adversarial Explanations/temp_store/across_models/"):
    models_all = defaultdict(list)
    for num_layers in range(n_models):
        print(10 * "#")
        print("num_layers {}".format(num_layers))
        print(10 * "#")
        for seed in range(n_seeds):
            print("M#{}-seed#{}".format(num_layers, seed))

            model_lists_names = [MODEL_MODIFIED, MODEL_CONSTANT, MODEL_ORIGINAL, ]
            num_layers = num_layers
            #     seed = 0
            model_lists = load_model_lists(model_lists_names, directory, num_layers, seed, alpha=alpha)
            models_all[num_layers].append(model_lists)
    return models_all







def main(file_dir, num_layers):
    # models_list_SEEDS = []

    global_timer = Timer()
    for seed in range(10):
        model_orig_list = get_original_models(seed, num_layers=num_layers)
        models_nulify = get_nulify_models(seed, num_layers=num_layers)
        alpha_vals = [5e-03, 5e-02, 5e-01, 3e0, 1e1, 2e1, 5e1]
        alpha = alpha_vals[3]
        for alpha in alpha_vals[3:]:
            print("alpha: {}".format(alpha))
            models_list_p, models_list_p_r, models_list_p_R = get_modified_models(model_orig_list, alpha, seed,
                                                                                  sample_size=30000,
                                                                                  num_layers=num_layers)
            # model_lists = [models_list_p, models_list_p_r, models_list_p_R, models_nulify, model_orig_list, ]
            # model_lists_names = ["Modified", "Modified (r)", "Modified (R)", "x_i Constant", "Original", ]
            model_lists = [models_list_p, models_nulify, model_orig_list, ]
            model_lists_names = [MODEL_MODIFIED, MODEL_CONSTANT, MODEL_ORIGINAL, ]
            # save models
            directory = os.path.join(file_dir, "seed_{}".format(seed))
            if not os.path.exists(directory):
                os.makedirs(directory)
            save_model_lists(model_lists, model_lists_names, alpha,
                             file_dir="{}".format(
                                 directory))
            print("seed: {} ==== TIME: {}".format(seed, global_timer.get_time_hhmmss()))
            # models_list_SEEDS.append(model_lists)
