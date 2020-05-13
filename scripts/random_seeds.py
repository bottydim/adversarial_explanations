
import tensorflow as tf
if __name__ == '__main__':
    tf.enable_eager_execution()
#### THIS SHOULD BE AT THE TOP!



# this should be only in the call module, all other modules should not have it!!!
# best keep it in the main fx!
from utils_across import get_original_models, get_nulify_models, get_modified_models
config = tf.ConfigProto()
# config.gpu_options.visible_device_list = str('1')
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
#### use with keras
# from keras.backend.tensorflow_backend import set_session
# set_session(sess)


print(tf.__version__)

from utils import *
import argparse


# In[26]:


# In[27]:


def main(file_dir, num_layers):
    # models_list_SEEDS = []

    global_timer = Timer()
    for seed in range(5, 10):
        model_orig_list = get_original_models(seed, num_layers=num_layers)
        models_nulify = get_nulify_models(seed, num_layers=num_layers)
        alpha_vals = [5e-03, 5e-02, 5e-01, 3e0, 1e1, 2e1, 5e1]
        alpha = alpha_vals[3]
        for alpha in alpha_vals[3:4]:
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


if __name__ == '__main__':
    # READ!!!
    # across_models sample_size=1000!!!!!

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_layers", type=int, help="the number of layers for the model", default=2, )
    args = parser.parse_args()
    directory = "/homes/btd26/xai-research/xai-notebooks/Adversarial Explanations/temp_store/across_models/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.join(directory, "model_{}".format(args.num_layers))
    main(directory, num_layers=args.num_layers)
