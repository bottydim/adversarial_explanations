import tensorflow as tf

# this should be only in the call module, all other modules should not have it!!!
# best keep it in the main fx!
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

from utils import *
from partial_dependence import *

# read models!
models_all = defaultdict(list)
data_list = get_data_list()

n_models = 6
n_seeds = 10
model_lists_names = [MODEL_MODIFIED, MODEL_CONSTANT, MODEL_ORIGINAL, ]
# for num_layers in range(n_models):
for num_layers in [5]:
    print(10 * "#")
    print("num_layers {}".format(num_layers))
    print(10 * "#")
    figs = []
    f_dict_list = []
    f_dict_list_FC = []
    for seed in range(n_seeds):
        print("M#{}-seed#{}".format(num_layers, seed))
        directory = "/homes/btd26/xai-research/xai-notebooks/Adversarial Explanations/temp_store/across_models/"

        #     seed = 0
        model_lists = load_model_lists(model_lists_names, directory, num_layers, seed, alpha=3.0)
        models_all[num_layers].append(model_lists)

#
#     f_dict_list_models.append(f_dict_list)
#     f_dict_list_models_FC.append(f_dict_list_FC)
