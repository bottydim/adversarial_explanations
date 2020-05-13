import tensorflow as tf
from scipy import stats
#this should be only in the call module, all other modules should not have it!!!
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


from tensorflow import keras
from matplotlib import pyplot as plt
import os
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
print(tf.__version__)

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical


from explain_perturb import *
from evaluate import *
from train_utils import *
from datasets import *

if __name__ == '__main__':
    sensitive_feature_name = "gender"
    sensitive_features_dict = adult_sensitive_features_dict
    z_idx = sensitive_features_dict[sensitive_feature_name]
    Xtr, Xts, ytr, yts, Ztr, Zts = get_adult(sensitive_feature_name)
    X_test, X_train, Y_test, Y_train = prep_data(Xtr, Xts, ytr, yts)

    inputs = tf.convert_to_tensor(X_train, dtype=tf.float32)
    outputs = tf.convert_to_tensor(Y_train, dtype=tf.float32)
    models = []
    num_models = 5
    for i in range(num_models):
        model = build_model(input_shape=(X_train.shape[-1],), num_layers=i)
        file_name = "../temp_store/models/adult-{}.h5".format(i)
        model.load_weights(file_name)
        model.evaluate(X_train, Y_train)
        model.evaluate(X_test, Y_test)
        models.append(model)

    i = 3
    z_idx = sensitive_features_dict[sensitive_feature_name]

    model = models[i]
    print("Model-{}".format(i))
    acc, e_loss, p_loss = wrap_adv_train(X_test, inputs, Y_test, outputs, model, z_idx,
                                         n_epochs=50, lr=10e-7, alpha=0.27, normalise=True)
    # l_acc_list.append(acc)
    # l_e_loss.append(e_loss)
    # l_p_loss.append(p_loss)

    original_weights = []
    for i, model in enumerate(models):
        old_weights = model.get_weights()
        original_weights.append(old_weights)


    old_weights = original_weights[i]
    model_orig = tf.keras.models.clone_model(model)
    model_orig.build(input_shape=X_train.shape)  # replace 10 with number of variables in input layer
    model_orig.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model_orig.set_weights(old_weights)
    model_orig.layers[-1].activation = tf.keras.activations.softmax

    df = evaluate_config(X_test, X_train, Y_test, Y_train, acc, e_loss, p_loss, model_orig,
                         inputs, outputs, model, z_idx, title="Model-{}".format(i))