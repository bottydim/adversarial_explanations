import numpy as np
import pandas as pd
import tensorflow as tf

# could check the below code
# https://github.com/tensorflow/cleverhans/blob/f1d233688e5aea473c93d7afdc04910292f2c2b3/cleverhans/utils_tf.py#L760

def no_attack(model, X, y, **kwargs):
    delta = tf.Variable(tf.zeros_like(X.shape[1:], dtype="float32"))
    return delta


def fgsm(model, X, y, epsilon=0.1, **kwargs):
    """ Construct FGSM adversarial examples on the examples X"""
    y = tf.cast(y, 'float32')
    delta = tf.Variable(tf.zeros_like(X))
    loss = tf.keras.losses.categorical_crossentropy
    loss_eval = loss(model(X + delta), y)
    optimizer = tf.keras.optimizers.RMSprop()
    with tf.GradientTape(persistent=True) as t:
        t.watch(delta)
        loss_eval = loss(model(X + delta), y)
    grads = t.gradient(loss_eval, delta)
    optimizer.apply_gradients(zip(grads, [delta]))
    return epsilon * tf.sign(grads)


def pgd_linf(model, X, y, epsilon=0.1, alpha=0.01, num_iter=30, randomize=False, **kwargs):
    """ Construct pgd_linf adversarial examples on the examples X"""
    assert X.dtype == tf.float32, "Cast X to float32"
    assert y.dtype == tf.float32, "Cast y to float32"
    y = tf.cast(y, 'float32')
    if randomize:
        delta = tf.Variable(tf.random.uniform(X.shape[1:]))
        delta = delta * 2 * epsilon - epsilon
    else:
        delta = tf.Variable(tf.random.uniform(X.shape[1:]))
    delta = tf.cast(delta, 'float32')
    for t in range(num_iter):
        loss = tf.keras.losses.categorical_crossentropy
        loss_eval = loss(model(X + delta), y)
        with tf.GradientTape(persistent=True) as t:
            t.watch(delta)
            loss_eval = loss(model(X + delta), y)
        grads = t.gradient(loss_eval, delta)
        #         optimizer.apply_gradients(zip([grads], [delta]))
        delta = tf.Variable(tf.clip_by_value((delta + alpha * tf.sign(grads)), -epsilon, epsilon))
    return delta


def epoch(loader, model, optimizer=None, verbose=0):
    """Standard training/evaluation epoch over the dataset"""
    total_loss, total_err = 0., 0.
    loss = tf.keras.losses.categorical_crossentropy
    n_batch = 0
    for X, y in loader:
        if optimizer:
            with tf.GradientTape(persistent=True) as t:
                yp = model(X)
                loss_eval = loss(y_true=y, y_pred=yp, from_logits=False)
            grads = t.gradient(loss_eval, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        loss_val, err = model.evaluate(X, y, verbose=verbose)
        err = 1 - err
        total_err += err
        err = 1 - err
        total_loss += loss_val
        n_batch += 1
    return total_err / n_batch, total_loss / n_batch


def epoch_adversarial(loader, model, attack, optimizer=None, **kwargs):
    """Adversarial training/evaluation epoch over the dataset"""
    total_loss, total_err = 0., 0.
    loss = tf.keras.losses.categorical_crossentropy
    n_batch = 0
    for X, y in loader:
        delta = attack(model, X, y, **kwargs)
        if optimizer:
            with tf.GradientTape(persistent=True) as t:
                t.watch(delta)
                yp = model(X + delta)
                loss_eval = loss(y_true=y, y_pred=yp, from_logits=False)
            grads = t.gradient(loss_eval, model.weights)
            optimizer.apply_gradients(zip(grads, model.weights))

        loss_val, err, = model.evaluate(X + delta, y, verbose=0)
        err = 1 - err
        total_err += err
        total_loss += loss_val
        n_batch += 1
    return total_err / n_batch, total_loss / n_batch


# options
# performance_loss = robust loss or non
# e_loss = performance_loss/

#TODO clip the gradient
def epoch_explanation(loader, model, attack, sensitive_feature_id, norm=1, normalise=True, e_alpha=0.25, optimizer=None,
                      **kwargs):
    """Adversarial training/evaluation epoch over the dataset"""
    total_loss, total_err = 0., 0.
    loss = tf.keras.losses.categorical_crossentropy
    n_batch = 0
    for X, y in loader:
        delta = attack(model, X, y, **kwargs)
        if optimizer:
            with tf.GradientTape(persistent=True) as t:
                t.watch(delta)
                t.watch(X)
                yp = model(X + delta)

                performance_loss = loss(y_true=y, y_pred=yp, from_logits=False)
                l = model.layers[-1]
                l.activation = tf.keras.activations.linear
                # TODO for CNN

                # TODO X or X + delta?
                explanation = t.gradient(performance_loss, [X])[0][:, sensitive_feature_id]
                # define the explanation loss to be L1 norm over the gradient of the sensitive feature of the input
                # L1 norm because we want to induce sparsity (this is a guess)
                # L1 norm makes is important when there is a differnce whether x is exactly 0 or not
                explanation_loss = tf.norm(explanation, norm)
                if normalise:
                    explanation_loss = tf.compat.v2.math.divide(explanation_loss, tf.cast(X.shape[0], "float32"))

                    # L = original_loss + \alpha * explanatoin_loss
                total_loss = performance_loss + e_alpha * explanation_loss

                # check position
                l.activation = tf.keras.activations.softmax

            # optimize with an optimizer
            grads = t.gradient(total_loss, [*model.weights])
            optimizer.apply_gradients(zip(grads, model.weights))

        loss_val, err, = model.evaluate(X + delta, y, verbose=0)
        err = 1 - err
        total_err += err
        total_loss += loss_val
        n_batch += 1
    return total_err / n_batch, total_loss / n_batch



def epoch_eval(adult_train, adult_test, model, z_idx):
    test_err, test_loss = epoch(adult_test, model)
    adv_err, adv_loss = epoch_adversarial(adult_test, model, pgd_linf, epsilon=0.25, alpha=0.08, num_iter=30)
    adv_err_f, adv_loss = epoch_adversarial(adult_test, model, fgsm)
    from explain_perturb import *
    e_loss = compute_explanation_loss_ds(adult_test, model, sensitive_feature_id=z_idx)
    e_loss_train = compute_explanation_loss_ds(adult_train, model, sensitive_feature_id=z_idx)
    return adv_err, adv_err_f, e_loss, e_loss_train, test_err
