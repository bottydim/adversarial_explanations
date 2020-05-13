import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats


# this should be only in the call module, all other modules should not have it!!!
# # best keep it in the main fx! 
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)

# -----------------------------------------------------------------------------
# ATTRIBUTION METHODS
# -----------------------------------------------------------------------------
def saliency(model, X, ys=None):
    temp_act = model.layers[-1].activation
    model.layers[-1].activation = tf.keras.activations.linear
    with tf.GradientTape(persistent=True) as t:
        t.watch(X)
        y_ = model(X)
        y_ = tf.cond(ys is None, lambda: y_, lambda: tf.math.multiply(y_, ys))
    # as long as inside the grad tape => equivalent!
    #     if ys is not None:
    #         y_ = tf.math.multiply(y_,ys)
    g = t.gradient(y_, X)
    model.layers[-1].activation = temp_act
    return g


def gradient_x_input(model, X, ys=None):
    g = saliency(model, X, ys)
    return g * X


def integrated_grads(model, X, ys=None, x_baseline=None, x_steps=25):
    # from https://github.com/PAIR-code/saliency/blob/master/saliency/integrated_gradients.py#L21
    if x_baseline is None:
        x_baseline = np.zeros_like(X)

    assert x_baseline.shape == X.shape

    x_diff = X - x_baseline
    total_gradients = np.zeros_like(X)

    for alpha in np.linspace(0, 1, x_steps):
        x_step = x_baseline + alpha * x_diff

        g = saliency(model, x_step, ys)
        #         with tf.GradientTape(persistent=True) as t:
        #             t.watch(x_step)
        #             y_ = model(x_step)
        #         g = t.gradient(y_,x_step)
        total_gradients += g

    return total_gradients * x_diff / x_steps


@tf.custom_gradient
def guided_relu(x):
    f = tf.nn.relu(x)

    #     def grad(grad):
    #         print(grad[2,:],tf.where(grad>0,grad,tf.zeros_like(grad))[2,:])
    #         gate_g = tf.cast(grad > 0, "float32")
    #         gate_y = tf.cast(f > 0, "float32")
    #         return gate_y * gate_g * grad

    def grad(g):
        gate_g = tf.where(g > 0, g, tf.zeros_like(g))
        gate_y = tf.where(f > 0, tf.ones_like(g), tf.zeros_like(g))
        gate_g = tf.where(gate_y > 0, gate_g, tf.zeros_like(gate_g))
        return gate_g

    return f, grad


def guided_backprop(model, X, ys=None):
    # fix bug with the first reference to guided_relu
    _ = guided_relu
    # manual operation override
    swap_fx(model, tf.nn.relu, guided_relu)
    g = saliency(model, X, ys)
    # manual operation override reset
    swap_fx(model, guided_relu, tf.nn.relu)
    return g


def lime_explanation(model, X, ys=None, num_samples=1000, multiprocessing=True):
    from lime import lime_tabular
    import warnings
    warnings.filterwarnings("ignore", message="Singular matrix")
    warnings.filterwarnings("ignore", message="Ill-conditioned")

    # identify categorical data
    # https://stackoverflow.com/questions/47094676/how-to-identify-the-categorical-variables-in-the-200-numerical-variables?noredirect=1&lq=1
    X = X.numpy()
    ys = ys.numpy()
    df = pd.DataFrame(X)
    categorical_features = detect_categorical_top_k(df)
    # print(categorical_features)
    lime_expl = lime_tabular.LimeTabularExplainer(training_data=X,
                                                  training_labels=ys,
                                                  categorical_features=categorical_features,
                                                  class_names=np.unique(ys),
                                                  discretizer="decile"
                                                  )

    num_features = X.shape[0]
    predict_fn = lambda x: model.predict_proba(x)

    def lime_explanation_row(data_row, predict_fn, data_labels, num_features, num_samples):
        # print(data_row.shape)
        # experiment with lavels:
        # either labels=[data_label] or None
        explanation = lime_expl.explain_instance(data_row, predict_fn, labels=data_labels, top_labels=1,
                                                 num_features=num_features,
                                                 num_samples=num_samples, distance_metric='euclidean',
                                                 model_regressor=None)

        data_label = list(explanation.local_exp.keys())[0]
        # print(explanation.local_exp[data_label])

        # sort and return importance
        feature_importance = list(zip(*sorted(explanation.local_exp[data_label])))[1]
        return feature_importance

    from functools import partial

    lime_partial = partial(lime_explanation_row, predict_fn=predict_fn, data_labels=[], num_features=num_features,
                           num_samples=num_samples)

    if multiprocessing:
        print("*" * 10)
        print("NOT IMPLEMENTED: reverting to SLOW compute")
        print("*" * 10)
        result = np.apply_along_axis(lime_partial, 1, X)
        # result = parallel_apply_along_axis(lime_explanation_row, axis=1, arr=X,)
    else:
        result = np.apply_along_axis(lime_partial, 1, X)
    return result


def shap(model,X,ys=None):
    import shap
    # explain predictions of the model on four images
    if tf.is_tensor(X):
        X = X.numpy()
    e = shap.GradientExplainer(model, X)
    # ...or pass tensors directly
    # e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
    shap_values = e.shap_values(X)
    shap_cls = np.array(shap_values)
    if ys is None:
        vals = np.mean(np.abs(shap_cls), axis=0)
    else:
        idx_y = np.argmax(ys, axis=1)
        vals = np.zeros(shap_cls.shape[1:])
        for i in range(shap_cls.shape[1]):
            vals[i, :] = shap_cls[idx_y[i], i, :]
    return vals

def shapley(model, X, ys=None, samples=5, sampling_dims=None):
    '''
    X: EagerTensor
    sampling_dims: list of feature_ids to sample
    '''
    # verify
    # TODO Check proper type
    # TODO Reassign instead of using different variable
    if type(X) != np.ndarray:
        xs_numpy = X.numpy()
    else:
        xs_numpy = X
    dims = len(xs_numpy.shape)
    #     if has_multiple_inputs:
    #         raise RuntimeError('Multiple inputs not yet supported for perturbation methods')
    if sampling_dims is not None:
        if not 0 < len(sampling_dims) <= (dims - 1):
            raise RuntimeError('sampling_dims must be a list containing 1 to %d elements' % (dims - 1))
        if any([x < 1 or x > dims - 1 for x in sampling_dims]):
            raise RuntimeError('Invalid value in sampling_dims')
    else:
        sampling_dims = list(range(1, dims))

    X_shape = list(xs_numpy.shape)
    batch_size = xs_numpy.shape[0]
    n_features = int((np.prod([xs_numpy.shape[i] for i in sampling_dims])))
    result = np.zeros((X_shape[0], n_features))

    run_shape = list(X_shape)  # a copy
    # delete based on sampling
    run_shape = np.delete(run_shape, sampling_dims).tolist()
    run_shape.insert(-1, -1)

    # ?
    reconstruction_shape = [X_shape[0]]
    for j in sampling_dims:
        reconstruction_shape.append(X_shape[j])

    y_ = model(X)
    y = y_ if ys is None else np.multiply(y_, ys)

    for r in range(samples):
        p = np.random.permutation(n_features)
        x = X.reshape(run_shape)
        x = tf.Variable(x.reshape(X_shape), dtype=tf.float32)

        for i in p:
            t = x.numpy()
            t[:, i] = 0
            x = tf.assign(x, t)

            y0 = model(x)
            # double check whether we need np.abs!
            delta = np.abs(y - y0)
            result[:, i] += np.sum(delta)
            # double check whether that should be the case
            # could compare the difference to comparing to the original
            y = y0

    shapley = result / samples
    return shapley.reshape(reconstruction_shape)


attribution_methods = [saliency, gradient_x_input, integrated_grads, guided_backprop, shap,lime_explanation]
att_method_str = ["Gradients", "Gradient*Input", "Integrated Gradients", "Guided-Backprop","SHAP", "LIME"]


# -----------------------------------------------------------------------------
# STATISTICAL METHODS
# -----------------------------------------------------------------------------
def get_ranking(g, feature):
    '''
    return: ranks the abs value of a feature within an array
    '''
    sort_grad = np.argsort(np.abs(g), axis=1)[:, ::-1]
    return np.where(sort_grad == feature)[1]


def get_stats(r):
    s_o = stats.describe(r)
    min_o, max_o = s_o.minmax
    mean_o = np.round(s_o.mean, 2)
    skew_o = np.round(s_o.skewness, 2)
    kurtosis_o = np.round(s_o.kurtosis, 2)
    return min_o, max_o, mean_o, skew_o, kurtosis_o


def get_mode(g, feature):
    return stats.mode(get_ranking(g, feature)).mode[0]


def get_mode_(ranking, feature):
    return stats.mode(ranking).mode[0]


def num_points_top_k(ranking, k):
    '''
    #num points w/ ranking in top k
    '''
    n_features = np.max(ranking)
    if n_features <= k:
        return len(ranking)
    assert k > 0 and k < n_features, "k={} < n_features={}".format(k, n_features)
    # (array([  0, 800]), array([ 0,  k, n_features]))
    # the first bin is [1, 2) (including 1, but excluding 2) 
    return np.histogram(ranking, bins=[0, k, n_features])[0][0]


def get_num_max_points(r_o):
    np.histogram(r_o, range=(r_o.min(), int(r_o.shape[1])))[0][-1]


# -----------------------------------------------------------------------------
# ADVERSARIAL WEIGHT TRAINING
# -----------------------------------------------------------------------------
def adv_train(model, inputs, outputs, loss, learning_rate, alpha,
              sensitive_feature_id=-1, norm=1,
              normalise=False,
              verbose=True
              ):
    '''
    performs adverarial training.
    
    Alternative Implememtation that uses an optimizer!!:
    
    variables = [w1, b1, w2, b2]
    optimizer = tf.train.AdamOptimizer()
    with tf.GradientTape() as tape:
     y_pred = model.predict(x, variables)
     loss = model.compute_loss(y_pred, y)
     grads = tape.gradient(loss, variables)
     optimizer.apply_gradients(zip(grads, variables))
     
     https://www.tensorflow.org/guide/eager
    
    '''
    performance_loss, explanation_loss, total_loss = 0, 0, 0
    with tf.GradientTape(persistent=True) as t:
        # track the variable to differentiate with respect to
        t.watch(inputs)
        # compute the original loss that targets performance
        performance_loss = loss(y_true=outputs, y_pred=model(inputs), from_logits=False)
        # correct computation of the gradient with softmax necessitates 
        # the change of the activation function of the final layer 
        l = model.layers[-1]
        l.activation = tf.keras.activations.linear

        # compute the gradient wrt the sensitive feature of the input
        # d L_orig / dx 

        # c.f for package PAIR: saliency: "
        # y must be of size one, otherwise the gradient we get from tf.gradients
        # will be summed over all ys."

        explanation = t.gradient(performance_loss, [inputs])[0][:, sensitive_feature_id]
        # define the explanation loss to be L1 norm over the gradient of the sensitive feature of the input
        # L1 norm because we want to induce sparsity (this is a guess)
        # L1 norm makes is important when there is a differnce whether x is exactly 0 or not
        explanation_loss = tf.norm(explanation, norm)
        if normalise:
            explanation_loss = tf.compat.v2.math.divide(explanation_loss, tf.cast(inputs.shape[0], "float32"))
        # compute the gradient wrt to parameters to perform backpropagation

        # L = original_loss + \alpha * explanatoin_loss
        total_loss = performance_loss + alpha * explanation_loss

        # check position
        l.activation = tf.keras.activations.softmax

        # hackie, let's find a way to do it from optimizer
        ds = t.gradient(total_loss, [*model.weights])
        # TODO: not sure how this affects the explanation gradient, when params are changes
        l.activation = tf.keras.activations.softmax

        if verbose:
            print("Explanation loss:", explanation_loss)
    for i, w in enumerate(model.weights):
        w.assign_sub(learning_rate * ds[i])  # w = w*-nu*dL/dW
    #     acc = model.evaluate(inputs,outputs)
    history = {}
    history["performance_loss"] = performance_loss
    history["explanation_loss"] = explanation_loss
    history["total_loss"] = total_loss
    #     history["acc"] = acc[1]
    #     history["explanation_loss"] = explanation_loss

    return history


def compute_explanation_loss_ds(dataset,model,sensitive_feature_id,
                             norm=1,
                             normalise=True,
                             loss=None,
                             ):
    # raise NotImplemented()
    e_loss= 0
    n_batch = 0
    for inputs,outputs in dataset:
        e_loss += compute_explanation_loss(inputs, outputs, model, sensitive_feature_id,
                                 norm,normalise,loss)
        n_batch+=1
    return e_loss/n_batch


def compute_explanation_loss(inputs, outputs, model, sensitive_feature_id,
                             norm=1,
                             normalise=True,
                             loss=None,
                             ):
    with tf.GradientTape(persistent=True) as t:
        # track the variable to differentiate with respect to
        t.watch(inputs)
        # compute the original loss that targets performance
        if loss is None:
            loss = model.loss
        performance_loss = loss(y_true=outputs, y_pred=model(inputs), from_logits=False)
        # correct computation of the gradient with softmax necessitates
        # the change of the activation function of the final layer
        l = model.layers[-1]
        l.activation = tf.keras.activations.linear

        # compute the gradient wrt the sensitive feature of the input
        # d L_orig / dx

        # c.f for package PAIR: saliency: "
        # y must be of size one, otherwise the gradient we get from tf.gradients
        # will be summed over all ys."

    explanation = t.gradient(performance_loss, [inputs])[0][:, sensitive_feature_id]
    # define the explanation loss to be L1 norm over the gradient of the sensitive feature of the input
    # L1 norm because we want to induce sparsity (this is a guess)
    # L1 norm makes is important when there is a differnce whether x is exactly 0 or not
    explanation_loss = tf.norm(explanation, norm)
    if normalise:
        explanation_loss = tf.compat.v2.math.divide(explanation_loss, tf.cast(inputs.shape[0], "float32"))
    # compute the gradient wrt to parameters to perform backpropagation

    # reset activation
    l.activation = tf.keras.activations.softmax

    return explanation_loss

def compute_hessian_diag(inputs, outputs, model, sensitive_feature_id,
                             norm=1,
                             normalise=True,
                             loss=None,
                             ):
    with tf.GradientTape(persistent=True) as t:
        # track the variable to differentiate with respect to
        t.watch(inputs)
        # compute the original loss that targets performance
        if loss is None:
            loss = model.loss
        performance_loss = loss(y_true=outputs, y_pred=model(inputs), from_logits=False)
        # correct computation of the gradient with softmax necessitates
        # the change of the activation function of the final layer
        l = model.layers[-1]
        l.activation = tf.keras.activations.linear

        # compute the gradient wrt the sensitive feature of the input
        # d L_orig / dx

        # c.f for package PAIR: saliency: "
        # y must be of size one, otherwise the gradient we get from tf.gradients
        # will be summed over all ys."

        explanation = t.gradient(performance_loss, [inputs])[0][:, sensitive_feature_id]

    hessian_diag = t.gradient(explanation, [inputs])[0][:, sensitive_feature_id]
    # reset activation
    l.activation = tf.keras.activations.softmax

    return hessian_diag


def compute_hessian_full_batch(inputs, outputs, model, sensitive_feature_id,
                             norm=1,
                             normalise=True,
                             loss=None,
                             ):
    with tf.GradientTape(persistent=True) as t:
        # track the variable to differentiate with respect to
        t.watch(inputs)
        # compute the original loss that targets performance
        y_pred = model(inputs)
        if loss is None:
            loss = model.loss
        performance_loss = loss(y_true=outputs, y_pred=y_pred, from_logits=False)
        # correct computation of the gradient with softmax necessitates
        # the change of the activation function of the final layer
        l = model.layers[-1]
        l.activation = tf.keras.activations.linear

        # compute the gradient wrt the sensitive feature of the input
        # d L_orig / dx

        # c.f for package PAIR: saliency: "
        # y must be of size one, otherwise the gradient we get from tf.gradients
        # will be summed over all ys."

        explanation = t.gradient(performance_loss, [inputs])[0]  # [:, sensitive_feature_id]

    hessian_full = t.batch_jacobian(explanation, inputs)
    # reset activation
    l.activation = tf.keras.activations.softmax

    return hessian_full


def compute_max_eigen(hessian_full):
    eigen_t = tf.linalg.eigh(
        hessian_full,
        name="Eigen_H")
    max_eigen_val = tf.reduce_max(eigen_t[0], axis=1)
    return max_eigen_val

def analyze_max_eigen(inputs, outputs, model):
    hessian_full = compute_hessian_full_batch(inputs, outputs, model)
    max_eigen_val = compute_max_eigen(hessian_full)
    from utils import distplot
    distplot(max_eigen_val.numpy())
# -----------------------------------------------------------------------------
# WRAPPER
# -----------------------------------------------------------------------------
def wrap_adv_train(X_test, inputs, Y_test, outputs, model, z_idx,
                   n_epochs=40,
                   lr=10e-7,
                   alpha=2e-4,
                   normalise=True,
                   norm=1,
                   loss = None,
                   verbose=True
                   ):
    if hasattr(model, 'loss'):
        loss = model.loss
    elif loss is None:
        raise Exception("Model Custom Loss Required")
    # loss = tf.keras.losses.categorical_crossentropy


    epochs = range(n_epochs)

    sensitive_feature_id = z_idx
    print("Sensitive feature: {}".format(sensitive_feature_id))

    # compute explanation loss:
    with tf.GradientTape(persistent=True) as t:
        t.watch(inputs)
        performance_loss = model.loss(y_true=outputs, y_pred=model(inputs), from_logits=False)
    explanation = t.gradient(performance_loss, [inputs])[0][:, sensitive_feature_id]
    # define the explanation loss to be L1 norm over the gradient of the sensitive feature of the input
    # L1 norm because we want to induce sparsity (this is a guess)
    # L1 norm makes is important when there is a differnce whether x is exactly 0 or not
    explanation_loss = tf.norm(explanation, norm)

    loss_tr_o, acc_tr_o = model.evaluate(inputs, outputs)
    l = loss_tr_o
    a = acc_tr_o
    p_loss = [l]
    e_loss = [explanation_loss.numpy()]
    t_loss = [l]
    acc_list = [[l, a]]
    acc_list_test = []
    # lr = 0.00001
    # alpha=0.001

    # conf 1
    # lr = 10e-7
    # normalise=True
    # alpha=2e-1

    # these were the params to get the kick-ass model?
    # w/o l.activation = tf.keras.activations.softmax
    # lr = 10e-8
    # alpha=2e-4
    # normalise=False
    # lr_decay_stage = 0

    lr_decay_stage = 0
    from train_utils import step_decay
    for epoch in epochs:
        # exponential decay drops too quickly!

        if epoch > 1 and e_loss[-1] > e_loss[-2]:
            lr_decay_stage += 1
            #     lr = exp_decay(epoch,lr)
            lr = step_decay(lr_decay_stage, initial_lrate=lr, drop=0.9, epochs_drop=1)

        performance_loss = model.loss(y_true=outputs, y_pred=model(inputs), from_logits=False)
        # sensitive_feature_id=-1, because again we appended age to the end

        history = adv_train(model, inputs, outputs, loss,
                            learning_rate=lr, alpha=alpha,
                            sensitive_feature_id=sensitive_feature_id,
                            norm=norm,
                            normalise=normalise,
                            verbose=verbose)
        performance_loss = history["performance_loss"]
        explanation_loss = history["explanation_loss"]
        total_loss = history["total_loss"]
        #     acc = history["acc"]
        p_loss.append(np.mean(performance_loss))
        e_loss.append(explanation_loss.numpy())
        t_loss.append(np.mean(total_loss))
        acc_list.append(model.evaluate(inputs, outputs, verbose=0))
        acc_list_test.append(model.evaluate(X_test, Y_test, verbose=0))
    return acc_list, e_loss, p_loss


# -----------------------------------------------------------------------------
# UTILS 
# -----------------------------------------------------------------------------

def swap_fx(model, fx_old, fx_new):
    for l in model.layers:
        if l.activation == fx_old:
            l.activation = fx_new


def detect_categorical_1(df):
    likely_cat = {}
    for var in df.columns:
        likely_cat[var] = 1. * df[var].nunique() / df[var].count() < 0.05  # or some other threshold
    return likely_cat


def detect_categorical_top_k(df, top_n=10):
    likely_cat = {}
    for var in df.columns:
        likely_cat[var] = 1. * df[var].value_counts(normalize=True).head(top_n).sum() > 0.8
    return likely_cat


# MULTIPROCESSING

def unpacking_apply_along_axis(all_args):
    (func1d, axis, arr, args, kwargs) = all_args
    """
    Like numpy.apply_along_axis(), but with arguments in a tuple
    instead.

    This function is useful with multiprocessing.Pool().map(): (1)
    map() only handles functions that take a single argument, and (2)
    this function can generally be imported from a module, as required
    by map().
    """
    return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)


def parallel_apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """
    Like numpy.apply_along_axis(), but takes advantage of multiple
    cores.
    """
    import multiprocessing
    # Effective axis where apply_along_axis() will be applied by each
    # worker (any non-zero axis number would work, so as to allow the use
    # of `np.array_split()`, which is only done on axis 0):
    effective_axis = 1 if axis == 0 else axis
    if effective_axis != axis:
        arr = arr.swapaxes(axis, effective_axis)

    # Chunks for the mapping (only a few chunks):
    chunks = [(func1d, effective_axis, sub_arr, args, kwargs)
              for sub_arr in np.array_split(arr, multiprocessing.cpu_count())]

    pool = multiprocessing.Pool()
    individual_results = pool.map(unpacking_apply_along_axis, chunks)
    # Freeing the workers:
    pool.close()
    pool.join()

    return np.concatenate(individual_results)



