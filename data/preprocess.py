import itertools
import random
import numpy as np
import numpy.ma as ma
from sklearn.preprocessing import PowerTransformer

def normalize(X):
    mean, std = np.nanmean(X, axis=0), np.nanstd(X, axis=0) + 1e-5
    X = (X - mean) / std
    return np.clip(X, -100, 100)

def normalize_by_used_features(X, n_used_features, n_features, sqrt):
    if sqrt:
        return X / (n_used_features / n_features)**(0.5)
    return X / (n_used_features / n_features)

def remove_outliers(X, n_sigma):
    mean, std = np.nanmean(X, axis=0), np.nanstd(X, axis=0)
    cutoff = std * n_sigma
    lower, upper = mean - cutoff, mean + cutoff

    mask = (X >= lower) & (X <= upper) & ~np.isnan(X)
    masked_x = ma.masked_array(X, ~mask)
    mean, std = masked_x.mean(axis=0), masked_x.std(axis=0)
    cutoff = std * n_sigma
    lower =  mean - cutoff
    upper = mean + cutoff
    X = np.maximum(-np.log(1 + np.abs(X)) + lower, X)
    X = np.minimum(np.log(1 + np.abs(X)) + upper, X)
    return X

def power_transform(X, pt):
    for col in range(X.shape[-1]):
        pt.fit(X[:, col].reshape(-1, 1))
        X[:, col] = pt.transform(X[:, col].reshape(-1, 1)).reshape(-1)
    return X

def pad_inputs(x, max_feats=100):
    n_feats = x.shape[-1]
    padding = np.zeros((x.shape[0], max_feats - n_feats))
    return np.concatenate([x, padding], axis=-1)


def generate_configs(n_feats, n_classes):
    np.random.seed(42)

    feat_shift_configs = np.random.permutation(n_feats)
    class_shift_configs = np.random.permutation(n_classes)

    configs = list(itertools.product(feat_shift_configs, class_shift_configs))
    
    rng = random.Random(42)
    rng.shuffle(configs)

    preprocess_configs = ['none', 'power_all']
    style_configs = range(0, 1)
    configs = list(itertools.product(configs, preprocess_configs, style_configs))
    return configs

def prepare_inputs(x, y, n_ensemble_configs):
    n_feats = x.shape[-1]
    n_classes = len(np.unique(y))
    configs = generate_configs(n_feats, n_classes)[:n_ensemble_configs]

    x_inputs, y_inputs = [], []
    for config in configs:
        (feat_config, class_config), prep_config, style_config = config
        # preprocess input
        prepped = preprocess_input(x, prep_config)

        # rotate features
        y = (y + class_config) % n_classes
        x = np.concatenate([prepped[..., feat_config:], prepped[..., :feat_config]], axis=-1)
        x = pad_inputs(x, 100)
        
        x_inputs.append(x)
        y_inputs.append(y.astype(float))
    
    batch_x = np.stack(x_inputs, axis=0)
    batch_y = np.stack(y_inputs, axis=0)
    return batch_x, batch_y, configs


def preprocess_input(x, transform):

    x = normalize(x)
    if transform == 'power_all':
        pt = PowerTransformer(standardize=True)
        x = power_transform(x, pt)
    x = remove_outliers(x, 4)
    x = normalize_by_used_features(x, x.shape[-1], 100, True)
    return x


if __name__ == "__main__":
    dummy_x = np.random.rand(24, 25)
    dummy_y = np.random.randint(0, 9, (24, 1))

    x, y, configs = prepare_inputs(dummy_x, dummy_y, 3)
    
