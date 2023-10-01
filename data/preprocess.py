import itertools
import random
import numpy as np
import numpy.ma as ma
import torch
from sklearn.preprocessing import PowerTransformer
from data.preprocess_paper import preprocess_input as paper_preprocess

def normalize(X: np.ndarray, train_idx) -> np.ndarray:
    mean, std = np.nanmean(X[:train_idx], axis=0), np.nanstd(X[:train_idx], axis=0) + 1e-5
    X = (X - mean) / std
    return np.clip(X, -100, 100)

def normalize_by_used_features(X: np.ndarray, n_used_features: int, n_features: int, sqrt: bool=False) -> np.ndarray:
    if sqrt:
        return X / (n_used_features / n_features)**(0.5)
    return X / (n_used_features / n_features)

def remove_outliers(X: np.ndarray, n_sigma: int, train_idx: int) -> np.ndarray:
    mean, std = np.nanmean(X[:train_idx], axis=0), np.nanstd(X[:train_idx], axis=0)
    cutoff = std * n_sigma
    lower, upper = mean - cutoff, mean + cutoff

    mask = (X >= lower) & (X <= upper) & ~np.isnan(X)
    masked_x = ma.masked_array(X, ~mask)
    mean, std = masked_x[:train_idx].mean(axis=0), masked_x[:train_idx].std(axis=0)
    cutoff = std * n_sigma
    lower =  mean - cutoff
    upper = mean + cutoff
    X = np.maximum(-np.log(1 + np.abs(X)) + lower, X)
    X = np.minimum(np.log(1 + np.abs(X)) + upper, X)
    return X

def power_transform(X: np.ndarray, pt: PowerTransformer, train_idx: int, categorical_features: list[int]) -> np.ndarray:
    for col in range(X.shape[-1]):
        if col not in categorical_features:
            pt.fit(X[:train_idx, col].reshape(-1, 1))
            X[:, col] = pt.transform(X[:, col].reshape(-1, 1)).reshape(-1)
    return X

def pad_inputs(x: np.ndarray, max_feats: int=100) -> np.ndarray:
    n_feats = x.shape[-1]
    padding = np.zeros((x.shape[0], max_feats - n_feats))
    return np.concatenate([x, padding], axis=-1)


def generate_configs(n_feats: int, n_classes: int) -> list:
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

def prepare_inputs(x: np.ndarray, y: np.ndarray, n_ensemble_configs: int, train_idx: int):
    n_feats = x.shape[-1]
    n_classes = len(np.unique(y))
    configs = generate_configs(n_feats, n_classes)[:n_ensemble_configs]

    x_inputs, y_inputs = [], []
    for config in configs:
        (feat_config, class_config), prep_config, style_config = config
        # preprocess input
        prepped = preprocess_input(x, prep_config, train_idx)
        #paper_prepped = paper_preprocess(torch.from_numpy(x[:, None, :]), y, ((class_config, feat_config), prep_config, style_config)).squeeze()

        prepped_y = (y + class_config) % n_classes
        prepped = np.concatenate([prepped[..., feat_config:], prepped[..., :feat_config]], axis=-1)
        prepped = pad_inputs(prepped, 100)
        
        x_inputs.append(prepped)
        y_inputs.append(prepped_y.astype(float))
        
    
    batch_x = np.stack(x_inputs, axis=0)
    batch_y = np.stack(y_inputs, axis=0)
    return batch_x, batch_y, configs

def preprocess_input(x: np.ndarray, transform: str, train_idx: int) -> np.ndarray:
    categorical_features = get_cat_feats(x)
    x = normalize(x, train_idx)
    if transform == 'power_all':
        pt = PowerTransformer(standardize=True)
        x = power_transform(x, pt, train_idx, categorical_features)
    x = remove_outliers(x, 4, train_idx)
    x = normalize_by_used_features(x, x.shape[-1], 100, False)
    x = np.nan_to_num(x, nan=0)
    return x

def get_cat_feats(data):
    n_cols = data.shape[-1]
    cats = []
    for i in range(n_cols):
        all_int = np.array_equal(data[:, i], np.floor(data[:, i]))
        if all_int:
            cats.append(i)
    return cats

if __name__ == "__main__":
    dummy_x = np.random.rand(24, 25)
    dummy_y = np.random.randint(0, 9, (24, 1))

    x, y, configs = prepare_inputs(dummy_x, dummy_y, 3)
