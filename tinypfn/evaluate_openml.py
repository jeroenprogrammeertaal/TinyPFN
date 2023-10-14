import numpy as np
import pickle

from tinygrad.tensor import Tensor
from data.preprocess import prepare_inputs
from tinypfn import TinyPFNTransformer, load_ckpt_weights
from tabpfn import TabPFNClassifier


from sklearn.metrics import roc_auc_score

def get_datasets(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def evaluate_dataset(tiny_model, paper_model, name, data, labels):
    train_idx = int(data.shape[0] * 0.5)
    n_labels = len(np.unique(labels))
    n_features = data.shape[-1]
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    data, labels = data[idx], labels[idx]
    print(f"{name}, n_classes: {n_labels}, n_features: {n_features}, n_training: {train_idx}, n_test: {len(labels[train_idx:])}\n")
    x, y, configs = prepare_inputs(data.numpy(), labels.numpy(), 1, train_idx=train_idx)
    tiny_output = tiny_model.forward(Tensor(x), Tensor(y).unsqueeze(-1), train_idx)
    tiny_preds = tiny_model.get_class_probs(tiny_output, configs, n_labels)
    
    paper_model.fit(data[:train_idx, :], labels[:train_idx], overwrite_warning=True)
    paper_preds = paper_model.predict_proba(data[train_idx:, :])

    if n_labels <= 2:
        tiny_preds = tiny_preds.argmax(-1)
        paper_preds = np.argmax(paper_preds, axis=1)

    tiny_roc_auc= roc_auc_score(labels[train_idx:], tiny_preds[train_idx:].numpy(), multi_class='ovo')
    paper_roc_auc = roc_auc_score(labels[train_idx:], paper_preds, multi_class='ovo')

    if n_labels > 2:
        tiny_preds = tiny_preds.argmax(-1)
        paper_preds = np.argmax(paper_preds, axis=1)

    tiny_acc = sum(tiny_preds[train_idx:].numpy() == labels[train_idx:].numpy()) / len(labels[train_idx:])
    paper_acc = sum(paper_preds == labels[train_idx:].numpy()) / len(labels[train_idx:])
    print(f"tiny roc auc: {tiny_roc_auc}, paper roc auc: {paper_roc_auc}, tiny acc: {tiny_acc}, paper_acc: {paper_acc}\n")

if __name__ == "__main__":
    model = TinyPFNTransformer(100, 512, 1024, 12, 0.5)
    load_ckpt_weights(model)
    np.random.seed(42)

    tabpfn = TabPFNClassifier(device='cpu', N_ensemble_configurations=1)

    datasets = get_datasets("data/openml_datasets/cc_valid_datasets_multiclass.pickle")
    for dataset in datasets:
        name, data, labels, classes, class_names, config = dataset
        perc_nan = np.count_nonzero(np.isnan(data.numpy())) / data.numel()
        evaluate_dataset(model, tabpfn, name, data, labels)