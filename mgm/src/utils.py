import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from mgm.src.evaluator import Evaluator
import os
import pickle
from mgm.src.MicroCorpus import MicroTokenizer, MicroCorpus

def eval_and_save(y_score, y_true, label_names, save_dir, activation="softmax"):
    if activation == "sigmoid":
        y_score = nn.Sigmoid()(torch.tensor(y_score)).numpy()
    elif activation == "softmax":
        y_score = nn.Softmax(dim=1)(torch.tensor(y_score)).numpy()
    elif activation == "none":
        y_score = torch.tensor(y_score).numpy()
    # y_score = nn.Softmax(dim=1)(torch.tensor(y_score)).numpy()
    evaluator = Evaluator(y_score, y_true, label_names=label_names, num_thresholds=100)
    metrics, avg_metrics = evaluator.eval()

    # save
    for label, metric in metrics.items():
        os.makedirs(os.path.join(save_dir), exist_ok=True)
        metric.to_csv(os.path.join(save_dir, f"{label}.csv"))
    avg_metrics.to_csv(os.path.join(save_dir, "avg.csv"))
    return avg_metrics

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'MicroTokenizer':
            return MicroTokenizer
        return super().find_class(module, name)