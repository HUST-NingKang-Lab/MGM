import pandas as pd
from tqdm import trange
import numpy as np
from sklearn.metrics import confusion_matrix
from functools import reduce


class Evaluator:

    def __init__(self, predictions, label_ids, label_names,
                 num_thresholds=100, par=None, nafill=0):
        self.predictions = predictions
        # if label_ids is not one-hot encoded, convert it to one-hot encoding
        if len(label_ids.shape) == 1:
            label_ids = np.eye(len(label_names))[label_ids]
        self.actual_labels = label_ids
        self.label_names = label_names
        
        self.num_thresholds = num_thresholds
        self.thresholds = (np.arange(num_thresholds+2) / num_thresholds).reshape(num_thresholds+2, 1) # col vector

        self.par = par
        self.nafill = nafill
        '''self.lw = 1
        self.colors = ListedColormap(sns.color_palette("husl", 4))
        colors = self.colors.colors'''
        #self.cmap = {name: color for name, color in zip(self.score_names + ['L'], colors)}

    def eval(self):
        
        labels = self.label_names
        predictions = pd.DataFrame(self.predictions, columns=labels)
        actual_sources = pd.DataFrame(self.actual_labels, columns=labels)
        metrics_layer = dict(eval_single_label(predictions[label], actual_sources[label], self.thresholds, self.nafill) for label in labels)
        

        # list all metrics to be averaged in order to calculate metrics for a layer
        avg_metrics = ['Acc', 'Sn', 'Sp', 'TPR', 'FPR', 'Rc', 'Pr', 'F1', 'F-max', 'ROC-AUC', 'PR-AUC']
        avg_metrics_layer = pd.DataFrame(np.concatenate( [np.expand_dims(metrics_layer[label][avg_metrics].to_numpy(), 2)
                                                for label in labels], axis=2).mean(axis=2), columns=avg_metrics)
        avg_metrics_layer = avg_metrics_layer.round(4)
        return metrics_layer, avg_metrics_layer


def eval_single_label(predictions, actual_sources, thresholds, nafill):
    label = actual_sources.name
    print('Evaluating biome source:', label)

    # calculate predicted label for samples
    # samples with contribution above the threshold are considered as POSITIVE, otherwise NEGATIVE
    # This is a vectorized version using numpy broadcasting
    pred_source = (predictions.to_numpy().reshape(1, predictions.shape[0]) >= thresholds).astype(np.uint)
    pred_source = pd.DataFrame(pred_source, columns=predictions.index)
    actual_sources = actual_sources.astype(np.uint)
    metrics = pd.DataFrame()
    metrics['t'] = thresholds.flatten()
    num_thresholds = metrics['t'].shape[0] - 2
    # calculate TP, TN, FN, FP using sklearn
    conf_matrix = metrics['t'].apply(lambda T: confusion_matrix(actual_sources, pred_source.iloc[int(T * num_thresholds), :], labels=[0, 1]).ravel())
    conf_metrics = pd.DataFrame(conf_matrix.tolist(), columns=['TN', 'FP', 'FN', 'TP']).astype(int)
    metrics = pd.concat( (metrics, conf_metrics), axis=1).set_index('t')
    metrics['Acc'] = metrics[['TP', 'TN']].sum(axis=1) / metrics.sum(axis=1)
    metrics['Sn'] = metrics['TP'] / (metrics['TP'] + metrics['FN'])
    metrics['Sp'] = metrics['TN'] / (metrics['TN'] + metrics['FP'])
    metrics['TPR'] = metrics['TP'] / (metrics['TP'] + metrics['FN'])
    metrics['FPR'] = metrics['FP'] / (metrics['TN'] + metrics['FP'])
    metrics['Rc'] = metrics['TP'] / (metrics['TP'] + metrics['FN'])
    metrics['Pr'] = metrics['TP'] / (metrics['TP'] + metrics['FP'])
    metrics = metrics.fillna(nafill)
    metrics['F1'] = (2 * metrics['Pr'] * metrics['Rc'] / (metrics['Pr'] + metrics['Rc']))
    idx = metrics.index
    metrics['ROC-AUC'] = ((metrics.loc[idx[:-1], 'TPR'].to_numpy() + metrics.loc[idx[1:], 'TPR'].to_numpy()) *
                          (metrics.loc[idx[:-1], 'FPR'].to_numpy() - metrics.loc[idx[1:], 'FPR'].to_numpy()) / 2).sum()
    metrics['PR-AUC'] = ((metrics.loc[idx[:-1], 'Pr'].to_numpy() + metrics.loc[idx[1:], 'Pr'].to_numpy()) *
                         (metrics.loc[idx[:-1], 'Rc'].to_numpy() - metrics.loc[idx[1:], 'Rc'].to_numpy()) / 2).sum()
    metrics['F-max'] = metrics['F1'].max()
    metrics = metrics.round(4)
    print(metrics)
    return label, metrics