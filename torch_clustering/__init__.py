# -*- coding: UTF-8 -*-
'''
@Project : torch_clustering 
@File    : __init__.py
@Author  : Zhizhong Huang from Fudan University
@Homepage: https://hzzone.github.io/
@Email   : zzhuang19@fudan.edu.cn
@Date    : 2022/10/19 12:21 PM 
'''
import torch
from .kmeans.kmeans import PyTorchKMeans
from .faiss_kmeans import FaissKMeans
from .gaussian_mixture import PyTorchGaussianMixture
from .beta_mixture import BetaMixture1D

import numpy as np
from munkres import Munkres
from sklearn import metrics
import warnings
from scipy.optimize import linear_sum_assignment

def evaluate_clustering(label, pred, eval_metric=['nmi', 'acc', 'ari'], phase='train', class_names=False, confusion_matrix_file=None):
    mask = (label != -1)
    label = label[mask]
    pred = pred[mask]
    results = {}
    num_classes = np.unique(label).size
    num_elems = label.size
    if 'nmi' in eval_metric:
        nmi = metrics.normalized_mutual_info_score(label, pred, average_method='arithmetic')
        results[f'{phase}_nmi'] = nmi
    if 'ari' in eval_metric:
        ari = metrics.adjusted_rand_score(label, pred)
        results[f'{phase}_ari'] = ari
    if 'f' in eval_metric:
        f = metrics.fowlkes_mallows_score(label, pred)
        results[f'{phase}_f'] = f
    if 'acc' in eval_metric:
        n_clusters = len(set(label))
        if n_clusters == len(set(pred)):
            pred_adjusted = get_y_preds(label, pred, n_clusters=n_clusters)
            acc = metrics.accuracy_score(pred_adjusted, label)
        else:
            acc = 0.
            warnings.warn('TODO: the number of classes is not equal...')
        results[f'{phase}_acc'] = acc
    if confusion_matrix_file is not None:
        match = _hungarian_match(pred, label, preds_k=num_classes, targets_k=num_classes)
        reordered_preds = np.zeros(num_elems, dtype=pred.dtype)
        for pred_i, target_i in match:
            reordered_preds[pred == int(pred_i)] = int(target_i)

        confusion_matrix(reordered_preds, label, class_names, confusion_matrix_file)
    return results


def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))
    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    cluster_labels = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_labels[i] = indices[i][1]
    return cluster_labels


def get_y_preds(y_true, cluster_assignments, n_clusters):
    """
    Computes the predicted labels, where label assignments now
    correspond to the actual labels in y_true (as estimated by Munkres)
    cluster_assignments:    array of labels, outputted by kmeans
    y_true:                 true labels
    n_clusters:             number of clusters in the dataset
    returns:    a tuple containing the accuracy and confusion matrix,
                in that order
    """
    confusion_matrix = metrics.confusion_matrix(y_true, cluster_assignments, labels=None)
    # compute accuracy based on optimal 1:1 assignment of clusters to labels
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)

    if np.min(cluster_assignments) != 0:
        cluster_assignments = cluster_assignments - np.min(cluster_assignments)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred


def confusion_matrix(predictions, gt, class_names, output_file=None):
    # Plot confusion_matrix and store result to output_file
    import sklearn.metrics
    import matplotlib.pyplot as plt
    confusion_matrix = sklearn.metrics.confusion_matrix(gt, predictions)
    confusion_matrix = confusion_matrix / np.sum(confusion_matrix, 1)

    fig, axes = plt.subplots(1)
    plt.imshow(confusion_matrix, cmap='Blues')
    axes.set_xticks([i for i in range(len(class_names))])
    axes.set_yticks([i for i in range(len(class_names))])
    axes.set_xticklabels(class_names, ha='right', fontsize=8, rotation=40)
    axes.set_yticklabels(class_names, ha='right', fontsize=8)

    for (i, j), z in np.ndenumerate(confusion_matrix):
        if i == j:
            axes.text(j, i, '%d' % (100 * z), ha='center', va='center', color='white', fontsize=6)
        else:
            pass

    plt.tight_layout()
    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


@torch.no_grad()
def _hungarian_match(flat_preds, flat_targets, preds_k, targets_k):
    # Based on implementation from IIC
    num_samples = flat_targets.shape[0]

    assert (preds_k == targets_k)  # one to one
    num_k = preds_k
    num_correct = np.zeros((num_k, num_k))

    for c1 in range(num_k):
        for c2 in range(num_k):
            # elementwise, so each sample contributes once
            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
            num_correct[c1, c2] = votes

    # num_correct is small
    match = linear_sum_assignment(num_samples - num_correct)
    match = np.array(list(zip(*match)))

    # return as list of tuples, out_c to gt_c
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))

    return res
