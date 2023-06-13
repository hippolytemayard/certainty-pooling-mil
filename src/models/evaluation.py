from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from certainty_pooling import certainty_pooling
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader


def evaluation(model: nn.Module, loader: DataLoader, device, n: int = 100) -> Tuple[int]:
    """evaluation of the model on the validation set using a Loader.

    Parameters
    ----------
    model : nn.Module
        model to evaluate
    loader : DataLoader
        The validation set
    n : int, optional
        number of MC iteration, by default 100

    Returns
    -------
    Tuple[int]
        returns the following metrics: auc, accuracy, recall, precision
    """
    model.eval()

    predictions_list = []
    target_list = []

    for i, sample in enumerate(loader, 0):
        inputs, targets = sample
        inputs_pooling = []
        with torch.no_grad():
            for input in inputs:
                certainty_pooling_ = certainty_pooling(model, input, n, device, epsilon=1.0e-3)

                inputs_pooling.append(certainty_pooling_)

            inputs_pooling = np.asarray(inputs_pooling)
            inputs = torch.tensor(inputs_pooling).float().cuda()
            targets = torch.tensor(targets).float().cuda()

            outputs = model(inputs)

            target_list.append(targets.cpu().detach().numpy())
            predictions_list.append(outputs.cpu().detach().numpy())

    predictions = np.concatenate(predictions_list).ravel()
    y_val = np.concatenate(target_list).ravel()

    # print(predictions.shape)
    # print(y_val.shape)

    fpr, tpr, thresholds = metrics.roc_curve(y_val, predictions)
    auc = metrics.auc(fpr, tpr)

    threshold = 0.5
    predictions[predictions >= threshold] = 1
    predictions[predictions < threshold] = 0

    confusion_mat = confusion_matrix(y_val, predictions)
    tn, fp, fn, tp = confusion_mat.ravel()

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = (tp + tn) / (tn + fp + fn + tp)

    print(f"\nAUC : {auc} , accuracy : {accuracy} , reccall = {recall} , precision = {precision}\n")

    print(confusion_mat)

    return auc, accuracy, recall, precision
