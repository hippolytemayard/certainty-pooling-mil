from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torch.nn as nn
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.models.certainty_pooling import certainty_pooling, certainty_pooling_btach


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

    predictions_list = []
    target_list = []

    for i, sample in enumerate(loader, 0):
        inputs, targets = sample

        with torch.no_grad():
            inputs_pooling = np.asarray(
                [certainty_pooling(model, input, n, device, epsilon=1.0e-3) for input in inputs]
            )
            model.eval()
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

    cf_matrix = confusion_matrix(y_val, predictions)
    tn, fp, fn, tp = cf_matrix.ravel()

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = (tp + tn) / (tn + fp + fn + tp)

    print(f"\nAUC : {auc} , accuracy : {accuracy} , reccall = {recall} , precision = {precision}\n")

    print(cf_matrix)

    df_cm = pd.DataFrame(cf_matrix, index=["gt 0", "gt 1"], columns=["pred 0", "pred 1"])
    plt.figure(figsize=(12, 7))
    fig = sn.heatmap(df_cm, annot=True).get_figure()

    return auc, accuracy, recall, precision, fig


def evaluation_tile(model: nn.Module, loader: DataLoader, device, n: int = 100) -> Tuple[int]:
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

        with torch.no_grad():
            inputs = torch.tensor(inputs).float().cuda()
            # inputs_pooling = certainty_pooling(model, input, n, device, epsilon=1.0e-3)
            # nputs = torch.tensor(inputs_pooling).float().cuda()
            targets = torch.tensor(targets).float().cuda()
            model.eval()
            outputs = model(inputs)

            target_list.append(targets.cpu().detach().numpy())
            predictions_list.append(outputs.cpu().detach().numpy())

    # print(len(predictions_list))
    predictions = np.concatenate(predictions_list, axis=1).ravel()
    y_val = np.concatenate(target_list).ravel()

    fpr, tpr, thresholds = metrics.roc_curve(y_val, predictions)
    auc = metrics.auc(fpr, tpr)

    threshold = 0.5
    predictions[predictions >= threshold] = 1
    predictions[predictions < threshold] = 0

    cf_matrix = confusion_matrix(y_val, predictions)
    tn, fp, fn, tp = cf_matrix.ravel()

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = (tp + tn) / (tn + fp + fn + tp)

    print(
        f"\nAUC_instance : {auc} , accuracy_instance : {accuracy} , reccall_instance = {recall} , precision_instance = {precision}\n"
    )
    print(cf_matrix)

    df_cm = pd.DataFrame(cf_matrix, index=["gt 0", "gt 1"], columns=["pred 0", "pred 1"])
    plt.figure(figsize=(12, 7))
    fig = sn.heatmap(df_cm, annot=True).get_figure()

    return auc, accuracy, recall, precision, fig


def evaluation_batch(model: nn.Module, loader: DataLoader, device, n: int = 100) -> Tuple[int]:
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

    predictions_list = []
    target_list = []

    for i, sample in enumerate(loader, 0):
        inputs, targets = sample

        with torch.no_grad():

            pooling_input, pooling_target = certainty_pooling_btach(model=model, batch=sample, T=n, device=device)


            model.eval()
            #inputs = torch.tensor(inputs_pooling).float().cuda()
            targets = torch.tensor(pooling_target).float().cuda()

            outputs = model(pooling_input)

            target_list.append(targets.cpu().detach().numpy())
            predictions_list.append(outputs.cpu().detach().numpy())

    predictions = np.concatenate(predictions_list).ravel()
    y_val = np.concatenate(target_list).ravel()


    fpr, tpr, thresholds = metrics.roc_curve(y_val, predictions)
    auc = metrics.auc(fpr, tpr)

    threshold = 0.5
    predictions[predictions >= threshold] = 1
    predictions[predictions < threshold] = 0

    cf_matrix = confusion_matrix(y_val, predictions)
    tn, fp, fn, tp = cf_matrix.ravel()

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = (tp + tn) / (tn + fp + fn + tp)

    print(f"\nAUC : {auc} , accuracy : {accuracy} , reccall = {recall} , precision = {precision}\n")

    print(cf_matrix)

    df_cm = pd.DataFrame(cf_matrix, index=["gt 0", "gt 1"], columns=["pred 0", "pred 1"])
    plt.figure(figsize=(12, 7))
    fig = sn.heatmap(df_cm, annot=True).get_figure()

    return auc, accuracy, recall, precision, fig
