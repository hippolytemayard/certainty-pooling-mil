from copy import deepcopy

import numpy as np
import torch.nn as nn
import torch


def certainty_pooling(model: nn.Module, x: np.ndarray, n: int, device, epsilon: float = 1.0e-3) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    model : nn.Module
        _description_
    x : np.ndarray
        _description_
    n : int
        _description_
    epsilon : float, optionals
        _description_, by default 1.e-3

    Returns
    -------
    _type_
        _description_
    """
    input = torch.tensor(x).float().to(device)

    model.eval()
    h = model(input)

    model.train()
    model = model.to(device)

    # TODO: faster training
    # for i in range(n):
    #    output = model(input)
    #    mc_dropout.append(output.squeeze(-1).cpu().detach().numpy())

    mc_dropout = np.asarray([model(input).squeeze(-1).cpu().detach().numpy() for _ in range(n)])

    std_vector = 1 / (np.std(mc_dropout, axis=0) + epsilon)

    certainty_weighted_output = deepcopy(h.cpu().detach().numpy())

    for i, std in enumerate(std_vector):
        certainty_weighted_output[i] *= std

    argmax = np.argmax(certainty_weighted_output)

    return x[argmax]
