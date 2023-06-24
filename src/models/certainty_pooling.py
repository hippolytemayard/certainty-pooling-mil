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
    #print(input.shape)
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



def certainty_pooling_btach(model: nn.Module, batch: np.ndarray, T: int, device, epsilon: float = 1.0e-3) -> torch.Tensor:


    batch_features, batch_targets = batch

    input_ = torch.cat([torch.tensor(i) for i in batch_features])
    h = model(input_.to(device).float())

    model.train()
    cat = torch.cat([torch.tensor(i) for i in batch_features])

    mc_output = torch.cat([model(cat.to(device).float()) for _ in range(T)], dim=0)
    mc_output = mc_output.reshape(T,cat.shape[0],1).squeeze(-1)

    output_split = torch.split(mc_output, split_size_or_sections=[i.shape[0] for i in batch_features], dim=1)
    std_output = torch.cat([1 / i.std(0) for i in output_split])

    z_m_global = (std_output * h.squeeze(-1))
    z_m = torch.split(z_m_global, split_size_or_sections=[i.shape[0] for i in batch_features], dim=0)
    k_star = torch.tensor([z_.argmax() for z_ in z_m])

    pooling_input = torch.cat([torch.tensor(bag[k_star[idx]]).unsqueeze(0) for idx, bag in enumerate(batch_features)], dim=0)
    pooling_target = torch.tensor(batch_targets)

    return pooling_input, pooling_target