import numpy as np
from omegaconf import OmegaConf


def collate_fn(batch):
    return tuple(zip(*batch))


def collate_fn_submission(batch):
    return tuple(batch)


def load_yaml(path: str):
    yaml_file = OmegaConf.load(path)
    return yaml_file


def compute_class_freqs(labels):
    labels = np.array(labels)

    N = labels.shape[0]

    positive_frequencies = np.sum(labels, axis=0) / N
    negative_frequencies = 1 - positive_frequencies

    return positive_frequencies, negative_frequencies
