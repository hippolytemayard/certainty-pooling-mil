from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


class bag_dataset(Dataset):
    def __init__(self, list_id, list_target, data_root, random_selection=False):
        self.list_id = list_id
        self.list_target = list_target
        self.random_selection = random_selection
        self.data_root = data_root

    def __len__(self):
        return len(self.list_id)

    def __getitem__(self, idx):
        path = self.data_root / str(self.list_id[idx] + ".npy")
        target = self.list_target[idx]

        data = np.load(path)
        x = data[:, 3:]

        if self.random_selection:
            n = 128
            if n > x.shape[0]:
                return x, target
            else:
                index = np.random.choice(x.shape[0], n, replace=False)
                x = x[index]

        return x, target


class bag_dataset_test(Dataset):
    def __init__(self, list_id, data_root):
        self.list_id = list_id
        self.data_root = data_root

    def __len__(self):
        return len(self.list_id)

    def __getitem__(self, idx):
        path = self.data_root / str(self.list_id[idx] + ".npy")
        # path = "/data/data/data-resnet-only/test_input/resnet_features/" + self.list_id[idx] + ".npy"

        data = np.load(path)
        x = data[:, 3:]
        # print(x.shape)

        return x
