import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from certainty_pooling import certainty_pooling


def train_loop(train_pooling_loader, model, criterion, optimizer, epoch, device, n=100):
    running_loss = 0.0
    for i, sample in enumerate(train_pooling_loader, 0):
        inputs, targets = sample

        inputs_pooling = []
        with torch.no_grad():
            for input in inputs:
                inputs_pooling.append(certainty_pooling(model, input, n, device, epsilon=1.0e-3))

        model.train()
        inputs_pooling = np.asarray(inputs_pooling)
        inputs = torch.tensor(inputs_pooling).float().cuda()
        targets = torch.tensor(targets).float().cuda()

        optimizer.zero_grad()

        outputs = model(inputs)
        outputs = outputs.squeeze(-1)

        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if i % 10 == 9:
            print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}")

    return running_loss
