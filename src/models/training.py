import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.certainty_pooling import certainty_pooling


def train_loop(train_pooling_loader, model, criterion, optimizer, epoch, device, n=100):
    running_loss = 0.0
    for i, sample in enumerate(train_pooling_loader, 0):
        inputs, targets = sample

        with torch.no_grad():
            inputs_pooling = np.asarray(
                [certainty_pooling(model, input, n, device, epsilon=1.0e-3) for input in inputs]
            )

        model.train()

        inputs = torch.tensor(inputs_pooling).float().to(device)
        targets = torch.tensor(targets).float().to(device)

        optimizer.zero_grad()

        outputs = model(inputs).squeeze(-1)

        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if i % 10 == 9:
            print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}")

    return running_loss
