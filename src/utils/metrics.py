# src/utils/metrics.py

import torch
import torch.nn as nn


def mse_loss(pred, target):
    return nn.MSELoss()(pred, target)


def mae_loss(pred, target):
    return nn.L1Loss()(pred, target)


def evaluate(model, dataloader, device, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_seq, target_seq = batch
            adj_matrices = input_seq["adj_matrices"].to(device)  # (batch, T, N, N)
            features = input_seq["features"].to(device)  # (batch, T, 6)
            targets = target_seq.to(device)  # (batch, m, 6)

            predictions = model(adj_matrices, features)  # (batch, m, 6)
            loss = loss_fn(predictions, targets)
            total_loss += loss.item() * adj_matrices.size(0)
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss
