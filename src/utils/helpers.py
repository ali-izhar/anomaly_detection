# src/utils/helpers.py

import torch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
