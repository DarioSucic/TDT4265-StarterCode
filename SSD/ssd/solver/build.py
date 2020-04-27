import torch
from .lr_scheduler import WarmupMultiStepLR


def make_optimizer(cfg, model, lr=None):
    lr = cfg.SOLVER.BASE_LR if lr is None else lr
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.SOLVER.WEIGHT_DECAY)


def make_lr_scheduler(cfg, optimizer, milestones=None):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=40, factor=0.99, cooldown=10
    )