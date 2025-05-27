import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, OneCycleLR


def get_scheduler(optimizer, params):
    if params['scheduler'] == 'none':
        scheduler = None
    elif params['scheduler'] == 'warmup':
        scheduler = get_inverse_sqrt_scheduler(optimizer, params)
    elif params['scheduler'] == 'cosine':
        scheduler = get_cosine_annealing_scheduler(optimizer, params)
    else:
        raise NotImplementedError("The scheduler is not implemented.")

    return scheduler


# warmup
def get_inverse_sqrt_scheduler(optimizer, params):
    warmup_steps = params['warmup_steps']
    d_model = params['hidden_dim']

    def lr_lambda(step):
        if step == 0:
            return 0.0
        return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))

    return LambdaLR(optimizer, lr_lambda)


def get_cosine_annealing_scheduler(optimizer, params):
    T_max = params['epochs']
    eta_min = params['eta_min']
    return CosineAnnealingLR(optimizer, T_max, eta_min=eta_min)
