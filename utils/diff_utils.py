import math
import torch
import numpy as np
from tqdm import tqdm

def adjust_learning_rate(optimizer, epoch, warmup_epochs=100, n_epochs=1000, lr_input=0.001):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < warmup_epochs:
        lr = lr_input * epoch / warmup_epochs
    else:
        lr = 0.0 + lr_input * 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs) / (n_epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

def make_beta_schedule(schedule="linear", num_timesteps=1000, start=1e-5, end=1e-2):
    if schedule == "linear":
        betas = torch.linspace(start, end, num_timesteps)
    elif schedule == "const":
        betas = end * torch.ones(num_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, num_timesteps) ** 2
    elif schedule == "jsd":
        betas = 1.0 / torch.linspace(num_timesteps, 1, num_timesteps)
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, num_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    elif schedule == "cosine" or schedule == "cosine_reverse":
        max_beta = 0.999
        cosine_s = 0.008
        betas = torch.tensor(
            [min(1 - (math.cos(((i + 1) / num_timesteps + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2) / (
                    math.cos((i / num_timesteps + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2), max_beta) for i in
             range(num_timesteps)])
    elif schedule == "cosine_anneal":
        betas = torch.tensor(
            [start + 0.5 * (end - start) * (1 - math.cos(t / (num_timesteps - 1) * math.pi)) for t in
             range(num_timesteps)])
    return betas











def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps):
    if ddim_discr_method == 'uniform':
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == 'quad':
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
    else:
        raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

    # assert ddim_timesteps.shape[0] == num_ddim_timesteps
    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    steps_out = ddim_timesteps + 1
    # print(f'Selected timesteps for ddim sampler: {steps_out}')

    return steps_out


def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta):
    # select alphas for computing the variance schedule
    device = alphacums.device
    alphas = alphacums[ddim_timesteps]
    alphas_prev = torch.tensor([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist()).to(device)

    sigmas = eta * torch.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    # print(f'Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}')
    # print(f'For the chosen value of eta, which is {eta}, '
    #       f'this results in the following sigma_t schedule for ddim sampler {sigmas}')
    return sigmas, alphas, alphas_prev





def ddim_sample_step(model, y_t, w, t, index, ddim_alphas, ddim_alphas_prev, ddim_sigmas, noisy_y=None):
    batch_size = w.shape[0]
    device = next(model.parameters()).device
    e_t = model(y_t, t, noisy_y=noisy_y, w=w).to(device).detach()

    sqrt_one_minus_alphas = torch.sqrt(1. - ddim_alphas)
    # select parameters corresponding to the currently considered timestep
    a_t = torch.full([batch_size, 1], ddim_alphas[index], device=device)
    a_t_m_1 = torch.full([batch_size, 1], ddim_alphas_prev[index], device=device)
    sigma_t = torch.full([batch_size, 1], ddim_sigmas[index], device=device)
    sqrt_one_minus_at = torch.full([batch_size, 1], sqrt_one_minus_alphas[index], device=device)

    # direction pointing to x_t
    dir_y_t = (1. - a_t_m_1 - sigma_t ** 2).sqrt() * e_t
    noise = sigma_t * torch.randn_like(y_t).to(device)

    # reparameterize x_0
    # if noisy_y is None:
    #     y_0_reparam = (y_t - sqrt_one_minus_at * e_t) / a_t.sqrt()
    #     y_t_m_1 = a_t_m_1.sqrt() * y_0_reparam + dir_y_t + noise
    # else:
    #     y_0_reparam = (y_t - (1 - a_t.sqrt()) * noisy_y - sqrt_one_minus_at * e_t) / a_t.sqrt()
    #     y_t_m_1 = a_t_m_1.sqrt() * y_0_reparam + (1 - a_t_m_1.sqrt()) * noisy_y + dir_y_t #+ noise
    
    y_0_reparam = (y_t - sqrt_one_minus_at * e_t) / a_t.sqrt()
    y_t_m_1 = a_t_m_1.sqrt() * y_0_reparam + dir_y_t + noise

    return y_t_m_1, y_0_reparam


def y_0_reparam(model, x, y, y_0_hat, y_T_mean, t, one_minus_alphas_bar_sqrt):
    """
    Obtain y_0 reparameterization from q(y_t | y_0), in which noise term is the eps_theta prediction.
    Algorithm 2 Line 4 in paper.
    """
    device = next(model.parameters()).device
    sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t, y)
    sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()
    eps_theta = model(x, y, t, y_0_hat).to(device).detach()
    # y_0 reparameterization
    y_0_reparam = 1 / sqrt_alpha_bar_t * (
            y - (1 - sqrt_alpha_bar_t) * y_T_mean - eps_theta * sqrt_one_minus_alpha_bar_t).to(device)
    return y_0_reparam
