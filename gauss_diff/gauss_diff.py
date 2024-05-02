import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def extract(input, t, x):
    shape = x.shape
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)

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

# Forward functions
def q_sample(y, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, t, noise=None, fq_x=None):

    if noise is None:
        noise = torch.randn_like(y).to(y.device)
    sqrt_alpha_bar_t = extract(alphas_bar_sqrt, t, y)
    sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t, y)
    # q(y_t | y_0, x)
    if fq_x is None:
        y_t = sqrt_alpha_bar_t * y + sqrt_one_minus_alpha_bar_t * noise
    else:
        y_t = sqrt_alpha_bar_t * y + (1 - sqrt_alpha_bar_t) * fq_x + sqrt_one_minus_alpha_bar_t * noise
    return y_t

def ddim_sample_loop(model, w, timesteps, y_dim, ddim_alphas, ddim_alphas_prev, ddim_sigmas, stochastic=True, x_embed=None):
    device = next(model.parameters()).device
    batch_size = w.shape[0]

    y_t = stochastic * torch.randn_like(torch.zeros([batch_size, y_dim])).to(device)

    # intermediates = {'y_inter': [y_t], 'pred_y0': [y_t]}
    time_range = np.flip(timesteps)
    total_steps = timesteps.shape[0]
    # print(f"Running DDIM Sampling with {total_steps} timesteps")

    for i, step in enumerate(time_range):
        index = total_steps - i - 1
        t = torch.full((batch_size,), step, device=device, dtype=torch.long)

        y_t, pred_y0 = ddim_sample_step(model, y_t, w, t, index, ddim_alphas,
                                        ddim_alphas_prev, ddim_sigmas, x_embed)

        # intermediates['y_inter'].append(y_t)
        # intermediates['pred_y0'].append(pred_y0)

    return y_t

def ddim_sample_step(model, y_t, w, t, index, ddim_alphas, ddim_alphas_prev, ddim_sigmas, x_embed=None):
    batch_size = w.shape[0]
    device = next(model.parameters()).device
    e_t = model(y_t, t, w=w, x_embed=x_embed).to(device).detach()

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

class GaussianDiffusion(nn.Module):
    def __init__(self, args, num_timesteps=1000, num_classes=10, w_dim=128, eps=20, device='cuda:1', beta_schedule='cosine',
                 ddim_num_steps=10):
        super().__init__()
        self.device = args.device
        self.num_timesteps = num_timesteps
        self.n_class = num_classes
        betas = make_beta_schedule(schedule=beta_schedule, num_timesteps=self.num_timesteps, start=0.0001, end=0.02)
        betas = self.betas = betas.float().to(self.device)
        self.betas_sqrt = torch.sqrt(betas)
        alphas = 1.0 - betas
        self.alphas = alphas
        self.one_minus_betas_sqrt = torch.sqrt(alphas)
        self.alphas_cumprod = alphas.cumprod(dim=0)
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_cumprod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_cumprod)
        alphas_cumprod_prev = torch.cat([torch.ones(1).to(self.device), self.alphas_cumprod[:-1]], dim=0)
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.posterior_mean_coeff_1 = (betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coeff_2 = (torch.sqrt(alphas) * (1 - alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        posterior_variance = (betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_variance = posterior_variance
        self.logvar = betas.log()
        

        self._denoise_fn = ConditionalModel(self.num_timesteps, y_dim=self.n_class, w_dim=w_dim,
                                      eps=eps, guidance=False, x_dim=768).to(self.device)

        self.ddim_num_steps = ddim_num_steps
        self.make_ddim_schedule(ddim_num_steps)

    
    def make_ddim_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0.):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.num_timesteps)

        assert self.alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.device)

        self.register_buffer('sqrt_alphas_cumprod', to_torch(torch.sqrt(self.alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(torch.sqrt(1. - self.alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(torch.log(1. - self.alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(torch.sqrt(1. / self.alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(torch.sqrt(1. / self.alphas_cumprod - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=self.alphas_cumprod,
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', torch.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)
    

    def forward_t(self, y, w, t, x_embed):
        w = w.to(self.device)

        e = torch.randn_like(y.float()).to(y.device)
        y_t = q_sample(y, self.alphas_bar_sqrt,
                             self.one_minus_alphas_bar_sqrt, t, noise=e, )

        output = self._denoise_fn(y_t, t, w, x_embed)

        return output, e
    
    def reverse_ddim(self, w, x_embed, stochastic=True):

        w = w.to(self.device)
        with torch.no_grad():

            label_t_0 = ddim_sample_loop(self._denoise_fn, w, self.ddim_timesteps, self.n_class, self.ddim_alphas,
                                         self.ddim_alphas_prev, self.ddim_sigmas, stochastic=stochastic,
                                         x_embed=x_embed)

        return label_t_0



class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, t):
        out = self.lin(x)
        gamma = self.embed(t)
        out = gamma.view(-1, self.num_out) * out
        return out
    
class ConditionalModel(nn.Module):
    def __init__(self, n_steps, y_dim=10, w_dim=128, eps=20, guidance=True, x_dim=768):
        super(ConditionalModel, self).__init__()
        n_steps = n_steps + 1
        self.y_dim = y_dim
        self.guidance = guidance
        self.norm = nn.BatchNorm1d(w_dim)

        # Unet
        if self.guidance:
            self.lin1 = ConditionalLinear(y_dim + x_dim, w_dim, n_steps)
        else:
            self.lin1 = ConditionalLinear(y_dim, x_dim, n_steps)
            # self.lin1 = ConditionalLinear(y_dim, w_dim, n_steps)

        # self.unetnorm1 = nn.BatchNorm1d(w_dim)
        # self.lin2 = ConditionalLinear(w_dim, w_dim, n_steps)
        # self.unetnorm2 = nn.BatchNorm1d(w_dim)
        # self.lin3 = ConditionalLinear(w_dim, w_dim, n_steps)
        # self.unetnorm3 = nn.BatchNorm1d(w_dim)
        # self.lin4 = nn.Linear(w_dim, y_dim)

        self.unetnorm1 = nn.BatchNorm1d(x_dim)
        self.lin2 = ConditionalLinear(x_dim, x_dim, n_steps)
        self.unetnorm2 = nn.BatchNorm1d(x_dim)
        self.lin3 = ConditionalLinear(x_dim, x_dim, n_steps)
        self.unetnorm3 = nn.BatchNorm1d(x_dim)
        self.lin4 = nn.Linear(x_dim, y_dim)

    def forward(self, y, t, w, x_embed):

        # x_embed = self.encoder_x(x)
        w = self.norm(w)
        if self.guidance:
            y = torch.cat([y, x_embed], dim=-1)

        y = self.lin1(y, t)
        y = self.unetnorm1(y)
        y = F.softplus(y)
        y = y * x_embed
        y = self.lin2(y, t)
        y = self.unetnorm2(y)
        y = F.softplus(y)
        y = self.lin3(y, t)
        y = self.unetnorm3(y)
        y = F.softplus(y)
        return self.lin4(y)

# class ConditionalModel(nn.Module):
#     def __init__(self, n_steps, y_dim=10, w_dim=128, eps=20, guidance=True, x_dim=768):
#         super(ConditionalModel, self).__init__()
#         n_steps = n_steps + 1
#         self.y_dim = y_dim
#         self.guidance = guidance
#         self.norm = nn.BatchNorm1d(w_dim)
#         self.dropout = nn.Dropout(0.5)

#         # Unet
#         if self.guidance:
#             self.lin1 = ConditionalLinear(y_dim + y_dim + x_dim, w_dim, n_steps)
#         else:
#             self.lin1 = ConditionalLinear(y_dim, w_dim, n_steps)

#         self.unetnorm1 = nn.BatchNorm1d(w_dim)
#         self.lin2 = ConditionalLinear(w_dim, w_dim // (eps//4), n_steps)
#         self.unetnorm2 = nn.BatchNorm1d(w_dim // (eps//4))
#         self.lin3 = ConditionalLinear(w_dim // (eps//4), w_dim // (eps//2), n_steps)
#         self.unetnorm3 = nn.BatchNorm1d(w_dim // (eps//2))
#         self.lin4 = ConditionalLinear(w_dim // (eps//2), w_dim // (eps//2), n_steps)
#         self.unetnorm4 = nn.BatchNorm1d(w_dim // (eps//2))

#         self.lin5 = nn.Linear(w_dim // (eps//2), y_dim)

#     def forward(self, y, t, w, x_embed):
#         w = self.norm(w)
#         if self.guidance:
#             y = torch.cat([y, x_embed], dim=-1)   

#         y = self.lin1(y, t)
#         y = self.unetnorm1(y)
#         y = F.softplus(y)
#         y = self.dropout(y)
#         y = w * y
#         y = self.lin2(y, t)
#         y = self.unetnorm2(y)
#         y = F.softplus(y)
#         y = self.dropout(y)
#         y = self.lin3(y, t)
#         y = self.unetnorm3(y)
#         y = F.softplus(y)
#         y = self.dropout(y)
#         y = self.lin4(y, t)
#         y = self.unetnorm4(y)
#         y = F.softplus(y)
#         y = self.dropout(y)
#         return self.lin5(y)