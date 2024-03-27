import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput
from utils.utils import logits_projection, scale

class SimplexDiffusionPipelineOutput(BaseOutput):
    simplex: np.ndarray
    logits: np.ndarray
    loss: np.ndarray


class SimplexDDPM(DiffusionPipeline):
    def __init__(
            self,
            model,
            scheduler,
            simplex_value
    ):
        super().__init__()
        self.register_modules(model=model, scheduler=scheduler)
        self.simpelx_value = simplex_value


    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[torch.Generator] = None
    ):
        simplex_shape = (batch_size, self.model.args.num_calsses)
        simplex = self.simplex_value * torch.randn(simplex_shape, generator=generator, device=self.device)

        logits_projection_fct = lambda x: logits_projection(x, self.simplex_value)

        for t in self.progress_bar(self.scheduler.timesteps):
            t_scaled = scale(t, len(self.scheduler))

            model_output = self.model(simplex=simplex, timestpes=t_scaled)

            model_output_logits = model_output.logits

            projected_logits = logits_projection_fct(model_output_logits)

            noise = self.simpelx_value * torch.randn(simplex_shape, generator=generator, device=self.device)
            simplex = self.scheduler.step(projected_logits, t, noise, generator=generator).prev_sample

        return SimplexDiffusionPipelineOutput(simplex=simplex, logits=model_output_logits, loss=model_output.loss)


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
    def __init__(self, n_steps, y_dim=10, w_dim=128, feature_dim=768, guidance=True):
        super(ConditionalModel, self).__init__()
        n_steps = n_steps + 1
        self.y_dim = y_dim
        self.guidance = guidance
        self.norm = nn.BatchNorm1d(feature_dim)

        # Unet
        if self.guidance:
            self.lin1 = ConditionalLinear(y_dim + y_dim + w_dim, feature_dim, n_steps)
        else:
            self.lin1 = ConditionalLinear(y_dim, feature_dim, n_steps)

        self.unetnorm1 = nn.BatchNorm1d(feature_dim)
        self.lin2 = ConditionalLinear(feature_dim, feature_dim, n_steps)
        self.unetnorm2 = nn.BatchNorm1d(feature_dim)
        self.lin3 = ConditionalLinear(feature_dim, feature_dim, n_steps)
        self.unetnorm3 = nn.BatchNorm1d(feature_dim)
        self.lin4 = nn.Linear(feature_dim, y_dim)

    def forward(self, y, noisy_y, w, x_embed, t):
        x_embed = self.norm(x_embed)
        if self.guidance:
            y = torch.cat([y, noisy_y,  w], dim=-1)

        y = self.lin1(y, t)
        y = self.unetnorm1(y)
        y = F.softplus(y)
        y = x_embed * y
        y = self.lin2(y, t)
        y = self.unetnorm2(y)
        y = F.softplus(y)
        y = self.lin3(y, t)
        y = self.unetnorm3(y)
        y = F.softplus(y)
        return self.lin4(y)