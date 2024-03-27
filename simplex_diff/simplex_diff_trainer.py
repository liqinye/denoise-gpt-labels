from transformers import Trainer
import torch
from torch import nn
from transformers import AdamW
from tqdm import tqdm
from utils.utils import convert_to_simplex, scale


class DiffusionTrainer(nn.Module):
    def __init__(
        self,
        model,
        noise_scheduler,
        inference_noise_scheduler,
        diffusion_args
    ):
        super().__init__()
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.args = diffusion_args
        self.inference_noise_scheduler = inference_noise_scheduler
        self.create_optimizer()
    

    def create_optimizer(self):
        opt_model = self.model

        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight", "timestep_embed.weight", "timestep_embed.bias"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in opt_model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in opt_model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)
        return self.optimizer

    def train(self, train_dataloader, valid_dataloader):
        self.model.train()
        
        print('Diffusion training start')

        for epoch in range(self.args.epochs):
            with tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'train diffusion epoch {epoch}', ncols=120) as pbar:
                for i, data_batch in pbar:
                    [w, noisy_y, y] = data_batch
                    y = y.squeeze().to(self.args.device)
                    noisy_y = noisy_y.to(self.args.device)
                    w = w.to(self.args.device).squeeze()

                    # convert true label into simplex representation
                    simplex = convert_to_simplex(y, self.args.simplex_value)
                    noise = self.args.simplex_value * torch.randn(simplex.shape, device=simplex.device, dtype=simplex.dtype)
                    batch_size = simplex.shape[0]
                    # Sample a random timestep for each simplex label representation
                    timesteps = torch.randint(0, len(self.noise_scheduler), (batch_size,), device=simplex.device, dtype=torch.int64)
                    # Adds noise to each simplex representation (Forward diffusion process)
                    noisy_simplex = self.noise_scheduler.add_noise(simplex, noise, timesteps)
                    timesteps = scale(timesteps, len(self.noise_scheduler))

        


