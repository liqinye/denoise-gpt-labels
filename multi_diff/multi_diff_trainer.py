import torch
import time
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.optim import Adam
from tqdm import tqdm

from utils.ema import EMA
from multi_diff.multi_diff import MultinomialDiffusion
from utils.utils import adjust_learning_rate

def multinomial_kl(log_prob1, log_prob2):
    kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
    return kl

class Multinomial_Trainer():
    def __init__(self, args, train_dataset, valid_dataloader, test_dataloader, w_dim, best_plm):
        self.args = args
        self.train_dataset = train_dataset
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.best_plm = best_plm

        self.multi_diffs = nn.ModuleList()
        self.EMAs = [EMA(mu=0.9999) for _ in range(self.args.n_model)]

        for i in range(self.args.n_model):
            multi_diff = MultinomialDiffusion(timesteps=self.args.num_timesteps, \
                        num_classes=self.args.num_classes, w_dim=w_dim)
            multi_diff.to(self.args.device)
            self.EMAs[i].register(multi_diff._denoise_fn)
            self.multi_diffs.append(multi_diff)
        
        self.optimizer = Adam(self.multi_diffs.parameters(), lr=self.args.lr, \
                            weight_decay=0.0, betas=(0.9, 0.999), amsgrad=False, eps=1e-08)
                            
    def sample_uncertain_y(self, y, weights, uncertain_idx):

        # filter out 0.0 in weights
        weights = [weight[weight != 0.0] for weight in weights]

        # sample a prior according to the probability
        # batch_size, model_branches
        sample_y = torch.zeros(len(weights))

        for i, weight in enumerate(weights):
            # sample_index = torch.multinomial(torch.tensor(weight), num_samples=1, replacement=True)
            sample_index = torch.multinomial(weight.clone().detach(), num_samples=1, replacement=True)
            sample_y[i] = y[i][sample_index]

        return sample_y.to(dtype=torch.long).to(self.args.device), weights

    
    def update_dataset(self, step, update_weight):
        batch_size = self.args.train_batch_size
        inputs, y, weights, uncertain, labels, true_labels = self.train_dataset[:]
        weights[step*batch_size:(step+1)*batch_size] = update_weight
        self.train_dataset = TensorDataset(inputs, y, weights, uncertain, labels, true_labels)

    def train(self):
        max_accuracy = 0.0
        acc_list = []

        print('Multinomial Diffusion Training Start')
        for epoch in range(self.args.diff_epoches):
            train_sampler = SequentialSampler(self.train_dataset)
            train_loader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

            self.multi_diffs.train()

            if epoch <= self.args.warmup_epochs:
                self.lambda_t = 0
            else:
                self.lambda_t = self.args.lambda_t

            with tqdm(enumerate(train_loader), total=len(train_loader), desc=f'train diffusion epoch {epoch}', ncols=120) as pbar:

                for step, data_batch in pbar:
                    adjust_learning_rate(self.optimizer, step / len(train_loader) + epoch, warmup_epochs=self.args.warmup_epochs, n_epochs=self.args.diff_epoches, lr_input=0.001)

                    [w, y, weights, uncertain_markers, noisy_y, true_labels] = data_batch
                    y = y.squeeze().to(self.args.device)
                    weights = weights.to(self.args.device)
                    uncertain_markers = uncertain_markers.to(self.args.device)
                    noisy_y = noisy_y.to(self.args.device)
                    w = w.to(self.args.device).squeeze()
                    w = torch.log(w.float().clamp(min=1e-30))
                    n = w.size(0)

                    total_diff_loss = 0.0
                    total_reg_loss = 0.0
                    prob_list = torch.zeros(self.args.n_model, n, self.args.num_classes)

                    for model_i in range(self.args.n_model):
                        y_model = y[:, model_i, :]
                        weights_model = weights[:, model_i, :]
                        uncertain_markers_model = uncertain_markers[:, model_i]
                        w_model = w[:, model_i, :]
                        uncertain_idx = torch.where(uncertain_markers_model==True)[0]
                        certain_idx = torch.where(uncertain_markers_model==False)[0]
                        
                        current_y_model = torch.zeros(y_model.size(0), device=self.args.device, dtype=torch.long) * -1
                        if uncertain_idx.numel() != 0 and (epoch > self.args.warmup_epochs):
                            uncertain_y_batch = y_model[uncertain_idx]

                            # filter ourt pad value -1
                            uncertain_y_batch = [y[y!=-1] for y in uncertain_y_batch]

                            uncertain_weights_batch = weights_model[uncertain_idx]

                            for sample_i in range(self.args.num_sample):
                                # sample the prior based on uncertain prior weight
                                sample_y, uncertain_weights = self.sample_uncertain_y(uncertain_y_batch, uncertain_weights_batch, uncertain_idx)

                                with torch.cuda.amp.autocast():
                                    self.multi_diffs[model_i].eval()
                                    logit, prob = self.multi_diffs[model_i].sample(sample_y, w[uncertain_idx, model_i, :])

                                # Update Uncertain Weights based on Model Feedback
                                # =================================================
                                pred_labels = torch.argmax(logit, dim=1)
                                for (i, idx) in enumerate(uncertain_idx):
                                    pred_y = pred_labels[i]
                                    # uncertain_y = torch.tensor(uncertain_y_batch[i], device=self.args.device)
                                    uncertain_y = uncertain_y_batch[i].clone().detach()
                                    # (num_class,)
                                    uncertain_weights = weights_model[idx]

                                    if pred_y in uncertain_y:
                                        # update the uncertain y weights
                                        pred_weights = uncertain_weights[pred_y].clone()
                                        update_pred_weights = pred_weights + ((1-pred_weights) / self.args.num_sample)
                                        uncertain_weights[pred_y] = update_pred_weights
                                        # normalize updated weights
                                        uncertain_weights = uncertain_weights / (update_pred_weights + (1-pred_weights))

                                        # update in original weights
                                        weights[idx, model_i] = uncertain_weights.to(self.args.device)
                                # =================================================

                            self.update_dataset(step, weights)
                            # Evaluate after weight updating
                            sample_y, uncertain_weights = self.sample_uncertain_y(uncertain_y_batch, weights[uncertain_idx,model_i, :], uncertain_idx)
                            
                            # update y_model for sample uncertain y
                            current_y_model[uncertain_idx] = sample_y.to(torch.long)

                        # certain data points
                        if certain_idx.numel() != 0:
                            certain_y_batch = y_model[certain_idx]
                            # filter out pad value -1
                            certain_y_batch = torch.tensor([y[y != -1] for y in certain_y_batch], device=self.args.device)
                            current_y_model[certain_idx] = certain_y_batch
                            current_y_model = current_y_model[current_y_model != -1]

                        with torch.cuda.amp.autocast():
                            self.multi_diffs[model_i].train()
                            weights_model = torch.tensor([weights_model[i, current_y_model[i]] for i in range(weights_model.size(0))], device=self.args.device)
                            diff_loss, prob = self.multi_diffs[model_i].log_prob(current_y_model, noisy_y, w_model, weights_model)
                            diff_loss = - diff_loss.sum() / (math.log(2) * n)
                        
                        prob_list[model_i] = prob
                        total_diff_loss += diff_loss

                    avg_pred = torch.mean(prob_list, dim=0)

                    for model_i in range(self.args.n_model):
                        temp_pred = prob_list[model_i]
                        reg_loss = multinomial_kl(avg_pred.squeeze(), temp_pred.squeeze())
                        total_reg_loss += torch.sum(reg_loss)

                    diff_loss, reg_loss = total_diff_loss/self.args.n_model, total_reg_loss/self.args.n_model
                    loss = diff_loss + self.lambda_t * reg_loss

                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.multi_diffs[model_i].parameters(), 1.0)
                    self.optimizer.step()

                    for model_i in range(self.args.n_model):
                        self.EMAs[model_i].update(self.multi_diffs[model_i]._denoise_fn)
            
            # validation & test
            if (epoch % 10 == 0 and epoch >= self.args.warmup_epochs) or epoch == self.args.diff_epoches-1:
                test_acc, plm_acc  = self.test()
                acc_list.append(test_acc)
                if test_acc >= max_accuracy:
                    torch.save(self.multi_diffs, f"best_multi_diffusion_{self.args.dataset}.pt")
                    print(f"Model saved, update best accuracy at Epoch {epoch}, test acc: {test_acc}")
                print(f"Epoch {epoch}: PLM acc: {plm_acc}, Denoising acc: {test_acc}")
                max_accuracy = max(max_accuracy, test_acc)
                
        print(acc_list)

    def test(self):
        self.multi_diffs.eval()
        start = time.time()
        with torch.no_grad():
            correct = 0
            plm_correct = 0
            all_sample = 0

            for test_batch_idx, data_batch in tqdm(enumerate(self.test_dataloader), total=len(self.test_dataloader), desc=f'Multinomial Diffusion Sampling...', ncols=100):
                input_ids, input_mask, target, w = data_batch 
                w = w.squeeze().to(self.args.device)
                w = torch.log(w.float().clamp(min=1e-30))
                target = target.squeeze().to(self.args.device)

                outputs = self.best_plm(input_ids, attention_mask=input_mask)
                logits = [output[0] for output in outputs]
                p_y_tilde = [F.softmax(logit, dim=-1).detach().cpu() for logit in logits]
                avg_p_y_tilde = torch.mean(torch.stack(p_y_tilde, dim=0), 0)
                _, plm_pred_labels = torch.max(avg_p_y_tilde, dim=-1)

                p_y_y_tilde_list = []
                for model_i in range(self.args.n_model):
                    p_y_bar_x_y_tilde = torch.zeros(target.size(0), self.args.num_classes, self.args.num_classes).to(self.args.device)
                    for label in range(self.args.num_classes):
                        labels = torch.ones(target.size(0)) * label
                        label_t_0, prob = self.multi_diffs[model_i].sample(labels.to(torch.long).to(self.args.device), w[:,model_i,:])
                        p_y_bar_x_y_tilde[:,:,label] = prob

                    # P(y|y^,x)*P(y^|x)=P(y,y^|x)
                    p_y_expansion = p_y_tilde[model_i].squeeze().reshape(w.size(0), 1, self.args.num_classes).repeat([1, self.args.num_classes, 1])
                    p_y_y_tilde = p_y_bar_x_y_tilde.cpu().detach() * p_y_expansion  # batch*class*label
                    p_y_y_tilde_list.append(p_y_y_tilde)
                if self.args.n_model == 1:
                    p_y_y_tilde_final = p_y_y_tilde_list[-1].squeeze()
                else:
                    p_y_y_tilde_final = torch.stack(p_y_y_tilde_list, dim=0).mean(0)
                _, pred_labels = torch.max(torch.sum(p_y_y_tilde_final, dim=2), dim=1)

                correct += torch.sum(pred_labels==target.detach().cpu()).item()
                plm_correct += torch.sum(plm_pred_labels==target.detach().cpu()).item()
                all_sample += w.size(0)

        print(f'time cost for sampling: {time.time() - start}')

        acc = 100 * correct / all_sample
        plm_acc = 100 * plm_correct / all_sample
        return acc, plm_acc