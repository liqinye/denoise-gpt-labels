import argparse
import os
import numpy as np
import torch
from torch import nn
from tqdm import tqdm, trange
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer
from torch.optim import Adam
from transformers  import BertTokenizer, BertConfig
from transformers  import AdamW, BertModel, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
from src.utils.utils2 import *
from src.model.generate_prior import *
from src.dynamics.euclidean_dist import *
from src.model.train_stage1 import *
from src.model.train_stage2 import *
from src.model.evaluate import *
from src.utils.generate_noise import *
# from train_stage1 import *


from utils.diff_utils import *
from multi_diff.multi_diff import *
from tqdm import tqdm
from utils.ema import EMA
from simplex_diff.simplex_utils import convert_to_simplex
# from knn_utils import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--vae_lr", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--train_batch_size", default=256, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=256, type=int, help="Batch size for training.")
    parser.add_argument("--vae_batch_size", default=256, type=int, help="Batch size for training.")
    parser.add_argument("--epochs", default=2, type=int, help="Number of epochs for training.")
    parser.add_argument("--vae_epochs", default=20, type=int, help="Number of epochs for training.")
    parser.add_argument("--seed", default=0, type=int, help="Number of epochs for training.")
    parser.add_argument("--dataset", default='20news', type=str, help="dataset")
    parser.add_argument("--noise_ratio", type=float, default=0.4, help='The ratio of noisy data to be poisoned.')
    parser.add_argument("--noise_type", type=str, default="SN")
    parser.add_argument("--n_model", type=int, default=1, help='The number of detection-relabeling iterations.')
    parser.add_argument("--selected_class", type=str, default='1', help='Choose the relabeling method: second_close')
    parser.add_argument("--prior_norm", type=int, default=5, help='Choose the relabeling method: second_close')
    parser.add_argument('--softplus_beta', type=float, default=1, help='softplus beta')
    parser.add_argument("--total_iter", type=int, default=10, help='total iter (Default : 10)')
    parser.add_argument('--beta', type=float, default=1.0, help='coefficient on kl loss, beta vae')
    parser.add_argument('--clip_gradient_norm', type=float, default=100000, help='max norm for gradient clipping')
    
    # switches
    parser.add_argument('--lambda_t', type=float, default=5)
    parser.add_argument('--alpha_t_hi', type=float, default=5)
    parser.add_argument("--path", type=str, default='./datasets/20news')
    parser.add_argument("--bert", type=str, default="bert-base-uncased")


    # diffusion
    parser.add_argument("--ddim_n_step", default=15, help="number of steps in ddim", type=int)
    parser.add_argument("--warmup_epochs", default=30, help="warmup_epochs", type=int)
    parser.add_argument("--num_timestpes", default=2000, help="warmup_epochs", type=int)
    parser.add_argument("--num_epoches", default=150, help="warmup_epochs", type=int)

    
    args = parser.parse_args()
    args.n_gpu = 1
    print(args)
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    set_seed(args)

    if args.dataset == '20news':
        num_labels = 20
        args.num_classes = 20

    z_test = torch.load("/home/remote/Desktop/DyGen/stage1/val_noisy/z_test.pt")
    z_train = torch.load("/home/remote/Desktop/DyGen/stage1/val_noisy/z_train.pt")
    z_val = torch.load("/home/remote/Desktop/DyGen/stage1/val_noisy/z_val.pt")
    noisy_train_labels = torch.load("/home/remote/Desktop/DyGen/stage1/val_noisy/noisy_train_labels.pt")
    train_inputs = torch.load("/home/remote/Desktop/DyGen/stage1/val_noisy/train_inputs.pt")
    train_masks = torch.load("/home/remote/Desktop/DyGen/stage1/val_noisy/train_masks.pt")
    train_labels = torch.load("/home/remote/Desktop/DyGen/stage1/val_noisy/train_labels.pt")
    validation_inputs = torch.load("/home/remote/Desktop/DyGen/stage1/val_noisy/validation_inputs.pt")
    validation_masks = torch.load("/home/remote/Desktop/DyGen/stage1/val_noisy/validation_masks.pt")
    validation_labels = torch.load("/home/remote/Desktop/DyGen/stage1/val_noisy/validation_labels.pt")
    noisy_validation_labels = torch.load("/home/remote/Desktop/DyGen/stage1/val_noisy/noisy_validation_labels.pt")
    test_inputs = torch.load("/home/remote/Desktop/DyGen/stage1/val_noisy/test_inputs.pt")
    test_masks = torch.load("/home/remote/Desktop/DyGen/stage1/val_noisy/test_masks.pt")
    test_labels = torch.load("/home/remote/Desktop/DyGen/stage1/val_noisy/test_labels.pt")
    best_model = torch.load("/home/remote/Desktop/DyGen/stage1/val_noisy/best_model.pt")
    markers_list = torch.load("/home/remote/Desktop/DyGen/stage1/val_noisy/markers_list.pt")

    train_labels = torch.cat((train_labels, validation_labels), dim=0)

    z_train = z_train[[1], :, :, :]
    z_val = z_val[[1], :, :, :]
    z_test = z_test[[1], :, :, :]

    print(z_train.shape) # (models, epochs, batch, dim)
    z_train = z_train.permute(2,0,1,3)
    B, M, N, D = z_train.shape
    z_train = z_train.reshape(B, M, N*D)
    z0_train = z_train[:, :, :D]

    z_val = z_val.permute(2,0,1,3)
    B2, M2, N2, D2 = z_val.shape
    z_val = z_val.reshape(B2, M2, N2*D2)
    z0_val = z_val[:, :, :D2]

    z_test = z_test.permute(2,0,1,3)
    B3, M3, N3, D3 = z_test.shape
    z_test = z_test.reshape(B3, M3, N3*D3)
    z0_test = z_test[:, :, :D3]
    train_priors = []
    val_priors = []
    for idx in range(M):
        knn_inputs = torch.cat((train_inputs, validation_inputs), 0)
        knn_masks = torch.cat((train_masks, validation_masks), 0)
        knn_z0 = torch.cat((z0_train[:, idx, :], z0_val[:, idx, :]), 0).squeeze()
        knn_labels = torch.cat((noisy_train_labels, noisy_validation_labels))
        knn_true_labels = torch.cat((train_labels, validation_labels))
        knn_data = TensorDataset(knn_inputs, knn_masks, knn_z0, knn_labels)
        knn_sampler = SequentialSampler(knn_data)
        knn_dataloader = DataLoader(knn_data, sampler=knn_sampler, batch_size=args.vae_batch_size)
        
        knn_prior = KNN_prior_dynamic(args, knn_data, knn_z0, knn_labels, knn_true_labels, markers_list[idx].squeeze())
        priors, uncertain_marker = knn_prior.get_prior(best_model)

        priors = torch.tensor(priors)
        train_priors.append(priors)

    train_priors = torch.stack(train_priors, dim=0)

    train_priors = train_priors.permute(1,0)

    if M == 1:
        train_priors = train_priors.squeeze().unsqueeze(-1)

    scaler = torch.cuda.amp.GradScaler()

    z_train = torch.cat((z_train, z_val), dim=0)
    noisy_train_labels = torch.cat((noisy_train_labels, noisy_validation_labels), dim=0)

    # prepare datasets for generative model
    train_z_data = TensorDataset(z_train, train_priors, noisy_train_labels)
    test_z_data = TensorDataset(test_inputs, test_masks, test_labels, z_test)

    train_z_sampler = SequentialSampler(train_z_data)
    test_z_sampler = SequentialSampler(test_z_data)
    train_z_dataloader = DataLoader(train_z_data, sampler=train_z_sampler, batch_size=args.vae_batch_size)
    test_z_dataloader = DataLoader(test_z_data, sampler=test_z_sampler, batch_size=args.vae_batch_size)

    multi_diffusion = MultinomialDiffusion(timesteps=args.num_timestpes, num_classes=20, w_dim=768*20).to(args.device)

    train(multi_diffusion, train_z_dataloader, test_z_dataloader, args.num_epoches, args, noisy_train_labels, best_model)
    




def train(diffusion_model, train_loader, test_loader, n_epochs, args, noisy_labels, best_plm):
    optimizer = optim.Adam(diffusion_model.parameters(), lr=0.0001, weight_decay=0.0, betas=(0.9, 0.999), amsgrad=False, eps=1e-08)
    diffusion_loss = nn.MSELoss(reduction='none')
    # diffusion_loss = nn.BCEWithLogitsLoss(reduction='none')
    criterion = nn.CrossEntropyLoss()

    ema_helper = EMA(mu=0.9999)
    ema_helper.register(diffusion_model._denoise_fn)

    max_accuracy = 0.0

    acc_list = []

    print('Diffusion training start')
    for epoch in range(n_epochs):
        diffusion_model._denoise_fn.train()

        with tqdm(enumerate(train_loader), total=len(train_loader), desc=f'train diffusion epoch {epoch}', ncols=120) as pbar:
            for i, data_batch in pbar:
                [w, y, noisy_y] = data_batch
                y = y.squeeze().to(args.device)
                noisy_y = noisy_y.to(args.device)
                w = w.to(args.device).squeeze()
                w = torch.log(w.float().clamp(min=1e-30))
                n = w.size(0)

                simplex = convert_to_simplex(y, 5.0, args.num_classes)

                adjust_learning_rate(optimizer, i / len(train_loader) + epoch, warmup_epochs=args.warmup_epochs, n_epochs=n_epochs, lr_input=0.001)
                # train with and without prior
                loss = - diffusion_model.log_prob(y, noisy_y, w).sum() / (math.log(2) * n)
                
                # loss = torch.sum(loss)
                pbar.set_postfix({'loss': loss.item()})
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), 1.0)
                optimizer.step()
                ema_helper.update(diffusion_model._denoise_fn)


        if (epoch % 10 == 0 and epoch >= args.warmup_epochs) or epoch == n_epochs-1:
            test_acc, plm_acc  = test(diffusion_model, test_loader, args, best_plm)
            acc_list.append(test_acc)
            print(f"Epoch {epoch}: PLM acc: {plm_acc}, Diff acc: {test_acc}")
            max_accuracy = max(max_accuracy, test_acc)
            if max_accuracy == test_acc:
                torch.save(diffusion_model._denoise_fn.state_dict(), "best_multi_diffusion.pt")
                print(f"Model saved, update best accuracy at Epoch {epoch}, test acc: {test_acc}")

    print(acc_list)



def test(diffusion_model, test_loader, args, best_plm):
    diffusion_model._denoise_fn.eval()
    start = time.time()
    with torch.no_grad():
        correct_cnt = 0
        all_cnt = 0
        plm_correct = 0
        for test_batch_idx, data_batch in tqdm(enumerate(test_loader), total=len(test_loader), desc=f'Doing DDIM...', ncols=100):
            input_ids, input_mask, target, w = data_batch 
            w = w.squeeze().to(args.device)
            w = torch.log(w.float().clamp(min=1e-30))
            target = target.squeeze().to(args.device)

            outputs = best_plm(input_ids, attention_mask=input_mask)
            logit = outputs[1][0]
            p_y_tilde = F.softmax(logit, dim=-1).detach().cpu()
            
            p_y_bar_x_y_tilde = torch.zeros(target.size(0), args.num_classes, args.num_classes).to(args.device)
            plm_pred = torch.argmax(p_y_tilde, dim=-1)
            label_t_0, prob = diffusion_model.sample(plm_pred.to(torch.long).to(args.device), w)
            pred_labels = torch.argmax(label_t_0, dim=-1).detach().cpu()
            # for label in range(args.num_classes):
            #     labels = torch.ones(target.size(0)) * label
            #     label_t_0, prob = diffusion_model.sample(labels.to(torch.long).to(args.device), w)
            #     p_y_bar_x_y_tilde[:,:,label] = prob
            # p_y_expansion = p_y_tilde.squeeze().reshape(w.size(0), 1, args.num_classes).repeat([1, args.num_classes, 1])

            # p_y_y_tilde = p_y_bar_x_y_tilde.detach().cpu() * p_y_expansion
            # pred_labels = torch.argmax(torch.sum(p_y_y_tilde, dim=-1), dim=-1).detach().cpu()
            
            # p_y_y_tilde = prob.detach().cpu() * p_y_tilde
            # pred_labels = torch.argmax(label_t_0, dim=-1).detach().cpu()

            correct_cnt += torch.sum(pred_labels==target.detach().cpu()).item()
            plm_correct += torch.sum(plm_pred==target.detach().cpu()).item()
            all_cnt += w.shape[0]

    print(f'time cost for CLR: {time.time() - start}')

    acc = 100 * correct_cnt / all_cnt
    plm_acc = 100 * plm_correct / all_cnt
    return acc, plm_acc

def cnt_agree(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k.
    """
    maxk = min(max(topk), output.size()[1])

    output = torch.softmax(-(output - 1)**2,  dim=-1)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return torch.sum(correct).item()

def cast_label_to_one_hot_and_prototype(y_labels_batch, n_class, return_prototype=True):
    """
    y_labels_batch: a vector of length batch_size.
    """
    y_one_hot_batch = nn.functional.one_hot(y_labels_batch, num_classes=n_class).float()
    if return_prototype:
        label_min, label_max = [0.001, 0.999]
        y_logits_batch = torch.logit(nn.functional.normalize(
            torch.clip(y_one_hot_batch, min=label_min, max=label_max), p=1.0, dim=1))
        return y_one_hot_batch, y_logits_batch
    else:
        return y_one_hot_batch

if __name__ == '__main__':
    main()