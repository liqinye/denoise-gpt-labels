import argparse
import pandas as pd 
import torch
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from torch.nn.utils.rnn import pad_sequence

import sys
sys.path.insert(0, '/home/remote/Desktop/denoiseGPT') # this should be the absolute path to project directory
from utils.utils import set_seed
from utils.knn import KNN_prior_dynamic
from multi_diff.multi_diff_trainer import Multinomial_Trainer



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_batch_size", default=128, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size for validation.")
    parser.add_argument("--epochs", default=3, type=int, help="Number of epochs for PLM training.")
    parser.add_argument("--dataset", default='numclaim', type=str, help="dataset")
    parser.add_argument("--saved_dataset", default='n', type=str, help="saved dataset or not")
    parser.add_argument("--path", type=str, default='')
    parser.add_argument("--n_model", type=int, default=2, help='The number of detection-relabeling iterations.')
    parser.add_argument("--seed", default=0, type=int, help="Number of epochs for training.")
    parser.add_argument("--bert", type=str, default="bert-base-uncased")
    parser.add_argument("--bert_type", type=str, default='bert', help="plm bert model choice")
    parser.add_argument("--lr", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument('--alpha_t_hi', type=float, default=5)
    parser.add_argument("--noise_ratio", type=float, default=0.2, help='The ratio of noisy data to be poisoned.')
    parser.add_argument("--diff_epoches", default=10, type=int, help="Number of epochs for training.")
    parser.add_argument("--warmup_epochs", default=5, help="warmup_epochs", type=int)
    parser.add_argument("--num_timesteps", default=100, help="Number of timesteps for diffusion model", type=int)
    parser.add_argument("--num_sample", default=6, help="Number of sample for dynamic prior", type=int)
    parser.add_argument('--lambda_t', type=float, default=2)
    parser.add_argument('--diff_batch_size', type=int, default=64, help='Batch size for diffusion training')
    

    args = parser.parse_args()
    set_seed(args)
    args.n_gpu = 1
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    args.device = device

    # Pre-process dataset
    if args.bert_type == 'bert':
        from bertPLM.data_utils import create_dataset
        from bertPLM.bert_trainer import Bert_Trainer

        train_data, train_sampler, train_dataloader, valid_data, valid_sampler, \
            valid_dataloader, test_data, test_sampler, test_dataloader = create_dataset(args)
        train_noisy_labels = torch.tensor([train_data[idx][-1] for idx in range(len(train_data))])
        train_inputs = torch.stack([train_data[idx][0] for idx in range(len(train_data))], dim=0)
        train_masks = torch.stack([train_data[idx][1] for idx in range(len(train_data))], dim=0)
        train_true_labels = torch.stack([train_data[idx][2] for idx in range(len(train_data))], dim=0)
        valid_noisy_labels = torch.tensor([valid_data[idx][-1] for idx in range(len(valid_data))])
        valid_inputs = torch.stack([valid_data[idx][0] for idx in range(len(valid_data))], dim=0)
        valid_masks = torch.stack([valid_data[idx][1] for idx in range(len(valid_data))], dim=0)
        valid_true_labels = torch.stack([valid_data[idx][2] for idx in range(len(valid_data))], dim=0)
        test_inputs = torch.stack([test_data[idx][0] for idx in range(len(test_data))], dim=0)
        test_masks = torch.stack([test_data[idx][1] for idx in range(len(test_data))], dim=0)
        test_labels = torch.stack([test_data[idx][2] for idx in range(len(test_data))], dim=0)
    


        print("================Start Training Stage I Model: Encode the Trajectory!================")
        bert_trainer = Bert_Trainer(args, train_dataloader, valid_dataloader, test_dataloader)
        # finetune pretrained LM on noisy labels
        z_train, z_valid, z_test, best_model, dists_list = bert_trainer.train()

    elif args.bert_type == 'sentbert':
        from sentBertPLM.sentBert_trainer import create_dataset
        from sentBertPLM.sentBert_trainer import sentBert_Trainer

        train_data, train_sampler, train_dataloader, valid_data, valid_sampler, \
            valid_dataloader, test_data, test_sampler, test_dataloader = create_dataset(args)
        train_noisy_labels = torch.tensor([train_data[idx][-1] for idx in range(len(train_data))])
        train_inputs = torch.stack([train_data[idx][0] for idx in range(len(train_data))], dim=0)
        train_true_labels = torch.stack([train_data[idx][1] for idx in range(len(train_data))], dim=0)
        valid_noisy_labels = torch.tensor([valid_data[idx][-1] for idx in range(len(valid_data))])
        valid_inputs = torch.stack([valid_data[idx][0] for idx in range(len(valid_data))], dim=0)
        valid_true_labels = torch.stack([valid_data[idx][1] for idx in range(len(valid_data))], dim=0)
        test_inputs = torch.stack([test_data[idx][0] for idx in range(len(test_data))], dim=0)
        test_labels = torch.stack([test_data[idx][1] for idx in range(len(test_data))], dim=0)

        print("================Start Training Stage I Model: Encode the Trajectory!================")
        sentbert_trainer = sentBert_Trainer(args, train_dataloader, valid_dataloader, test_dataloader)
        # finetune pretrained LM on noisy labels
        z_train, z_valid, z_test, best_model, dists_list = sentbert_trainer.train()

    dists_score_list = []
    markers_list = []
    for idx in range(dists_list.shape[0]):
        dists = dists_list[idx].squeeze()
        dists_labels = torch.cat((train_noisy_labels, valid_noisy_labels),dim=0)
        dists_mean = torch.mean(dists, 0)
        dists_mean = torch.tensor([dists_mean[i, dists_labels[i]] for i in range(len(dists_labels))])
        dists_var = torch.std(dists, 0)
        dists_var = torch.tensor([dists_var[i, dists_labels[i]] for i in range(len(dists_labels))])
        dists_score = dists_mean + dists_var
        dists_score = dists_score[:len(dists_labels)]
        markers = torch.zeros(len(dists_labels))
        number_points = int(len(dists_score) * args.noise_ratio)
        noisy_points = torch.topk(dists_score, number_points, largest=True).indices
        markers[noisy_points] = 1
        dists_score_list.append(dists_score.unsqueeze(0))
        markers_list.append(markers.unsqueeze(0))
    dists_score_list = torch.stack(dists_score_list, dim=0)
    markers_list = torch.stack(markers_list, dim=0)

    print(z_train.shape) # (models, epochs, batch, dim)
    z_train = z_train.permute(2,0,1,3)
    B, M, N, D = z_train.shape
    z_train = z_train.reshape(B, M, N*D)
    z0_train = z_train[:, :, :D]

    z_valid = z_valid.permute(2,0,1,3)
    B2, M2, N2, D2 = z_valid.shape
    z_valid = z_valid.reshape(B2, M2, N2*D2)
    z0_valid = z_valid[:, :, :D2]

    z_test = z_test.permute(2,0,1,3)
    B3, M3, N3, D3 = z_test.shape
    z_test = z_test.reshape(B3, M3, N3*D3)
    z0_test = z_test[:, :, :D3]

    train_priors = []
    train_prior_weights = []
    train_uncertain_marker = []

    z_train = torch.cat((z_train, z_valid), dim=0)

    for idx in range(M):
        knn_inputs = torch.cat((train_inputs, valid_inputs), 0)
        knn_masks = torch.cat((train_masks, valid_masks), 0)
        knn_z0 = torch.cat((z0_train[:, idx, :], z0_valid[:, idx, :]), 0).squeeze()
        knn_labels = torch.cat((train_noisy_labels, valid_noisy_labels))
        knn_true_labels = torch.cat((train_true_labels, valid_true_labels))
        knn_data = TensorDataset(knn_inputs, knn_masks, knn_z0, knn_labels)
        knn_sampler = SequentialSampler(knn_data)
        knn_dataloader = DataLoader(knn_data, sampler=knn_sampler, batch_size=args.train_batch_size)
        
        knn_prior = KNN_prior_dynamic(args, knn_data, knn_z0, knn_labels, knn_true_labels, markers_list[idx].squeeze())
        priors, weights, uncertain_marker, true_labels = knn_prior.get_dynamic_prior(k=10)

        train_priors.append(priors)
        train_prior_weights.append(weights)
        train_uncertain_marker.append(uncertain_marker)
        
    
    train_noisy_labels = torch.cat((train_noisy_labels, valid_noisy_labels), dim=0)
    train_uncertain_marker = torch.stack(train_uncertain_marker, dim=0)
    

    # make sure each model branch has the same length of priors
    train_priors = pad_sequence([model.transpose(0,1) for model in train_priors], batch_first=True, padding_value=-1).transpose(1,2)

    train_prior_weights = torch.stack(train_prior_weights, dim=0)

    train_priors = train_priors.permute(1,0,2)
    train_prior_weights = train_prior_weights.permute(1,0,2)
    train_uncertain_marker = train_uncertain_marker.permute(1,0)

    scaler = torch.cuda.amp.GradScaler()

    # prepare datasets for generative model
    train_dataset = TensorDataset(z_train, train_priors, train_prior_weights, train_uncertain_marker, train_noisy_labels, true_labels)
    test_dataset = TensorDataset(test_inputs, test_masks, test_labels, z_test)

    test_sampler = SequentialSampler(test_dataset)
    # train_z_dataloader = DataLoader(train_z_data, sampler=train_z_sampler, batch_size=args.train_batch_size)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.train_batch_size)

    # multi_diffusion = MultinomialDiffusion().to(args.device)
    multi_trainer = Multinomial_Trainer(args, train_dataset, None, test_dataloader, z_train.size(-1), best_model)
    
    multi_trainer.train()







    
        