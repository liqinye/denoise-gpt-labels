import argparse
import pandas as pd 
import torch
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

import sys
sys.path.insert(0, '/fintech_3/liqin/denoiseGPT') # this should be the absolute path to project directory
from utils.utils import set_seed
from utils.knn import KNN_prior_dynamic
from simplex_diff_trainer import Simplex_Trainer
# from simplex_diff import SimplexDiffusion, ConditionalModel




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_batch_size", default=128, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size for validation.")
    parser.add_argument("--epochs", default=3, type=int, help="Number of epochs for PLM training.")
    parser.add_argument("--dataset", default='numclaim', type=str, help="dataset")
    parser.add_argument("--saved_dataset", default='n', type=str, help="saved dataset or not")
    parser.add_argument("--path", type=str, default='')
    parser.add_argument("--n_model", type=int, default=3, help='The number of detection-relabeling iterations.')
    parser.add_argument("--seed", default=0, type=int, help="Number of epochs for training.")
    parser.add_argument("--bert", type=str, default="bert-base-uncased", help='bert-base-uncased or stsb-bert-base')
    parser.add_argument("--sentbert", type=str, default="all-mpnet-base-v2", help='all-mpnet-base-v2')
    parser.add_argument("--bert_type", type=str, default='bert', help="plm bert model choice")
    parser.add_argument("--lr", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument('--alpha_t_hi', type=float, default=5)
    parser.add_argument("--noise_ratio", type=float, default=0.2, help='The ratio of noisy data to be poisoned.')

    parser.add_argument("--diff_epochs", default=10, type=int, help="Number of epochs for training.")
    parser.add_argument("--warmup_epochs", default=5, help="warmup_epochs", type=int)
    parser.add_argument("--num_timesteps", default=500, help="Number of timesteps for diffusion model", type=int)
    parser.add_argument("--num_sample", default=6, help="Number of sample for dynamic prior", type=int)
    parser.add_argument('--lambda_t', type=float, default=2)
    parser.add_argument('--diff_batch_size', type=int, default=64, help='Batch size for diffusion training')
    parser.add_argument("--beta_schedule", type=str, default="squaredcos_improved_ddpm")
    parser.add_argument("--simplex_value", type=float, default=5.0)
    parser.add_argument("--clip_sample", type=bool, default=False, help="Whether to clip predicted sample between -1 and 1 for numerical stability in the noise scheduler.")


    


    args = parser.parse_args()
    set_seed(args)
    args.n_gpu = 1
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    print(f'Device using: {device}')
    args.device = device
    model_path_prefix = '/fintech_3/hf_models/'

    args.bert = model_path_prefix + args.bert
    print(args.bert)


    if args.dataset.lower() == '20news':
        num_labels = 20
        args.num_classes = 20
    
    elif args.dataset.lower() == 'numclaim':
        args.num_classes = 2

    elif args.dataset.lower() == 'sa':
        args.num_classes = 3

    elif args.dataset.lower() == 'fomc':
        args.num_classes = 3
        
    # # Pre-process dataset
    # if args.bert_type == 'bert':
    #     from bertPLM.data_utils import create_dataset
    #     from bertPLM.bert_trainer import Bert_Trainer

    #     train_data, train_sampler, train_dataloader, valid_data, valid_sampler, \
    #         valid_dataloader, test_data, test_sampler, test_dataloader, train_embedding, valid_embedding, test_embedding = create_dataset(args)
    #     train_noisy_labels = torch.tensor([train_data[idx][-2] for idx in range(len(train_data))])
    #     train_inputs = torch.stack([train_data[idx][0] for idx in range(len(train_data))], dim=0)
    #     train_masks = torch.stack([train_data[idx][1] for idx in range(len(train_data))], dim=0)
    #     train_true_labels = torch.stack([train_data[idx][2] for idx in range(len(train_data))], dim=0)
    #     valid_noisy_labels = torch.tensor([valid_data[idx][-2] for idx in range(len(valid_data))])
    #     valid_inputs = torch.stack([valid_data[idx][0] for idx in range(len(valid_data))], dim=0)
    #     valid_masks = torch.stack([valid_data[idx][1] for idx in range(len(valid_data))], dim=0)
    #     valid_true_labels = torch.stack([valid_data[idx][2] for idx in range(len(valid_data))], dim=0)
    #     test_inputs = torch.stack([test_data[idx][0] for idx in range(len(test_data))], dim=0)
    #     test_masks = torch.stack([test_data[idx][1] for idx in range(len(test_data))], dim=0)
    #     test_labels = torch.stack([test_data[idx][2] for idx in range(len(test_data))], dim=0)
    


    #     print("================Start Training Stage I Model: Encode the Trajectory!================")
    #     bert_trainer = Bert_Trainer(args, train_dataloader, valid_dataloader, test_dataloader)
    #     # finetune pretrained LM on noisy labels
    #     z_train, z_valid, z_test, best_model, dists_list = bert_trainer.train()

    # elif args.bert_type == 'sentbert':
    #     from sentBertPLM.data_utils import create_dataset
    #     from sentBertPLM.sentBert_trainer import sentBert_Trainer

    #     train_data, train_dataloader, valid_data,\
    #         valid_dataloader, test_data, test_dataloader = create_dataset(args)
    #     train_noisy_labels = torch.tensor([train_data[idx][-1] for idx in range(len(train_data))])
    #     # train_inputs = torch.stack([train_data[idx][0] for idx in range(len(train_data))], dim=0)
    #     train_inputs = [train_data[idx][0] for idx in range(len(train_data))]
    #     train_true_labels = torch.stack([train_data[idx][1] for idx in range(len(train_data))], dim=0)
    #     valid_noisy_labels = torch.tensor([valid_data[idx][-1] for idx in range(len(valid_data))])
    #     # valid_inputs = torch.stack([valid_data[idx][0] for idx in range(len(valid_data))], dim=0)
    #     valid_inputs = [valid_data[idx][0] for idx in range(len(valid_data))]
    #     valid_true_labels = torch.stack([valid_data[idx][1] for idx in range(len(valid_data))], dim=0)
    #     # test_inputs = torch.stack([test_data[idx][0] for idx in range(len(test_data))], dim=0)
    #     test_inputs = [test_data[idx][0] for idx in range(len(test_data))]
    #     test_labels = torch.stack([test_data[idx][1] for idx in range(len(test_data))], dim=0)

    #     print("================Start Training Stage I Model: Encode the Trajectory!================")
    #     sentbert_trainer = sentBert_Trainer(args, train_dataloader, valid_dataloader, test_dataloader)
    #     # finetune pretrained LM on noisy labels
    #     z_train, z_valid, z_test, best_model, dists_list = sentbert_trainer.train()

    # z_train = torch.load(f'../stage1_buffer/{args.dataset}/z_train.pt')
    # z_valid = torch.load(f'../stage1_buffer/{args.dataset}/z_val.pt')
    # z_test = torch.load(f'../stage1_buffer/{args.dataset}/z_test.pt')
    # best_model = torch.load(f'../stage1_buffer/{args.dataset}/best_model.pt')
    # # markers_list = torch.load(f'../stage1_buffer/{args.dataset}/markers_list.pt')
    # train_inputs = torch.load(f'../stage1_buffer/{args.dataset}/train_inputs.pt')
    # valid_inputs = torch.load(f'../stage1_buffer/{args.dataset}/valid_inputs.pt')
    # test_inputs = torch.load(f'../stage1_buffer/{args.dataset}/test_inputs.pt')
    # train_masks = torch.load(f'../stage1_buffer/{args.dataset}/train_masks.pt')
    # valid_masks = torch.load(f'../stage1_buffer/{args.dataset}/valid_masks.pt')
    # test_masks = torch.load(f'../stage1_buffer/{args.dataset}/test_masks.pt')
    # train_true_labels = torch.load(f'../stage1_buffer/{args.dataset}/train_true_labels.pt')
    # valid_true_labels = torch.load(f'../stage1_buffer/{args.dataset}/valid_true_labels.pt')
    # test_true_labels = torch.load(f'../stage1_buffer/{args.dataset}/test_true_labels.pt')
    # train_noisy_labels = torch.load(f'../stage1_buffer/{args.dataset}/train_noisy_labels.pt')
    # valid_noisy_labels = torch.load(f'../stage1_buffer/{args.dataset}/valid_noisy_labels.pt')
    # test_labels = torch.load(f'../stage1_buffer/{args.dataset}/test_labels.pt')
    # train_embedding = torch.load(f'../stage1_buffer/{args.dataset}/train_embedding.pt')
    # valid_embedding = torch.load(f'../stage1_buffer/{args.dataset}/valid_embedding.pt')
    # test_embedding = torch.load(f'../stage1_buffer/{args.dataset}/test_embedding.pt')
    # dists_list = torch.load(f'../stage1_buffer/{args.dataset}/dists_list.pt')

    z_train = torch.load(f'/fintech_3/liqin/DyGen/stage1_buffer/{args.dataset}/z_train.pt', map_location=args.device)
    z_valid = torch.load(f'/fintech_3/liqin/DyGen/stage1_buffer/{args.dataset}/z_val.pt', map_location=args.device)
    z_test = torch.load(f'/fintech_3/liqin/DyGen/stage1_buffer/{args.dataset}/z_test.pt', map_location=args.device)
    best_model = torch.load(f'/fintech_3/liqin/DyGen/stage1_buffer/{args.dataset}/best_model.pt', map_location=args.device)
    train_inputs = torch.load(f'/fintech_3/liqin/DyGen/stage1_buffer/{args.dataset}/train_inputs.pt', map_location=args.device)
    valid_inputs = torch.load(f'/fintech_3/liqin/DyGen/stage1_buffer/{args.dataset}/valid_inputs.pt', map_location=args.device)
    test_inputs = torch.load(f'/fintech_3/liqin/DyGen/stage1_buffer/{args.dataset}/test_inputs.pt', map_location=args.device)
    train_masks = torch.load(f'/fintech_3/liqin/DyGen/stage1_buffer/{args.dataset}/train_masks.pt', map_location=args.device)
    valid_masks = torch.load(f'/fintech_3/liqin/DyGen/stage1_buffer/{args.dataset}/valid_masks.pt', map_location=args.device)
    test_masks = torch.load(f'/fintech_3/liqin/DyGen/stage1_buffer/{args.dataset}/test_masks.pt', map_location=args.device)
    train_true_labels = torch.load(f'/fintech_3/liqin/DyGen/stage1_buffer/{args.dataset}/train_true_labels.pt', map_location=args.device)
    valid_true_labels = torch.load(f'/fintech_3/liqin/DyGen/stage1_buffer/{args.dataset}/valid_true_labels.pt', map_location=args.device)
    test_true_labels = torch.load(f'/fintech_3/liqin/DyGen/stage1_buffer/{args.dataset}/test_true_labels.pt', map_location=args.device)
    train_noisy_labels = torch.load(f'/fintech_3/liqin/DyGen/stage1_buffer/{args.dataset}/train_noisy_labels.pt', map_location=args.device)
    valid_noisy_labels = torch.load(f'/fintech_3/liqin/DyGen/stage1_buffer/{args.dataset}/valid_noisy_labels.pt', map_location=args.device)
    test_labels = torch.load(f'/fintech_3/liqin/DyGen/stage1_buffer/{args.dataset}/test_true_labels.pt', map_location=args.device)
    train_embedding = torch.load(f'/fintech_3/liqin/DyGen/stage1_buffer/{args.dataset}/train_embedding.pt', map_location=args.device)
    valid_embedding = torch.load(f'/fintech_3/liqin/DyGen/stage1_buffer/{args.dataset}/valid_embedding.pt', map_location=args.device)
    test_embedding = torch.load(f'/fintech_3/liqin/DyGen/stage1_buffer/{args.dataset}/test_embedding.pt', map_location=args.device)
    dists_list = torch.load(f'/fintech_3/liqin/DyGen/stage1_buffer/{args.dataset}/dists_list.pt', map_location=args.device)

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

    # torch.save(z_train, f'../stage1_buffer_sent/{args.dataset}/z_train.pt')
    # torch.save(z_valid, f'../stage1_buffer_sent/{args.dataset}/z_val.pt')
    # torch.save(z_test, f'../stage1_buffer_sent/{args.dataset}/z_test.pt')
    # torch.save(best_model, f'../stage1_buffer_sent/{args.dataset}/best_model.pt')
    # torch.save(markers_list, f'../stage1_buffer_sent/{args.dataset}/markers_list.pt')
    # torch.save(train_inputs, f'../stage1_buffer_sent/{args.dataset}/train_inputs.pt')
    # torch.save(valid_inputs, f'../stage1_buffer_sent/{args.dataset}/valid_inputs.pt')
    # torch.save(test_inputs, f'../stage1_buffer_sent/{args.dataset}/test_inputs.pt')
    # torch.save(train_masks, f'../stage1_buffer_sent/{args.dataset}/train_masks.pt')
    # torch.save(valid_masks, f'../stage1_buffer_sent/{args.dataset}/valid_masks.pt')
    # torch.save(test_masks, f'../stage1_buffer_sent/{args.dataset}/test_masks.pt')
    # torch.save(train_true_labels, f'../stage1_buffer_sent/{args.dataset}/train_true_labels.pt')
    # torch.save(valid_true_labels, f'../stage1_buffer_sent/{args.dataset}/valid_true_labels.pt')
    # torch.save(test_labels, f'../stage1_buffer_sent/{args.dataset}/test_true_labels.pt')
    # torch.save(train_noisy_labels, f'../stage1_buffer_sent/{args.dataset}/train_noisy_labels.pt')
    # torch.save(valid_noisy_labels, f'../stage1_buffer_sent/{args.dataset}/valid_noisy_labels.pt')
    # torch.save(test_labels, f'../stage1_buffer_sent/{args.dataset}/test_labels.pt')
    # torch.save(train_embedding, f'../stage1_buffer_sent/{args.dataset}/train_embedding.pt')
    # torch.save(valid_embedding, f'../stage1_buffer_sent/{args.dataset}/valid_embedding.pt')
    # torch.save(test_embedding, f'../stage1_buffer_sent/{args.dataset}/test_embedding.pt')
    # torch.save(dists_list, f'../stage1_buffer_sent/{args.dataset}/dists_list.pt')

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
        # knn_z0 = torch.cat((z0_train[:, idx, :], z0_valid[:, idx, :]), 0).squeeze()
        knn_labels = torch.cat((train_noisy_labels, valid_noisy_labels))
        knn_true_labels = torch.cat((train_true_labels, valid_true_labels))
        knn_z0 = torch.cat((train_embedding, valid_embedding), 0).squeeze()

        knn_prior = KNN_prior_dynamic(args, knn_z0, knn_labels, knn_true_labels, markers_list[idx].squeeze())
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

    train_embedding = torch.cat([train_embedding, valid_embedding], dim=0)

    # prepare datasets for generative model
    train_dataset = TensorDataset(z_train, train_priors, train_prior_weights, train_uncertain_marker, train_noisy_labels, true_labels, train_embedding)
    # train_dataset = TrainDataset(z_train, train_priors, train_prior_weights, train_uncertain_marker, train_noisy_labels, true_labels, train_embedding)
    if args.bert_type == 'bert':
        test_dataset = TensorDataset(test_inputs, test_masks, test_labels, z_test, test_embedding)
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.train_batch_size)
    elif args.bert_type == 'sentbert':
        test_dataset = test_data = list(zip(test_inputs, test_labels, z_test))
        test_dataloader = DataLoader(test_dataset, batch_size=args.train_batch_size)



    simplex_trainer = Simplex_Trainer(args, train_dataset, None, test_dataloader, z_train.size(-1), best_model)

    
    simplex_trainer.train()