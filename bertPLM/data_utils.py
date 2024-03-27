import numpy as np
import torch
import random
import pandas as pd
from scipy import stats
from math import inf
from sklearn.datasets import fetch_20newsgroups
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def corrupt_dataset_SN(args, data, num_labels):
    total_number = len(data)
    new_data = data.detach().clone()
    noise_ratio = args.noise_ratio * num_labels / (num_labels - 1)
    for i in range(len(new_data)):
        if random.random() > noise_ratio:
            continue
        else:
            new_data[i] = torch.randint(low=0, high=num_labels, size=(1, ))
    return new_data 

def corrupt_dataset_ASN(args, data, num_labels):
    total_number = len(data)
    new_data = data.detach().clone()
    for i in range(len(new_data)):
        if random.random() > args.noise_ratio:
            continue
        else:
            new_data[i] = (new_data[i] + 1) % num_labels
    return new_data

def corrupt_dataset_IDN(args, inputs, labels, num_labels):
    flip_distribution = stats.truncnorm((0-args.noise_ratio)/0.1, (1-args.noise_ratio)/0.1, loc=args.noise_ratio, scale=0.1)
    flip_rate = flip_distribution.rvs(len(labels))
    W = torch.randn(num_labels, inputs.shape[-1], num_labels).float()
    new_label = labels.detach().clone()
    for i in range(len(new_label)):
        p = inputs[i].float().view(1,-1).mm(W[labels[i].long()].squeeze(0)).squeeze(0)
        p[labels[i]] = -inf
        p = flip_rate[i] * torch.softmax(p, dim=0)
        p[labels[i]] += 1 - flip_rate[i]
        new_label[i] = torch.multinomial(p,1)
    return new_label


def load_dataset(args, dataset):
    if dataset == "20news":
        VALIDATION_SPLIT = 0.8
        newsgroups_train  = fetch_20newsgroups(data_home='datasets/', subset='train',  shuffle=True, random_state=args.seed)
        newsgroups_train  = fetch_20newsgroups(data_home=args.path, subset='train',  shuffle=True, random_state=args.seed)
        print(newsgroups_train.target_names)
        print(len(newsgroups_train.data))

        newsgroups_test  = fetch_20newsgroups(data_home='datasets/', subset='test',  shuffle=False)

        print(len(newsgroups_test.data))

        train_len = int(VALIDATION_SPLIT * len(newsgroups_train.data))

        train_input_sent = newsgroups_train.data[:train_len]
        valid_input_sent = newsgroups_train.data[train_len:]
        test_input_sent = newsgroups_test.data
        train_true_labels = newsgroups_train.target[:train_len]
        valid_true_labels = newsgroups_train.target[train_len:]
        test_labels = newsgroups_test.target 

        train_noisy_labels = None
        valid_noisy_labels = None
    
    elif dataset.lower() == 'numclaim' or dataset.lower() == 'sa' or dataset.lower() == 'fomc':
        train_file_path = f"../datasets/train_{dataset.lower()}_example.csv"
        valid_file_path = f"../datasets/valid_{dataset.lower()}_example.csv"
        test_file_path = f"../datasets/test_{dataset.lower()}_example.csv"

        train_file = pd.read_csv(train_file_path)
        train_true_labels = torch.tensor(train_file['true_label'].values, device=args.device)
        train_noisy_labels = torch.tensor(train_file['noisy_label'].values, device=args.device)
        train_input_sent = train_file['original_sent'].values

        valid_file = pd.read_csv(valid_file_path)
        valid_true_labels = torch.tensor(valid_file['true_label'].values, device=args.device)
        valid_noisy_labels = torch.tensor(valid_file['noisy_label'].values, device=args.device)
        valid_input_sent = valid_file['original_sent'].values

        test_file = pd.read_csv(test_file_path)
        test_labels = torch.tensor(test_file['true_label'].values, device=args.device)
        test_noisy_labels = torch.tensor(test_file['noisy_label'].values, device=args.device)
        test_input_sent = test_file['original_sent'].values

    return train_input_sent, valid_input_sent, test_input_sent, train_true_labels, train_noisy_labels, valid_true_labels, valid_noisy_labels, test_labels

def create_dataset(args):
    if args.dataset == '20news':
        num_labels = 20
        args.num_classes = 20
    
    elif args.dataset == 'numclaim':
        args.num_classes = 2

    elif args.dataset == 'SA':
        args.num_classes = 3

    elif args.dataset == 'FOMC':
        args.num_classes = 3

    if args.saved_dataset == 'n':
        # Get raw sentences and labels
        train_input_sent, valid_input_sent, test_input_sent, train_true_labels, \
            train_noisy_labels, valid_true_labels, valid_noisy_labels, test_labels = load_dataset(args, args.dataset)
        
        train_true_labels = torch.tensor(train_true_labels, dtype=torch.long, device=args.device)
        valid_true_labels = torch.tensor(valid_true_labels, dtype=torch.long, device=args.device)
        test_labels = torch.tensor(test_labels, dtype=torch.long, device=args.device)
        
        if train_noisy_labels is None and valid_noisy_labels is None:
            if args.noise_type == 'SN':
                train_noisy_labels = corrupt_dataset_SN(args, train_true_labels, num_labels)
                valid_noisy_labels = corrupt_dataset_SN(args, valid_true_labels, num_labels)
            elif args.noise_type == 'ASN':
                train_noisy_labels = corrupt_dataset_ASN(args, train_true_labels, num_labels)
                valid_noisy_labels = corrupt_dataset_ASN(args, valid_true_labels, num_labels)
            elif args.noise_type == 'IDN':
                train_noisy_labels = corrupt_dataset_IDN(args, train_input_sent, train_true_labels, num_labels)
                valid_noisy_labels = corrupt_dataset_IDN(args, valid_input_sent, valid_true_labels, num_labels)

        train_noisy_labels = torch.tensor(train_noisy_labels, dtype=torch.long, device=args.device)
        valid_noisy_labels = torch.tensor(valid_noisy_labels, dtype=torch.long, device=args.device)
        

        if args.dataset == '20news':
            MAX_LEN = 150
        elif args.dataset == 'chemprot':
            MAX_LEN = 512
        else:
            MAX_LEN = 128

        # Encode train/test text
        # ===========================
        tokenizer = BertTokenizer.from_pretrained(args.bert, do_lower_case=True)
        train_input_ids = []
        train_attention_masks = []
        for sent in train_input_sent:
            encoded_sent = tokenizer.encode(
                                sent,                      # Sentence to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                # This function also supports truncation and conversion
                                # to pytorch tensors, but we need to do padding, so we
                                # can't use these features :( .
                                max_length = MAX_LEN,          # Truncate all sentences.
                                #return_tensors = 'pt',     # Return pytorch tensors.
                        )
            train_input_ids.append(encoded_sent)

        train_input_ids = pad_sequences(train_input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
        for seq in train_input_ids:
            seq_mask = [float(i>0) for i in seq]
            train_attention_masks.append(seq_mask)

        train_inputs = torch.tensor(train_input_ids, device=args.device)
        train_masks = torch.tensor(train_attention_masks, device=args.device)

        valid_input_ids = []
        valid_attention_masks = []
        for sent in valid_input_sent:
            encoded_sent = tokenizer.encode(
                                    sent,                      # Sentence to encode.
                                    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                    # This function also supports truncation and conversion
                                    # to pytorch tensors, but we need to do padding, so we
                                    # can't use these features :( .
                                    max_length = MAX_LEN,          # Truncate all sentences.
                                    #return_tensors = 'pt',     # Return pytorch tensors.
                            )
            valid_input_ids.append(encoded_sent)

        valid_input_ids = pad_sequences(valid_input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
        for seq in valid_input_ids:
            seq_mask = [float(i>0) for i in seq]
            valid_attention_masks.append(seq_mask)

        valid_inputs = torch.tensor(valid_input_ids, device=args.device)
        valid_masks = torch.tensor(valid_attention_masks, device=args.device)

        test_input_ids = []
        test_attention_masks = []
        for sent in test_input_sent:
            encoded_sent = tokenizer.encode(
                                    sent,                      # Sentence to encode.
                                    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                    # This function also supports truncation and conversion
                                    # to pytorch tensors, but we need to do padding, so we
                                    # can't use these features :( .
                                    max_length = MAX_LEN,          # Truncate all sentences.
                                    #return_tensors = 'pt',     # Return pytorch tensors.
                            )
            test_input_ids.append(encoded_sent)

        test_input_ids = pad_sequences(test_input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
        for seq in test_input_ids:
            seq_mask = [float(i>0) for i in seq]
            test_attention_masks.append(seq_mask)

        test_inputs = torch.tensor(test_input_ids, device=args.device)
        test_masks = torch.tensor(test_attention_masks, device=args.device)
        # ===========================
        train_data = TensorDataset(train_inputs, train_masks, train_true_labels, train_noisy_labels)
        train_sampler = SequentialSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        valid_data = TensorDataset(valid_inputs, valid_masks, valid_true_labels, valid_noisy_labels)
        valid_sampler = SequentialSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=args.eval_batch_size)

        test_data = TensorDataset(test_inputs, test_masks, test_labels)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

        if args.dataset.lower() in ['numclaim', 'sa', 'fomc']:
            file_path = args.path + f'../saved_data/{args.dataset.lower()}-train-{args.seed}.pt'
            torch.save(train_data, file_path)
            file_path = args.path + f'../saved_data/{args.dataset.lower()}-valid-{args.seed}.pt'
            torch.save(valid_data, file_path)
            file_path = args.path + f'../saved_data/{args.dataset.lower()}-test-{args.seed}.pt'
            torch.save(test_data, file_path)
        else:
            file_path = args.path + f'../saved_data/{args.dataset.lower()}-{args.noise_type}-{args.noise_ratio}-train-{args.seed}.pt'
            torch.save(train_data, file_path)
            file_path = args.path + f'../saved_data/{args.dataset.lower()}-{args.noise_type}-{args.noise_ratio}-valid-{args.seed}.pt'
            torch.save(valid_data, file_path)
            file_path = args.path + f'../saved_data/{args.dataset.lower()}-{args.noise_type}-{args.noise_ratio}-test-{args.seed}.pt'
            torch.save(test_data, file_path)

    else:
        if args.dataset.lower() in ['numclaim', 'sa', 'fomc']:
            file_path = args.path + f'../saved_data/{args.dataset.lower()}-train-{args.seed}.pt'
            train_data = torch.load(file_path)
            train_sampler = SequentialSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

            file_path = args.path + f'../saved_data/{args.dataset.lower()}-valid-{args.seed}.pt'
            valid_data = torch.load(file_path)
            valid_sampler = SequentialSampler(valid_data)
            valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=args.eval_batch_size)

            file_path = args.path + f'../saved_data/{args.dataset.lower()}-test-{args.seed}.pt'
            test_data = torch.load(file_path)
            test_sampler = SequentialSampler(test_data)
            test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)
        else:
            file_path = args.path + f'../saved_data/{args.dataset.lower()}-{args.noise_type}-{args.noise_ratio}-train-{args.seed}.pt'
            train_data = torch.load(file_path)
            train_sampler = SequentialSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

            file_path = args.path + f'../saved_data/{args.dataset.lower()}-{args.noise_type}-{args.noise_ratio}-valid-{args.seed}.pt'
            valid_data = torch.load(file_path)
            valid_sampler = SequentialSampler(valid_data)
            valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=args.eval_batch_size)

            file_path = args.path + f'../saved_data/{args.dataset.lower()}-{args.noise_type}-{args.noise_ratio}-test-{args.seed}.pt'
            test_data = torch.load(file_path)
            test_sampler = SequentialSampler(test_data)
            test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)


    return train_data, train_sampler, train_dataloader, valid_data, valid_sampler, valid_dataloader, test_data, test_sampler, test_dataloader



    