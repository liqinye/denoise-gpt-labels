"""
Compute the prior function.
"""
import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class KNN_prior_dynamic:
    def __init__(self, args, z0, noisy_train_labels, true_train_labels, noisy_markers):
        self.args = args
        self.n_classes = self.args.num_classes
        self.time = time.time()

        self.y_hat = noisy_train_labels
        self.y = true_train_labels
        self.noisy_markers = noisy_markers
        self.z0 = z0
        self.emb_dim = z0.shape[-1]

        print('\n===> Prior Generation with KNN Start')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # For each datapoint, get initial class name
    def get_prior(self, best_model):
        # Load Classifier
        self.net = best_model
        self.net.to(self.device)

        # k-nn
        self.net.eval()
        # knn mode on
        neigh = KNeighborsClassifier(n_neighbors=10, weights='distance')
        embeddings, class_confi = [], []
        knn_embeds = self.z0[self.noisy_markers==0, :]
        class_confi = self.y_hat[self.noisy_markers==0]

        # class_confi = self.y_hat
        knn_embeds = knn_embeds.cpu().detach().numpy()
        class_confi = class_confi.cpu().detach().numpy()
        neigh.fit(knn_embeds, class_confi)
        print('Time : ', time.time() - self.time)

        # 2. predict class of training dataset
        knn_embeds = self.z0.cpu().detach().numpy()
        class_preds = neigh.predict(knn_embeds)
        class_preds = torch.tensor(np.int64(class_preds))
        
        # ==============
        class_prob = neigh.predict_proba(knn_embeds)
        
        class_prob = torch.tensor(class_prob)

        max_proba, max_idx = torch.max(class_prob, dim=1)

        uncertain_marker = max_proba < 0.8

        # print(max_idx.size())
        # print(max_idx[uncertain_marker==False].size())
        return max_idx.numpy(), uncertain_marker
        # ==============

        print('Prior made {} errors with train/val noisy labels'.format(torch.sum(class_preds!=self.y_hat)))
        print('Prior made {} errors with train/val clean labels'.format(torch.sum(class_preds!=self.y)))
        noisy_preds = torch.tensor([(class_preds[i] != self.y_hat[i]) and (class_preds[i] == self.y[i]) for i in range(len(class_preds))])
        print('Prior detected {} real noisy samples'.format(torch.sum(noisy_preds)))
        correct_marker = class_preds == self.y

        # # proba
        dict = {}
        model_output = neigh.predict_proba(knn_embeds)
        if model_output.shape[1] < self.n_classes:
            tmp = np.zeros((model_output.shape[0], self.n_classes))
            tmp[:, neigh.classes_] = neigh.predict_proba(embeddings)
            dict['proba'] = tmp
        else:
            dict['proba'] = model_output  # data*n_class

        print('Time : ', time.time() - self.time, 'proba information saved')

        return dict['proba'], class_preds.numpy()


    def get_dynamic_prior(self, k=20, weighted=True):
        neigh = KNeighborsClassifier(n_neighbors=k, weights='distance')
        embeddings, class_confi = [], []
        # get clean data point's embeddings and corresponding labels
        clean_embed = self.z0[self.noisy_markers==0, :]
        clean_label = self.y_hat[self.noisy_markers==0]

        clean_embed = clean_embed.cpu().detach().numpy()
        clean_label = clean_label.cpu().detach().numpy()

        neigh.fit(clean_embed, clean_label)
        print('Time : ', time.time() - self.time)

        # 2. predict class of training dataset
        all_train_embed = self.z0.cpu().detach().numpy()
        num_sample = self.z0.size()[0]

        # Get neighbor embed index
        neighbor_embed_idx = neigh.kneighbors(all_train_embed, k, return_distance=False)

        proba = neigh.predict_proba(all_train_embed)

        proba = torch.tensor(proba)

        # get the potential neighbor labels with probability > 0
        neighbor = []
        for i in range(proba.size()[0]):
            neighbor.append(torch.where(proba[i, :]>0.0)[0].cpu().detach().numpy().tolist())

        neighbor = [torch.tensor(labels) for labels in neighbor]

        # get the max probability and its corresponding index
        max_proba, max_idx = torch.max(proba, dim=1)

        # get the marker for uncertain label
        # uncertain label: the max probability is less than 80%
        uncertain_marker = max_proba < 0.8

        # convert certain data point's probability into one hot
        certain_proba = F.one_hot(max_idx[uncertain_marker==False], num_classes=self.n_classes).to(dtype=torch.float64)

        # change the original proba for certain data point
        proba[uncertain_marker==False] = certain_proba

        # filter out certain/determined labels
        uncertain_proba = proba[uncertain_marker==True]

        # get the max 2 probs and idx in uncertain sets
        top2_proba, top2_idx = torch.topk(uncertain_proba, 2)
        
        # check the sum of max 2 probs if it is over 0.8
        dominant_marker = torch.sum(top2_proba, dim=1) >= 0.8

        # eliminate other trival labels if the sum is over 0.8
        dominant_idx = top2_idx[dominant_marker == True]

        # normalize the proability to make sure they add up to 1
        dominant_proba = top2_proba[dominant_marker==True] / torch.sum(top2_proba[dominant_marker==True], dim=1, keepdim=True)

        # initialize need-to-update uncertain probability (which is dominant above)
        update_uncertain_proba = torch.zeros_like(uncertain_proba[dominant_marker==True])
        row_idx = torch.arange(dominant_idx.size()[0])
        
        # broadcast to update dominant probability
        update_uncertain_proba[row_idx, dominant_idx[:, 0]] = dominant_proba[row_idx, 0]
        update_uncertain_proba[row_idx, dominant_idx[:, 1]] = dominant_proba[row_idx, 1]

        # update in original probability
        aux_uncertain_proba = proba[uncertain_marker==True]
        aux_uncertain_proba[dominant_marker==True] = update_uncertain_proba
        proba[uncertain_marker==True] = aux_uncertain_proba
        
        # get neighbor labels
        neighbor = []
        for i in range(proba.size()[0]):
            neighbor.append(torch.where(proba[i, :]>0.0)[0])

        # pad with -1 to maintain same size
        neighbor = pad_sequence(neighbor, batch_first=True, padding_value=-1)

        return neighbor, proba, uncertain_marker, self.y