import torch
from torch import nn
from sentence_transformers import SentenceTransformer
from bertPLM.data_utils import *

class SentenceBertForClassification(nn.Module):
    def __init__(self, args, sentBert, num_classes, hidden_size=768, dropout=0.1):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self.sentBert = SentenceTransformer(sentBert)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, self.num_classes)
        

    def forward(self, 
                sentences=None,
                labels=None,
                ):
        embeddings = self.sentBert.encode(sentences, convert_to_tensor=True)
        embeddings = self.dropout(embeddings)
        logits = self.classifier(embeddings)

        outputs = (logits,) + (embeddings, )
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.to(torch.long).view(-1))
            outputs = (loss,) + outputs
        
        return outputs


if __name__ == '__main__':
    model = SentenceTransformer('all-MiniLM-L6-v2')
    VALIDATION_SPLIT = 0.8
    newsgroups_train  = fetch_20newsgroups(data_home='datasets/', subset='train',  shuffle=True, random_state=0)
    # newsgroups_train  = fetch_20newsgroups(data_home=args.path, subset='train',  shuffle=True, random_state=0)
    newsgroups_test  = fetch_20newsgroups(data_home='datasets/', subset='test',  shuffle=False)
    train_len = int(VALIDATION_SPLIT * len(newsgroups_train.data))

    train_sentences = newsgroups_train.data[:train_len]
    val_sentences = newsgroups_train.data[train_len:]
    test_sentences = newsgroups_test.data
    train_labels = newsgroups_train.target[:train_len]
    val_labels = newsgroups_train.target[train_len:]
    test_labels = newsgroups_test.target

    train_dataloader = DataLoader(list(zip(train_sentences, train_labels)), batch_size=1)
    for i in train_dataloader:
        print(i)
        break

    
    

                                
