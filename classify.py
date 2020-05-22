SEED = 1234
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from Dataset import Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, f1_score, classification_report
import pickle, math
import numpy as np
import random
import os
import argparse
import sys

random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, input_size, hidden_dim, n_classes, n_layers, dropout, is_attn=False):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.dropout = dropout
        self.is_attn = is_attn
        self.rnn = nn.LSTM(self.input_size, self.hidden_dim, self.n_layers, dropout=(0 if self.n_layers == 1 else self.dropout), batch_first=True)
        #self.rnn = nn.GRU(self.input_size, self.hidden_dim, self.n_layers, dropout=(0 if self.n_layers == 1 else self.dropout), batch_first=True)
        if self.is_attn:
            self.attn = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.v = nn.Parameter(torch.FloatTensor(self.hidden_dim))
        self.fc = nn.Linear(self.hidden_dim, self.n_classes)
        #self.activation = nn.Softmax(dim=-1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # batch_size = x.size(0)
        # hidden = self.init_hidden(batch_size)
        outputs, hidden = self.rnn(x)
        if self.is_attn:
            energies = self.score(outputs)  # (B, T)
            attn_weights = F.softmax(energies, dim=1).unsqueeze(1)  # (B, 1, T)
            att_output = torch.bmm(attn_weights, outputs).squeeze(1)
            output = self.fc(att_output)
        else:
            output = self.fc(outputs[:, -1])
        return output

    def logits(self, x):
        output = self.activation(x)
        return output

    # rnn_outputs.shape (B, T, D)
    def score(self, rnn_outputs):
        energy = torch.tanh(self.attn(rnn_outputs))
        energy = torch.sum(self.v * energy, dim=2)
        return energy  # (B, T)

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim), torch.zeros(self.n_layers, batch_size, self.hidden_dim))
        return hidden


parser = argparse.ArgumentParser(description='Patient Classification')
parser.add_argument('--dropout', default=0.2, type=float)
parser.add_argument('--hidden_dim', default=32, type=int)
parser.add_argument('--n_layers', default=1, type=int)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--cw', default='no', type=str)
args = parser.parse_args()
dropout = float(args.dropout)
hidden_dim = args.hidden_dim
n_layers = args.n_layers
lr = float(args.lr)
cw = args.cw

## define some parameters
num_epochs = 100
batch_size = 64
is_attn = True
num_fold = 5
clip_value = 10
n_classes = 1


## to keep results
results = {}
for i in range(1, num_fold+1):
    results['fold_{}'.format(i)] = []


## Read data
x_path = 'Results/FeatureVector/segment200_40/sup_ratio/feature_mat_dist_dl_shape_offset_spike_o_2_0.7_sp_2.5_50_1.pkl'
y_path = 'Data/patient_labels_720.pkl'
x = pickle.load(open(x_path, 'rb'))
y = pickle.load(open(y_path, 'rb'))
x = np.transpose(x, (0, 2, 1))
x = (x - x.mean()) / x.std()
#x = (x - np.amin(x)) / (np.amax(x) - np.amin(x))
print("x.shape: {}, y.shape: {}".format(x.shape, y.shape))
print('recovered:', np.shape(np.where(y==1)), 'dead:', np.shape(np.where(y==0)))
sys.stdout.flush()

if cw == 'yes':
    weights = [float(max(np.shape(np.where(y==1))[1], np.shape(np.where(y==0))[1]))/np.shape(np.where(y==0))[1], 
               float(max(np.shape(np.where(y==1))[1], np.shape(np.where(y==0))[1]))/np.shape(np.where(y==1))[1]]
    print("class weights: {}".format(weights))
    class_weights = torch.FloatTensor(weights).to(device)
    pos_weight = float(max(np.shape(np.where(y==1))[1], np.shape(np.where(y==0))[1]))/np.shape(np.where(y==1))[1]
    print("weight: {}".format(pos_weight))
    pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(device)
    sys.stdout.flush()

## Adjust folds and train/evaluate model.
skf = StratifiedKFold(n_splits=num_fold, random_state=None, shuffle=True)
fold = 1
for train_indices, test_indices in skf.split(x, y):
    ## Get data for the fold.
    fold_train_x, fold_train_y = x[train_indices], np.expand_dims(y[train_indices], axis=-1)
    fold_test_x, fold_test_y = x[test_indices], np.expand_dims(y[test_indices], axis=-1)
    
    #print(train_indices, fold_train_y)

    ## Create tensors from data for the fold.
    fold_test_x_tensor = torch.from_numpy(fold_test_x).to(device, dtype=torch.float)
    fold_test_y_tensor = torch.tensor(fold_test_y).to(device, dtype=torch.float)

    training_dataset = Dataset(fold_train_x, fold_train_y)
    training_generator = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

    ## Create model.
    model = Model(input_size=fold_train_x.shape[-1], hidden_dim=hidden_dim, n_layers=n_layers, 
                  n_classes=n_classes, dropout=dropout, is_attn=is_attn)
    model.to(device)

    ## Define Loss, Optimizer.
    if cw == 'no':
        #criterion = nn.CrossEntropyLoss()
        criterion = nn.BCEWithLogitsLoss()
    elif cw == 'yes':
        #criterion = nn.CrossEntropyLoss(weight=class_weights)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ## Train and evaluate the model.
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_losses = []
        for train_batch_x, train_batch_y in training_generator:
            optimizer.zero_grad()
            batch_y_hat = model(train_batch_x)
            loss = criterion(batch_y_hat, train_batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            epoch_losses.append(loss.item())
            #print('Epoch: {}/{}.............Training Loss: {:.4f}'.format(epoch, num_epochs, loss.item()), end='\r')
        #print('Epoch: {}/{}.............Training Loss: {:.4f}'.format(epoch, num_epochs, np.mean(epoch_losses)))

        if epoch % 1 == 0:
            #print("--- Epoch: {} ---".format(epoch))
            #print("Loss: {}".format(np.mean(epoch_losses)))

            model.eval()
            test_y_hat = model(fold_test_x_tensor)
            test_y_hat_logits = model.activation(test_y_hat)
            
            results['fold_{}'.format(fold)].append({'true_labels': fold_test_y, 'pred_logits': test_y_hat_logits.data, 
                                                   'training_loss': np.mean(epoch_losses)})
        if epoch % 10 == 0:
            print('Epoch: {}/{}.............Training Loss: {:.4f}'.format(epoch, num_epochs, np.mean(epoch_losses)))
            sys.stdout.flush()
        
    fold+=1

pickle.dump(results, open('Results/dropout_{}_lr_{}_hidden_dim_{}_n_layers_{}_cw_{}.pkl'.format(dropout, lr, 
                                                                                                hidden_dim, n_layers, cw), 'wb'))
            