SEED = 1234
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from Dataset import Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, f1_score, classification_report
import pickle, math
import numpy as np
import random
import os

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
        self.rnn = nn.LSTM(self.input_size, self.hidden_dim, self.n_layers, dropout=(0 if self.n_layers == 1 else self.dropout),
                           batch_first=True)
        #self.gru = nn.GRU(self.input_size, self.hidden_size, self._layers, dropout=(0 if self.n_layers == 1 else self.dropout), batch_first=True)
        if self.is_attn:
            self.attn = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.v = nn.Parameter(torch.FloatTensor(self.hidden_dim))
        self.fc = nn.Linear(self.hidden_dim, self.n_classes)
        self.activation = nn.Softmax(dim=-1)

    def forward(self, x):
        #batch_size = x.size(0)
        #hidden = self.init_hidden(batch_size)
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

## TODO: Implement this function to calculate the AUC, accuracy, Precision, Recall and F-score.
#   Create a result variable global to keep these measures at the end of each epoch for each fold.
#   Then, you can calculate avg of folds for each epoch.
#def evaluate(predicted_logits, true_labels):


## define some parameters
num_epochs = 100
lr = 0.001
batch_size = 32
dropout = 0.2
is_attn = False

## Read data
x_path = 'Results/FeatureVector/segment200_40/sup_ratio/feature_mat_dist_dl_shape_offset_o_2_0.7.pkl'
y_path = 'Data/patient_labels_720.pkl'
x = pickle.load(open(x_path, 'rb'))
y = pickle.load(open(y_path, 'rb'))
print("x.shape: {}, y.shape: {}".format(x.shape, y.shape))

## Adjust folds and train/evaluate model.
skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
for train_indices, test_indices in skf.split(x, y):
    ## Get data for the fold.
    fold_train_x, fold_train_y = x[train_indices], y[train_indices]
    fold_test_x, fold_test_y = x[test_indices], y[test_indices]

    ## Create tensors from data for the fold.
    fold_test_x_tensor = torch.from_numpy(fold_test_x).to(device, dtype=torch.float)
    fold_test_y_tensor = torch.tensor(fold_test_y).to(device, dtype=torch.long)

    training_dataset = Dataset(fold_train_x, fold_train_y)
    training_generator = DataLoader(training_dataset, batch_size=batch_size, shuffle=False)

    ## Create model.
    model = Model(input_size=fold_train_x.shape[-1], hidden_dim=32, n_layers=1, n_classes=2, dropout=dropout, is_attn=is_attn)
    model.to(device)

    ## Define Loss, Optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ## Train and evaluate the model.
    for epoch in range(1, num_epochs+1):
        model.train()
        epoch_losses = []
        for train_batch_x, train_batch_y in training_generator:
            batch_y_hat = model(train_batch_x)
            loss = criterion(batch_y_hat, train_batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        if epoch % 10 == 0:
            print("--- Epoch: {} ---".format(epoch))
            print("Loss: {}".format(np.mean(epoch_losses)))
        model.eval()
        test_y_hat = model(fold_test_x_tensor)
        test_y_hat_logits = model.activation(test_y_hat)
        #evaluate(test_y_hat_logits.data.numpy(), fold_test_y)

