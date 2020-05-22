import numpy as np
import csv
import argparse
import pickle
from sklearn import metrics
import copy

def evaluate(pred_logits, true_labels):
    #pred_probs = pred_logits[:,1]
    #pred_labels = np.argmax(pred_logits, axis=1)
    pred_probs = pred_logits.squeeze()
    pred_labels = copy.deepcopy(pred_logits)
    pred_labels[pred_labels>=0.5] = 1.
    pred_labels[pred_labels<0.5] = 0.
    #print(pred_probs.shape, pred_labels.shape, true_labels.shape)
    
    fpr, tpr, thresholds = metrics.roc_curve(true_labels, pred_probs, pos_label=0)
    auc = metrics.auc(fpr, tpr)
    acc = metrics.accuracy_score(true_labels, pred_labels)
    pre, rec, f1, _ = metrics.precision_recall_fscore_support(true_labels, pred_labels, average=None, labels=[0, 1])
    roc_auc = metrics.roc_auc_score(true_labels, pred_probs) 
    #print("auc: {} -- acc: {}".format(auc, acc))
    #print("roc_auc: {}".format(roc_auc))
    #print('pre: {} -- rec: {} -- f1: {} '.format(pre, rec, f1))
    return roc_auc, acc, pre[0], rec[0], f1[0]
    


parser = argparse.ArgumentParser(description='Patient Evaluation')
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

input_file_path = 'Results/dropout_{}_lr_{}_hidden_dim_{}_n_layers_{}_cw_{}.pkl'.format(dropout, lr, hidden_dim, n_layers, cw)
results = pickle.load(open(input_file_path, 'rb'))

output_file_path = 'Results/dropout_{}_lr_{}_hidden_dim_{}_n_layers_{}_cw_{}.csv'.format(dropout, lr, hidden_dim, n_layers, cw)
with open(output_file_path, mode='w') as output_file:
    writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['AUC', 'ACC', 'Precision', 'Recall', 'F1', 'Training Loss'])

    num_epochs = len(results['fold_1'])
    for epoch in range(1, num_epochs+1):
        true_labels = np.concatenate([results['fold_1'][epoch-1]['true_labels'], 
                                      results['fold_2'][epoch-1]['true_labels'],
                                      results['fold_3'][epoch-1]['true_labels'], 
                                      results['fold_4'][epoch-1]['true_labels'], 
                                      results['fold_5'][epoch-1]['true_labels']], axis=0)
        pred_logits = np.concatenate([results['fold_1'][epoch-1]['pred_logits'], 
                                      results['fold_2'][epoch-1]['pred_logits'],
                                      results['fold_3'][epoch-1]['pred_logits'], 
                                      results['fold_4'][epoch-1]['pred_logits'], 
                                      results['fold_5'][epoch-1]['pred_logits']], axis=0)
        avg_epoch_loss = np.mean([results['fold_1'][epoch-1]['training_loss'], 
                                  results['fold_2'][epoch-1]['training_loss'], 
                                  results['fold_3'][epoch-1]['training_loss'], 
                                  results['fold_4'][epoch-1]['training_loss'], 
                                  results['fold_5'][epoch-1]['training_loss']])
        #print(true_labels.shape, pred_logits.shape)

        roc_auc, acc, pre, rec, f1 = evaluate(pred_logits, true_labels)
        writer.writerow([roc_auc, acc, pre, rec, f1, avg_epoch_loss])
        
        


