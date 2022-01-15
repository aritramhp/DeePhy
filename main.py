# coding: utf-8
import argparse
import time
import math
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.onnx
import torch.optim as optim
import copy
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, matthews_corrcoef, classification_report, r2_score

import data
import model

parser = argparse.ArgumentParser(description='DeePhy model for triplet construction')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--subdir', type=str, required=True, help="subdirectory of dataset, e.g. GFP")
parser.add_argument('--workers', type=int, help='number of data loading workers', default=16)
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--outf', type=str, default='cls_triplet', help='output folder')
parser.add_argument('--save', type=str, default='model.pt', help='path to save the final model')

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(random.randint(1, 10000))
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

print('='*9)
print(device)
print('='*9)

try:
    os.makedirs(args.outf)
except OSError:
    pass

logfile = 'log_triplet'
fp_log = open(logfile,'w')
fp_log.close()

num_points = 1356
###############################################################################
# Load data
# train and validation data are used during training and validation processes
###############################################################################
print('Loading data...')
start_time = time.time()
# dataset = data.TripletDataset(root=args.dataset, subdir=args.subdir, npoints=num_points)

dataset_train = np.load(os.path.join(args.dataset, 'train_test_split', args.subdir, 'train.npy'), allow_pickle=True)
train_target = np.load(os.path.join(args.dataset, 'train_test_split', args.subdir, 'target_train.npy'), allow_pickle=True)
dataset_valid = np.load(os.path.join(args.dataset, 'train_test_split', args.subdir, 'valid.npy'), allow_pickle=True)
valid_target = np.load(os.path.join(args.dataset, 'train_test_split', args.subdir, 'target_valid.npy'), allow_pickle=True)
dataset_test = np.load(os.path.join(args.dataset, 'train_test_split', args.subdir, 'test.npy'), allow_pickle=True)
test_target = np.load(os.path.join(args.dataset, 'train_test_split', args.subdir,'target_test.npy'), allow_pickle=True)


##############################################################################
# Batchify data to speed up the process
##############################################################################

def batchify(data, label, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = np.size(data,0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data[0:nbatch * bsz, :]
    label = label[0:nbatch * bsz]
    batch_data = []
    # Evenly divide the data across the bsz batches.
    pnt_1, pnt_2, pnt_3, target=[],[],[],[]
    batch_count=0
    for i in range(np.size(data,0)):
        x, y, z = data[i]    
        t = label[i]
        t = torch.from_numpy(np.nonzero(np.array(t))[0])

        pnt_1.append(x.T.reshape((-1,1,num_points)))
        pnt_2.append(y.T.reshape((-1,1,num_points)))
        pnt_3.append(z.T.reshape((-1,1,num_points)))
        target.append(t)
        batch_count+=1
        if batch_count==bsz:
            pnt_1, pnt_2, pnt_3, target = np.array(pnt_1), np.array(pnt_2), np.array(pnt_3), np.array(target),
            batch_data.append([torch.from_numpy(pnt_1), torch.from_numpy(pnt_2), torch.from_numpy(pnt_3), torch.from_numpy(target)])
            pnt_1, pnt_2, pnt_3, target=[],[],[],[]
            batch_count=0
    return batch_data, nbatch*bsz

print('Data loading completed. TIME: %f \nBatchifying data...' % (time.time() - start_time))


# Batchifying
start_time = time.time()
eval_batch_size = 1
test_data, num_test = batchify(dataset_test, test_target, eval_batch_size)


##############################################################################
# Prediction
##############################################################################
def evaluate(dataloader):
    classifier.eval()
    eval_loss = []
    correct = 0
    full_target = torch.Tensor()
    full_pred = torch.Tensor()
    
    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            t1, t2, t3, target = data
            t1, t2, t3, target = t1.to(device), t2.to(device), t3.to(device), target.to(device)
            pred = classifier(t1,t2,t3)   
            pred_choice = pred.data.max(1)[1]   
            correct += (target == pred_choice).sum().item()
    return correct


# Load the best saved model.
classifier = model.Triplet()
classifier.load_state_dict(torch.load(args.save))
classifier.to(device)

# Run on test data.
test_correct = evaluate(test_data)
print('Accuracy: {}'.format(float(test_correct)*100/num_test))

