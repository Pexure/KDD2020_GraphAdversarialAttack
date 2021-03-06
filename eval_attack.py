import pickle as pkl
from model import GCN
import torch
from data import preprocess_adj, sparse_symmetric_add
import torch.nn as nn
import numpy as np
import math as m
import torch.nn.functional as F
from config import args

###################################
# load data                       #
###################################
y_train = pkl.load(open('../experimental_train.pkl', 'rb'))  # np.ndarray; (543486,); label within [0, 17]
y_test = pkl.load(open('./data/y_test_eye1.pkl', 'rb'))  # tensor; (50000,); label within [0, 17]
adj = pkl.load(open('../experimental_adj.pkl', 'rb'))  # sparse.csr.csr_matrix (593486, 593486)
X_all = pkl.load(open('../experimental_features.pkl', 'rb'))  # numpy.ndarray (593486, 100)



###################################
# preprocess                      #
###################################
X_all = torch.from_numpy(X_all)
y_train = torch.from_numpy(y_train).long()  # why long? int64

# mask
num_all = X_all.shape[0]
num_train_val = y_train.shape[0]
num_train = round(num_train_val * 0.8)
num_val = num_train_val - num_train
num_test = 50000
# feature_dim = X_all.shape[1]
feature_dim = 100
num_classes = 18


########################################
# add attack nodes and edges           #
########################################
row = []
for i in range(500):
    row.append([i] * 100)
row = np.concatenate(row) + num_all # [0,0,0...1,1,1...4,4,4]
# col = np.array(get_col()[:size_attack * 100]) + num_train_val + offset
col = np.array(list(range(500 * 100))) + num_train_val
new_adj = sparse_symmetric_add(adj, row, col, num_all + 500)

supports = preprocess_adj(new_adj)
i = torch.from_numpy(supports[0]).long()  # i.shape = (6810490, 2)
v = torch.from_numpy(supports[1])  # tensor (6810490, )
adj_shape = supports[2]
support = torch.sparse.DoubleTensor(i.t(), v, adj_shape)
# print(support.shape)  # (593986, 593986)


############################################
# train attack nodes features              #
############################################
# TODO: 'cuda'
device = torch.device(args.dev)
# device = torch.device('cuda')

z = np.load('./result/feature.npy')
z = torch.from_numpy(z)

X_all = X_all.to(device)
support = support.to(device)
y_test = y_test.to(device)


net = GCN(feature_dim, num_classes)
net.load_state_dict(torch.load('./data/param_eye1_39.pkl', map_location=device))
# net.load_state_dict(torch.load('./data/parameters0.48.pkl', map_location=device))
net = net.to(device)
net.eval()
# X_attack.requires_grad = True
test_label_mask = (torch.tensor(list(range(500 * 100)))).to(device)
test_logits_mask = (test_label_mask + num_train_val).to(device)

print('eval_attack')
'''z_std = torch.std(z, dim=1).reshape(-1, 1).to(device)
z_mean = torch.mean(z, dim=1).reshape(-1, 1).to(device)
z_norm = ((z - z_mean) / z_std * m.sqrt(0.1)).to(device)'''
z_norm = nn.Tanhs
# print(z[0])
# print(z_norm[0])

X_attack = torch.cat((X_all, z), dim=0).to(device)
out = net((X_attack, support))
logits = out[0]  # shape = (num_all + #attack nodes, 18)
acc = (torch.argmax(logits[test_logits_mask], dim=1) == y_train[test_label_mask]).double().mean()
print(acc.item())
