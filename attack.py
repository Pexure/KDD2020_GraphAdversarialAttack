import pickle as pkl
from model import GCN
import torch
from data import preprocess_adj, sparse_symmetric_add
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from config import args

# TODO: feel free to change lr & seed
lr = 0.2
lr_decay = 0.9
seed = 43
np.random.seed(seed)
torch.random.manual_seed(seed)

###################################
# load data                       #
###################################
y_train = pkl.load(open('../experimental_train.pkl', 'rb'))  # np.ndarray; (543486,); label within [0, 17]
y_test = pkl.load(open('y_test.pkl', 'rb'))  # tensor; (50000,); label within [0, 17]
adj = pkl.load(open('../experimental_adj.pkl', 'rb'))  # sparse.csr.csr_matrix (593486, 593486)
X_all = pkl.load(open('../experimental_features.pkl', 'rb'))  # numpy.ndarray (593486, 100)



###################################
# preprocess                      #
###################################
X_all = torch.from_numpy(X_all)

'''supports = preprocess_adj(adj)
i = torch.from_numpy(supports[0]).long()  # i.shape = (6810490, 2)
v = torch.from_numpy(supports[1])  # tensor (6810490, )
adj_shape = supports[2]
support = torch.sparse.DoubleTensor(i.t(), v, adj_shape)'''

y_train = torch.from_numpy(y_train).long()  # why long? int64

# mask
num_all = X_all.shape[0]
num_train_val = y_train.shape[0]
num_train = round(num_train_val * 0.8)
num_val = num_train_val - num_train
num_test = 50000
'''perm = torch.randperm(num_train_val)
train_mask = perm[:num_train]
val_mask = perm[num_train:num_train_val]
test_mask = torch.tensor(list(range(50000))) + num_train_val'''
# feature_dim = X_all.shape[1]
feature_dim = 100
num_classes = 18



########################################
# add attack nodes and edges           #
########################################
# TODO: feel free to change size_attack
size_attack = 5
kk = 0  # 0 <= kk < 500//size_attack
offset = kk * size_attack * 100
row = []
for i in range(size_attack):
    row.append([i] * 100)
row = np.concatenate(row) + num_all # [0,0,0...1,1,1...4,4,4]
# row = list(range(size_attack)) * 100
# row = np.array(row) + num_all
col = np.array(list(range(size_attack * 100))) + num_train_val + offset
new_adj = sparse_symmetric_add(adj, row, col, num_all + size_attack)

# X_attack = 0.5 * torch.randn((size_attack, 100)).double()
z = 0.5 * torch.randn((size_attack, 100)).double()
X_attack = torch.cat((X_all, z), dim=0)
# print(X_attack.shape)  # (593986, 100)

supports = preprocess_adj(new_adj)
i = torch.from_numpy(supports[0]).long()  # i.shape = (6810490, 2)
v = torch.from_numpy(supports[1])  # tensor (6810490, )
adj_shape = supports[2]
support = torch.sparse.DoubleTensor(i.t(), v, adj_shape)
# print(support.shape)  # (593487, 593487)


############################################
# train attack nodes features              #
############################################
def max_loss(logits, test_labels):
    loss = torch.tensor(0).double()
    for x, y in zip(logits, test_labels):
        # x.shape = (18,)
        x = F.softmax(x, dim=0)
        loss += torch.max(x) - x[y]
    return loss

# TODO: 'gpu' or 'cuda'
device = torch.device('cpu')
X_attack = X_attack.to(device)
support = support.to(device)
y_test = y_test.to(device)


net = GCN(feature_dim, num_classes).to(device)
net.load_state_dict(torch.load('parameters0.48.pkl', map_location=device))
net.eval()
X_attack.requires_grad = True
test_label_mask = (torch.tensor(list(range(size_attack * 100))) + offset).to(device)
test_logits_mask = (test_label_mask + num_train_val + offset).to(device)
min_f = -2.2
max_f = 2.6

print('start')
for epoch in range(args.attack_epochs):
    epoch += 1
    print('epoch:', epoch)

    out = net((X_attack, support))
    logits = out[0]  # shape = (num_all + #attack nodes, 18)

    loss = max_loss(logits[test_logits_mask], y_test[test_label_mask])
    print('loss:', loss)

    dx = torch.autograd.grad(outputs=loss, inputs=X_attack)[0]  # tuple: (tensor,)
    print('derivative done.')
    if epoch % 5 == 0:
        lr = lr * lr_decay
    X_attack[-size_attack:] += lr * dx[-size_attack:]
    X_attack[-size_attack:] = torch.clamp(X_attack[-size_attack:], min_f, max_f)

    acc = (torch.argmax(logits[test_logits_mask], dim=1) == y_test[test_label_mask]).double().mean()
    print('test acc:', acc.item())
    print('max:', torch.max(X_attack[-size_attack:]).item(), 'min:', torch.min(X_attack[-size_attack:]).item())
    print()

# acc = (torch.argmax(logits[val_mask], dim=1) == y_train[val_mask]).float().mean()
# print(acc.item())

# one fake node attacks 100 test nodes
# 'fake_k': attack test nodes within [100 * size_attack * k, 100 * size_attack * (k+1))
file_path = 'fake_' + str(kk) + '.pkl'
f = open(file_path, 'wb')
dic = dict()
dic['seed'] = seed
dic['feat'] = X_attack[-size_attack:]
pkl.dump(dic, f)
