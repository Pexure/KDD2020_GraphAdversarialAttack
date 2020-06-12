import pickle as pkl
import math as m
from col_design import get_col
from model import GCN
import torch
from data import preprocess_adj, sparse_symmetric_add
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch import optim
from config import args

# TODO: feel free to change lr & seed
lr = args.lr
lr_decay = 0.9
seed = args.seed
np.random.seed(seed)
torch.random.manual_seed(seed)

###################################
# load data                       #
###################################
# y_train = pkl.load(open('../experimental_train.pkl', 'rb'))  # np.ndarray; (543486,); label within [0, 17]
y_test = pkl.load(open('./data/y_test.pkl', 'rb'))  # tensor; (50000,); label within [0, 17]
adj = pkl.load(open('../experimental_adj.pkl', 'rb'))  # sparse.csr.csr_matrix (593486, 593486)
X_all = pkl.load(open('../experimental_features.pkl', 'rb'))  # numpy.ndarray (593486, 100)



###################################
# preprocess                      #
###################################
X_all = torch.from_numpy(X_all)
# for i in range(10):
#     print(torch.var(X_all[i]).item(), torch.mean(X_all[i]).item())
# x[i]: mean around 0; variance around 0.1
# print(torch.max(X_all[0]).item(), torch.min(X_all[0]).item())

# mask
num_all = X_all.shape[0]
num_test = 50000
num_train_val = num_all - num_test
# feature_dim = X_all.shape[1]
feature_dim = 100
num_classes = 18



########################################
# add attack nodes and edges           #
########################################
# TODO: feel free to change size_attack
size_attack = args.size
kk = args.kk  # 0 <= kk < 500//size_attack
offset = kk * size_attack * 100
row = []
for i in range(size_attack):
    row.append([i] * 100)
row = np.concatenate(row) + num_all  # [0,0,0...1,1,1...4,4,4]
col = np.array(get_col()[:size_attack * 100]) + num_train_val + offset
# col = np.array(list(range(size_attack * 100))) + num_train_val + offset
new_adj = sparse_symmetric_add(adj, row, col, num_all + size_attack)

supports = preprocess_adj(new_adj)
i = torch.from_numpy(supports[0]).long()  # i.shape = (6810490, 2)
v = torch.from_numpy(supports[1])  # tensor (6810490, )
adj_shape = supports[2]
support = torch.sparse.DoubleTensor(i.t(), v, adj_shape)
# print(support.shape)  # (593487, 593487)


############################################
# train attack nodes features              #
############################################
def max_loss(logits, test_labels, dev):
    a = F.softmax(logits, dim=1).to(dev)
    # a = logits
    loss = torch.sum(torch.max(a, dim=1)[0] - a[range(len(test_labels)), test_labels]).to(dev)
    # bug fixed: didn't flush loss to device
    return loss

# TODO: 'cuda'
device = torch.device(args.dev)
print(device)
# device = torch.device('cuda')

sigma = args.sigma
mu = args.mu
if args.init_z == 'randn':
    z = (sigma * torch.randn((size_attack, feature_dim)) + mu).double()
else:
    z = (sigma * torch.rand((size_attack, 100)) + mu).double()
# z = torch.zeros((size_attack, feature_dim)).double()
# z[:, 0] += 1
# print('z[0] variance:', torch.var(z[0]).item())

X_all = X_all.to(device)
support = support.to(device)
y_test = y_test.to(device)
z = z.to(device)


net = GCN(feature_dim, num_classes)
net.load_state_dict(torch.load('./data/param_eye1_39.pkl', map_location=device))
net = net.to(device)
net.eval()
z.requires_grad = True
optimizer = optim.SGD([z], lr=lr)

test_label_mask = (torch.tensor(list(range(size_attack * 100))) + offset).to(device)
test_logits_mask = (test_label_mask + num_train_val).to(device)


print('start')
print('z[0]:', z[0])
for epoch in range(args.epochs):
    epoch += 1
    print('epoch:', epoch)

    '''z_std = torch.std(z, dim=1).reshape(-1, 1).to(device)
    z_mean = torch.mean(z, dim=1).reshape(-1, 1).to(device)
    z_norm = ((z - z_mean) / z_std * m.sqrt(0.1)).to(device)'''

    X_attack = torch.cat((X_all, z), dim=0).to(device)
    out = net((X_attack, support))
    logits = out[0].to(device)  # shape = (num_all + #attack nodes, 18)

    loss = max_loss(logits[test_logits_mask], y_test[test_label_mask], device)
    print('loss:', loss)

    optimizer.zero_grad()
    loss.backward()
    dz = z.grad
    if epoch % 5 == 0:
        lr = lr * lr_decay
    z.data += lr * dz  # bug: 'z += lr * dz'; optimizer.step() is OK either
    z.data = torch.clamp(z.data, args.minf, args.maxf)

    acc = (torch.argmax(logits[test_logits_mask], dim=1) == y_test[test_label_mask]).double().mean()
    print('test acc:', acc.item())
    # print('var:', z_std[0].item() ** 2, 'mean:', z_mean[0].item())
    print('max:', torch.max(z).item(), 'min:', torch.min(z).item())
    print()

print('z[0]:', z[0])

# acc = (torch.argmax(logits[val_mask], dim=1) == y_train[val_mask]).float().mean()
# print(acc.item())

# one fake node attacks 100 test nodes
# 'fake_k': attack test nodes within [100 * size_attack * k, 100 * size_attack * (k+1))
file_path = 'fake_' + str(kk) + '.pkl'
# f = open(file_path, 'wb')
dic = dict()
dic['seed'] = seed
dic['feat'] = z
torch.save(dic, file_path)
