import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

import numpy as np
import pickle as pkl
from data import preprocess_adj
from model import GCN
from config import args

# TODO:
# feel free to change seed or not
# hyperparameters tuning: config.py   priority: dropout > learning_rate > hidden
seed = args.seed
np.random.seed(seed)
torch.random.manual_seed(seed)

###############################################
# load data
y_train = pkl.load(open('../experimental_train.pkl', 'rb'))  # np.ndarray; (543486,); label within [0, 17]
adj = pkl.load(open('../experimental_adj.pkl', 'rb'))  # sparse.csr.csr_matrix (593486, 593486)
X_all = pkl.load(open('../experimental_features.pkl', 'rb'))  # numpy.ndarray (593486, 100)


################################################
# preprocess
# TODO: use cuda or gpu
# device = torch.device('cuda')
device = torch.device(args.dev)
print(device)
X_all = torch.from_numpy(X_all).to(device)

supports = preprocess_adj(adj)
i = torch.from_numpy(supports[0]).long().to(device) # i.shape = (6810490, 2)
v = torch.from_numpy(supports[1]).to(device)
support = torch.sparse.DoubleTensor(i.t(), v, supports[2]).to(device)

# res = torch.sparse.mm(support, X_all)
# print(res.shape)  # (593486, 100) success

y_train = torch.from_numpy(y_train).long().to(device)  # why long?

# mask
num_all = X_all.shape[0]
num_train_val = y_train.shape[0]
num_train = round(num_train_val * 0.8)
num_val = num_train_val - num_train
num_test = 50000
perm = torch.randperm(num_train_val)
train_mask = perm[:num_train].to(device)
val_mask = perm[num_train:num_train_val].to(device)
test_mask = (torch.randperm(num_test) + num_train_val).to(device)


# train
feature_dim = X_all.shape[1]
num_classes = 18

net = GCN(feature_dim, num_classes)
net.to(device)
optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss()

net.train()
for epoch in range(args.epochs):

    out = net((X_all, support))
    logits = out[0]
    # logp = F.log_softmax(logits, 1)
    # loss = F.nll_loss(logp[train_mask], y_train[train_mask])
    loss = criterion(logits[train_mask], y_train[train_mask])
    # TODO:
    # try: loss += args.weight_decay * net.l2_loss()  # L2 penalization

    acc = (torch.argmax(logits[val_mask], dim=1) == y_train[val_mask]).float().mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0 or epoch == args.epochs - 1:
        print(epoch, loss.item(), acc.item())

# TODO: save parameters
torch.save(net.state_dict(), './data/params.pkl')
'''net.eval()

out = net((X_all, support))
logits = out[0]
y_pred = torch.argmax(logits[test_mask], dim=1)
print('y_pred:', y_pred)'''

