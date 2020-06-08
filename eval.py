import pickle as pkl
from model import GCN
import torch
from data import preprocess_adj
import torch.nn as nn
import numpy as np

y_train = pkl.load(open('../experimental_train.pkl', 'rb'))  # np.ndarray; (543486,); label within [0, 17]
adj = pkl.load(open('../experimental_adj.pkl', 'rb'))  # sparse.csr.csr_matrix (593486, 593486)
X_all = pkl.load(open('../experimental_features.pkl', 'rb'))  # numpy.ndarray (593486, 100)

# preprocess
X_all = torch.from_numpy(X_all)

supports = preprocess_adj(adj)
i = torch.from_numpy(supports[0]).long()  # i.shape = (6810490, 2)
v = torch.from_numpy(supports[1])
support = torch.sparse.DoubleTensor(i.t(), v, supports[2])


num_train_val = y_train.shape[0]
feature_dim = 100
num_classes = 18
num_test = 50000
test_mask = torch.tensor(list(range(50000))) + num_train_val


net = GCN(feature_dim, num_classes)
net.load_state_dict(torch.load('parameters0.48.pkl', map_location=torch.device('cpu')))


net.eval()
out = net((X_all, support))
logits = out[0]  # shape = (num_all, 18)
y_test = torch.argmax(logits[test_mask], dim=1)
print(y_test[:100])
print(y_test.shape)  # (50000, )
f = open('y_test.pkl', 'wb')
pkl.dump(y_test, f)
