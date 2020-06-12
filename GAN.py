import os
import pickle as pkl
import math as m
from col_design import get_col
from model import GCN
from data import preprocess_adj, sparse_symmetric_add
import numpy as np
from torch import optim
from config import args
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


# Load data
y_test = pkl.load(open('./data/y_test.pkl', 'rb'))  # tensor; (50000,); label within [0, 17]
adj = pkl.load(open('../experimental_adj.pkl', 'rb'))  # sparse.csr.csr_matrix (593486, 593486)
X_all = pkl.load(open('../experimental_features.pkl', 'rb'))  # numpy.ndarray (593486, 100)

# Hyper-parameters
lr = args.learning_rate
seed = args.seed
np.random.seed(seed)
torch.random.manual_seed(seed)
latent_size = 10
hidden_1 = 30
hidden_2 = 50
feature_dim = 100
num_classes = 18
num_epochs = 200
batch_size = 500
kd = args.kd
lamda = args.lamda

# mask
num_all = X_all.shape[0]
num_test = 50000
num_train_val = num_all - num_test

# Handle data
X_all = torch.from_numpy(X_all)
size_attack = 500
row = []
for i in range(size_attack):
    row.append([i] * 100)
row = np.concatenate(row) + num_all  # [0,0,0...1,1,1...4,4,4]
# col = np.array(get_col()[:size_attack * 100]) + num_train_val
col = np.array(list(range(size_attack * 100))) + num_train_val
new_adj = sparse_symmetric_add(adj, row, col, num_all + size_attack)
supports = preprocess_adj(new_adj)
i = torch.from_numpy(supports[0]).long()  # i.shape = (6810490, 2)
v = torch.from_numpy(supports[1])  # tensor (6810490, )
adj_shape = supports[2]
# support = torch.sparse.FloatTensor(i.t(), v, adj_shape)
support = torch.sparse.DoubleTensor(i.t(), v, adj_shape)

# Discriminator
D = nn.Sequential(
    nn.Linear(feature_dim, hidden_2),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_2, hidden_1),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_1, 1),
    nn.Sigmoid())

# Generator 
G = nn.Sequential(
    nn.Linear(latent_size, hidden_1),
    nn.ReLU(),
    nn.Linear(hidden_1, hidden_2),
    nn.ReLU(),
    nn.Linear(hidden_2, feature_dim),
    nn.Tanh())

# Device setting
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device(args.dev)
D = D.to(device)
G = G.to(device)
X_all = X_all.to(device)
support = support.to(device)
y_test = y_test.to(device)
test_label_mask = (torch.tensor(list(range(size_attack * 100)))).to(device)
test_logits_mask = (test_label_mask + num_train_val).to(device)

# Binary cross entropy loss and optimizer
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=lr)
g_optimizer = torch.optim.Adam(G.parameters(), lr=lr)

# Load model
model = GCN(feature_dim, num_classes)
model.load_state_dict(torch.load('./data/param_eye1_39.pkl', map_location=device))
model = model.to(device)
model.eval()
# optimizer = optim.Adam(, lr=args.learning_rate)


def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()


def max_loss(logits, test_labels, dev):
    a = F.softmax(logits, dim=1).to(dev)
    # a = logits
    loss = torch.sum(torch.max(a, dim=1)[0] - a[range(len(test_labels)), test_labels]).to(dev)
    return loss


# Start training
for epoch in range(args.epochs):
    # Create the labels which are later used as input for the BCE loss
    real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)

    # ================================================================== #
    #                      Train the discriminator                       #
    # ================================================================== #
    for _ in range(kd):
        # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        # Second term of the loss is always zero since real_labels == 1
        idx = np.random.choice(num_all, batch_size)
        X_sample = X_all[idx].float()
        outputs = D(X_sample)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs

        # Compute BCELoss using fake images
        # First term of the loss is always zero since fake_labels == 0
        z = torch.randn(batch_size, latent_size).to(device)
        fake_nodes = 2.5 * G(z)
        outputs = D(fake_nodes)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs

        # Backprop and optimize
        d_loss = d_loss_real + d_loss_fake
        reset_grad()
        d_loss.backward()
        d_optimizer.step()

    # ================================================================== #
    #                        Train the generator                         #
    # ================================================================== #

    # Compute loss with fake images
    # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
    # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
    z = torch.randn(batch_size, latent_size).to(device)
    fake_nodes = G(z) * 2.5
    outputs = D(fake_nodes)
    g_loss_fake = criterion(outputs, real_labels)

    # g_loss_acc
    X_attack = torch.cat((X_all, fake_nodes.double()), dim=0).to(device)
    outputs = model((X_attack, support))
    logits = outputs[0]
    g_loss_acc = max_loss(logits[test_logits_mask], y_test[test_label_mask], device)

    # Backprop and optimize
    g_loss = g_loss_fake - lamda * g_loss_acc
    reset_grad()
    g_loss.backward()
    g_optimizer.step()

    print('Epoch [{}]'.format(epoch))
    print('d_loss_real: {:.4f}, d_loss_fake: {:.4f}, d_loss: {:.4f}'
          .format(d_loss_real.item(), d_loss_fake.item(), d_loss.item()))
    print('g_loss_fake: {:.4f}, g_loss_acc: {:.4f}, g_loss: {:.4f}'
          .format(g_loss_fake.item(), g_loss_acc.item(), g_loss.item()))
    print()

print("Finish")
z = torch.randn(batch_size, latent_size).to(device)
fake_nodes = G(z)
print(fake_nodes[:5])
np.save('feature.npy', fake_nodes.detach().cpu().numpy())
'''exit(0)
# Save the model checkpoints
torch.save(G.state_dict(), 'G.ckpt')
torch.save(D.state_dict(), 'D.ckpt')'''