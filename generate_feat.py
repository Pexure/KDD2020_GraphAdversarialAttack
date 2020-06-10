import pickle as pkl
import math
import numpy as np
import torch
from scipy.sparse import csr_matrix

'''f = np.load('./result/feature.npy')
print(f.shape)
print(f[0])
exit(0)'''
# f.shape = (500, 100)

# fake_k.pkl to feature.npy
res = []
for i in range(1):
    file = './data/fake_' + str(i) + '.pkl'
    f = open(file, 'rb')
    dic = pkl.load(f)
    res.append(dic['feat'])
    # print(dic['feat'][0])

res = torch.cat(res, dim=0).cpu()
res = res.detach().numpy()
print(res[0])
np.save('./result/feature.npy', res)




# naive features
'''f = np.zeros((500, 100))
x = np.zeros(100)
x[:50] -= 50
f += x
print(f[0])
np.save('feature.npy', f)'''


'''f = open('adj.pkl', 'rb')
mat = pickle.load(f)
print(mat[0].nonzero()[1])
print(mat[1].nonzero()[1])'''
