import pickle as pkl
import math
import numpy as np
import torch
from scipy.sparse import csr_matrix

res = np.load('./result/feature.npy')
print(res.shape)
print(res[0])
exit(0)
# res.shape = (500, 100)

# fake_k.pkl to feature.npy
'''res = []
for i in range(1):
    file = 'fake_' + str(i) + '.pkl'
    dic = torch.load(file)
    res.append(dic['feat'])
    # print(dic['feat'][0])

res = torch.cat(res, dim=0).cpu()
print(res[0])
res = res.detach().numpy()
res[res>0] = 99.99
res[res<0] = -99.99
print(res[0])'''
np.save('./result/feature.npy', res * 99.99)




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
