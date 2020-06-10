import pickle
import numpy as np
from scipy.sparse import csr_matrix
from col_design import get_col


num_train = 543486
num_all = 593486

row = []
for i in range(500):
    row = row + [i] * 100
row = np.array(row)

# col = list(range(50000))
# col = np.array(col) + num_train
col = np.array(get_col()) + num_train

print(row[:50], col[:50])
data = np.ones(50000)
res = csr_matrix((data, (row, col)), shape=(500, 500+num_all))

f = open('./result/adj.pkl', 'wb')
pickle.dump(res, f)

