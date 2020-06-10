import  numpy as np
import  pickle as pkl
import  networkx as nx
import  scipy.sparse as sp
from config import args


def sparse_symmetric_add(sparse_mx, row, col, size):
    mx = sparse_mx.tocoo()
    # mx.row: ndarray (6217004, )
    new_row = np.concatenate([mx.row, row, col])
    new_col = np.concatenate([mx.col, col, row])
    data = np.ones(new_row.shape[0])
    ret = sp.csr_matrix((data, (new_row, new_col)), shape=(size, size))
    return ret


def sparse_to_tuple(sparse_mx):
    """
    Convert sparse matrix to tuple representation.
    """
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    # adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)) # D
    #print(np.argwhere((rowsum == 0)))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten() # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt) # D^-0.5
    # return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo() # D^-0.5AD^0.5
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt) # D^-0.5AD^0.5


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    # adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    # return sparse_to_tuple(adj_normalized)
    adj_normalized = normalize_adj(adj + args.eye * sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)
