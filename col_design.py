import numpy as np
import pickle as pkl
from config import args


def get_col():
    # row: ([0] * 100 + [1] * 100 ... + [499] * 100)
    # return col (list) [0, 50000)
    '''num_class = 18
    dic = dict()
    y_test = pkl.load(open('./data/y_test.pkl', 'rb')).tolist()  # tensor; (50000,); label within [0, 17]
    for i, label in enumerate(y_test):
        if label not in dic.keys():
            dic[label] = []
        dic[label].append(i)

    res = []
    for i in range(num_class):
        if i in dic.keys():
            res = res + dic[i]'''

    res = list(range(50000))
    return res


