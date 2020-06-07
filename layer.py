import  torch
from    torch import nn
from    torch.nn import functional as F
from    utils import sparse_dropout, dot


class GraphConvolution(nn.Module):


    def __init__(self, input_dim, output_dim,
                 dropout=0.,
                 is_sparse_inputs=False,
                 bias=False,
                 activation=F.relu,
                 featureless=False):
        super(GraphConvolution, self).__init__()


        self.dropout = dropout
        self.activation = activation
        self.is_sparse_inputs = is_sparse_inputs
        self.featureless = featureless

        self.weight = nn.Parameter(torch.randn(input_dim, output_dim).double())
        self.bias = nn.Parameter(torch.zeros(output_dim).double())

    def forward(self, inputs):
        # print('inputs:', inputs)
        x, support = inputs

        if self.training:
            x = F.dropout(x, self.dropout)

        # convolve
        xw = torch.mm(x, self.weight)

        out = torch.sparse.mm(support, xw)

        if self.bias is not None:
            out += self.bias

        return self.activation(out), support

