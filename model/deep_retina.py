import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torch.autograd import Variable
from model.modules import PSoftPlus

import numpy as np


class DeepRetina(nn.Module):
    def __init__(self, nw=70, nh=70, nl=5,
                 n_filters=(15, 9),
                 kernel_size=(15, 11),
                 n_cell=1,
                 dropout=.5
                 ):
        """
        Reimplementation of deep_retina convolutional network in pytorch
        Reference: McIntosh et. al.(2016). Deep learning models of the retinal response to natural scenes
        Resource: https://github.com/baccuslab/deep-retina
        Args:
            - nW,nH: stimulation frame size in pixels
            - nL: Length of temporal filter in the first layer
            - nFilters: Number of filters in the first and second layer as a tuple
            - kernel_size: size of filters in the first and second layer as a tuple
            - nCell: Number of cells in the model
            - wts_list: list of filters, where each element in the list corresponds a layer
            - bias_list: list of bias terms, where each element in the list corresponds a layer
            - act_function: pointer to activation function for fully-connected layer
            - dropout: probability of dropout of first and second layer
        """
        super(DeepRetina, self).__init__()

        self.nW = nw
        self.nH = nh
        self.nL = nl
        self.nFiltL1 = n_filters[0]
        self.nFiltL2 = n_filters[1]
        self.szFiltL1 = kernel_size[0]
        self.szFiltL2 = kernel_size[1]
        self.nCell = n_cell
        self.l3_filt_shape = None

        self.conv1 = nn.Conv2d(in_channels=self.nL,
                               out_channels=self.nFiltL1,
                               kernel_size=(self.szFiltL1, self.szFiltL1),
                               stride=1,
                               padding=0,
                               bias=True)

        self.conv2 = nn.Conv2d(in_channels=self.nFiltL1,
                               out_channels=self.nFiltL2,
                               kernel_size=(self.szFiltL2, self.szFiltL2),
                               stride=1,
                               padding=0,
                               bias=True)

        in_shp, conv2outShape = self._compute_fc_in()
        self.l3_filt_shape = (self.nCell, *conv2outShape)

        self.fc = nn.Linear(in_features=in_shp,
                            out_features=self.nCell,
                            bias=True)

        self.dropout = nn.Dropout2d(p=dropout)
        self.act_function = PSoftPlus()

    def forward(self, inp):
        x = self.dropout(F.relu(self.conv1(inp.float())))
        x = self.dropout(F.relu(self.conv2(x)))
        x = self.act_function(self.fc(x.view([x.shape[0], -1])))
        return x

    def _compute_fc_in(self):
        xx = np.random.random([1, self.nL, self.nW, self.nH])
        xx = torch.from_numpy(xx)
        xx = self.conv1(xx.float())
        xx = self.conv2(xx)
        conv2shape = xx.size()[1:]
        xx = xx.view(xx.size(0), -1)
        return xx.size(1), conv2shape

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def save(self, path):
        print('Saving model... %s' % path)
        torch.save(self, path)

