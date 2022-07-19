from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
from torch_geometric.data import DataLoader, Data

###############################################################################
#
#                           Optimizer
#
###############################################################################

class Optimizer:
    
    ###########################################################################
    def __init__(
            self,
            net : torch.nn.Module,
            loss:str="decoding"
        ):
        if loss in ["decoding"]:
            self.optimizer = torch.optim.Adam(
                net.parameters(),
                lr = 2E-4,
                betas = (0.5, 0.999)
            )
        elif loss in ["prediction"]:
            self.optimizer = torch.optim.Adam(
                net.parameters(),
                lr = 1E-3,
                betas = (0.5, 0.999)
            )
        elif loss in ["RMSProp"]:
            self.optimizer = torch.optim.RMSprop(
                net.parameters(),
                lr = 1E-4,

            )
        elif loss in ["latency_prediction"]:
            self.optimizer = torch.optim.Adam(
                net.parameters(),
                lr = 2E-4,
                betas = (0.5, 0.999), 
                weight_decay = 1e-4
            )
        else:
            self.optimizer = torch.optim.Adam(
                net.parameters(),
                lr = 1E-4,
                betas = (0.0, 0.9)
            )


###############################################################################
#
#                           Loss
#
###############################################################################

class Loss(nn.Module):

    ###########################################################################
    def __init__(
            self, 
            device = 'cpu',  
            criterion_acc = torch.nn.MSELoss()
        ):

        super(Loss,self).__init__()

        self.device = device

        self.criterion_acc = criterion_acc
        self._loss_acc = self._acc


    ###########################################################################
    def _acc(self, y:torch.tensor, target:torch.tensor, **kwargs):
        return self.criterion_acc(y, target)

    ###########################################################################  
