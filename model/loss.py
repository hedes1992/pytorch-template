import torch.nn.functional as F
import torch


def nll_loss(output, target):
    return F.nll_loss(output, target)
def nll_loss_reweight1(output, target):
#    print('nll reweight1')
    return F.nll_loss(output, target, \
            weight=torch.cuda.FloatTensor([7.0, 1.5, 1.0, 4.0, 2.0, \
            1.0, 10.0, 1.0, 2.5, 3.0, \
            1.0, 3.5, 4.0, 1.0, 15.0, 4.5, 1.0]))
