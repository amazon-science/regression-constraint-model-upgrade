import torch
import torch.nn.functional as F

def linear_combination(x, y, epsilon):
    return epsilon*x + (1-epsilon)*y

def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss

class LabelSmoothingCrossEntropy(torch.nn.Module):
    def __init__(self, epsilon:float=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss/n, nll, self.epsilon)

class EnsModel(torch.nn.Module):
    '''
    TODO: this is a temporary implementation, check the original one
    '''
    def __init__(self, module_list, weight_list=None):
        super().__init__()
        assert len(module_list) > 0
        self.module_list = module_list
        if weight_list is None:
            self.weight_list = [1,] * len(module_list)
        else:
            assert len(module_list) == len(weight_list)
            self.weight_list = weight_list

    def forward(self, x):
        for i, module in enumerate(self.module_list):
            output_tp = module(x)
            if not isinstance(output_tp, tuple):
                if i == 0:
                    output = output_tp.detach()
                else:
                    output += output_tp.detach() * self.weight_list[i]
                continue
            if i == 0:
                output, feature = output_tp[0].detach(), output_tp[1].detach()
            else:
                output += output_tp[0].detach() * self.weight_list[i]
                feature += output_tp[1].detach() * self.weight_list[i] if output_tp[1].size(1) == feature.size(1) else 0
        if not isinstance(output_tp, tuple):
            return output / sum(self.weight_list)
        return output / sum(self.weight_list), feature / sum(self.weight_list)



