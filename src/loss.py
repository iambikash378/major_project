from torch import flatten
import torch

smooth = 1

def dsc(y_true, y_pred):

    y_true_flattened = flatten(y_true)
    y_pred_flattened = flatten(y_pred)

    intersection = torch.sum(y_true_flattened*y_pred_flattened, smooth = 1)
    coeff = (2. * intersection+smooth) / (torch.sum(y_true_flattened) + torch.sum(y_pred_flattened)+smooth)
    return coeff


def dice_loss(y_true, y_pred):
    loss = 1 - dsc(y_true, y_pred)
    return loss


