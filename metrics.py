import numpy as np
import torch


def mse_score(output, target):

    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
        
    difference_array = np.subtract(output, target)
    squared_array = np.square(difference_array)
    mse = squared_array.mean()

    return mse
