import os
import shutil
import time
import pprint

import torch


def calculate_dist(a, b):
    n = a.shape[0]
    m = b.shape[0]
    h = a.shape[1]
    assert h == b.shape[1]

    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    #a: [111111111,222222222,333333333,444444444,555555555]
    #b: [123456789,123456789,123456789,123456789,123456789]
    #return [n*m]
    return -torch.pow(a-b,2).sum(2)
