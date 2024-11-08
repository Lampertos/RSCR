import numpy as np

from mpmath import mp
import torch
from module.util import spec_radius, binary_pi

if torch.cuda.is_available():
    dev = torch.device("cuda")
elif torch.backends.mps.is_available():
    dev = torch.device("mps")
else:
    dev = torch.device("cpu")
device = torch.device(dev)


def cycle(n_res, **kwargs):
    spec_rad = kwargs.get('spectral_radius', 0.95)

    W = np.zeros((n_res, n_res))
    off_diag = np.ones((1, n_res - 1))[0]
    W = W + np.diag(off_diag, -1)
    W[0][n_res - 1] = 1  # upper righthand corner

    W *= spec_rad

    return W

get_bin = lambda x: format(x, 'b')
def reservoir(n_res, n_in, **kwargs):
    spec_rad = kwargs.get('spectral_radius', 0.95)
    rin = kwargs.get('rin', 0.05)

    win_rand = kwargs.get('win_rand', 0)

    W = np.zeros((n_res, n_res))
    off_diag = np.ones((1, n_res - 1))[0]
    W = W + np.diag(off_diag, -1)
    W[0][n_res - 1] = 1  # upper righthand corner

    radius = spec_radius(W)
    W *= spec_rad / radius


    Win = np.ones((n_res, n_in))

    if win_rand == 1:
        Win = np.random.rand(n_res, n_in)
    else:
        V = binary_pi(n_res)[:n_res] * 2 - 1
        V = V[:n_res]
        Win = V
    Win *= rin


    return W, Win

def reshape(x, y, B):
    '''Reshape tensors x and y from (B, d, n) to (B, d*n)'''
    x = np.reshape(x, (B, -1), order='C')
    y = np.reshape(y, (B, -1), order='C')
    return x, y


