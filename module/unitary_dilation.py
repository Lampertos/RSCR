import numpy as np
from scipy.linalg import sqrtm
from scipy.linalg import block_diag
from module.util import spec_radius

def dilation(W, N, field='C', custom_unitary=[], verbose = False):
    '''
    Construct unitary dilation given matrix and dilation power.

    Inputs:
      - W (n_res x n_res): square matrix with norm leq 1 (reservoir coupling matrix)
      - N (int): maximum dilation power
      - Field (str): 'C' for complex (unitary output), 'R' for real (orthogonal output)

    Outputs:
      - U ((N+1) * n_res x (N+1) * n_res):
    '''

    if len(custom_unitary) == 0:
        if verbose:
            print('Default dilation with identity, no custom unitary')

    if len(custom_unitary) != 0 and len(custom_unitary) != 1 and len(custom_unitary) != (N - 1):
        raise ValueError(
            "The input list of unitarys must be either 1 (where the identities are all replaced by that single "
            "element) or N - 1 (where each one is defined by user.)")

        if custom_unitary[0].shape != W.shape:
            raise ValueError("The internal blocks must be of the same shape")

    # Set up W
    if type(W) != np.array:
        W = np.array(W)

    if field == 'C':
        if W.dtype != 'complex128':
            W = W.astype('complex128')
        output_dtype = complex
    elif field == 'R':
        if W.dtype != 'float64':
            W = W.astype('float64')
        output_dtype = float

    n_res = W.shape[0]

    # Halmos dilation -- components

    Wstar = W.conj().T

    # For random matrices, sqrt may result in change of type.
    DW = sqrtm(np.eye(n_res, dtype=output_dtype) - Wstar @ W)
    DWstar = sqrtm(np.eye(n_res, dtype=output_dtype) - W @ Wstar)

    # Halmos dilation, for debugging to check that it's indeed unitary
    # Remember W must be contractive. For n_res = 2, N = 2 this works
    # U_Halmos = np.block([[W , DWstar], [DW, - Wstar]])
    # print(U_Halmos.dot(U_Halmos.conj().T))

    # Egerv\'{a}ry dilation ~~ Sz Nagy in finite dimensions:
    # First two rows of U
    r1 = [W]
    r2 = [DW]

    # (N + 1) blocks of n_res x n_res, -1 front and -1 last column, so we add N-1 blocks
    # and two extra blocks of n_res
    for i in range(N - 1):
        r1.append(np.zeros_like(W))
        r2.append(np.zeros_like(W))
    r1.append(DWstar)
    r2.append(-1 * Wstar)
    r1 = np.block([r1])
    r2 = np.block([r2])

    # Rest of U is a 1. zero N x N column, 2. N identity, and 3. another first column
    zero_col = np.zeros((n_res, (N - 1) * n_res))

    # Default dilation, changed to form below thus commented for bookkeeping
    # bot = np.block([[zero_col], [np.eye((N - 1) * n_res, dtype=output_dtype)], [zero_col]]).transpose()

    # Custom unitary diagonals for Boyu to play with
    if len(custom_unitary) == 0:
        internal_block = np.eye((N - 1) * n_res, dtype=output_dtype)
    # Already checked the correct lengths
    elif len(custom_unitary) == 1:
        # Duplicate the inside N - 1 times
        # custom_unitary = custom_unitary * N - 1
        # Need to transpose the block since the subsequent step block is done in transpose
        # * in front is passing tuples: https://stackoverflow.com/questions/43567259/block-diagonal-of-list-of-matrices
        internal_block = block_diag(*(custom_unitary * (N - 1))).transpose()
    elif len(custom_unitary) == N - 1:
        internal_block = block_diag(*custom_unitary).transpose()
    else:  # Still raise error just in case
        raise ValueError(
            "The input list of unitarys must be either 1 (where the identities are all replaced by that single "
            "element) or N - 1 (where each one is defined by user.)")

    # Transpose because blocks are done in rows
    bot = np.block([[zero_col], [internal_block], [zero_col]]).transpose()

    # Combine to form entire U
    U = np.block([[r1], [r2], [bot]])

    return U

def check_if_can_dilate(W, dilation_N = 2):
    '''
    For randomly generated W, there are instances where the type changes from float to complex when doing dilation
    This is fine for dilation theory but for our subsequent process of root completion this has to be avoided.
    This function checks whether there are complex components in the square root term in the dilation
    '''

    n_res = W.shape[0]

    W = W / spec_radius(W)

    Wstar = W.conj().T

    # For random matrices, sqrt may result in change of type.
    DW = sqrtm(np.eye(n_res, dtype=float) - Wstar @ W)
    DWstar = sqrtm(np.eye(n_res, dtype=float) - W @ Wstar)

    # Ur = dilation(W / spec_radius(W) , dilation_N, field = 'R')
    # print(Wstar.dtype)
    # print(DW.dtype)
    # print(DWstar.dtype)
    #
    if Wstar.dtype == 'complex128' or DW.dtype == 'complex128' or DWstar.dtype == 'complex128':
        # or Ur.dtype == 'complex128'):
        return False
    else:
        return True