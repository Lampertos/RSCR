import numpy as np


def ortho_xform(A1, A2, rtol = 1e-10, atol=1e-6):
    '''
    Assume A1 and A2 are canonical form of orthogonal matrices with similar but shuffled blocks ( \pm1 or 2x2 blocks)
    Generate orthogonal matrix such that P @ A1 @ P.T = A2

    Note: when the original coupling matrix W's eigenvalues are too close to 0 we get numerical percision error.
    '''
    n = A1.shape[0]

    P = np.zeros_like(A1)

    for i in range(n):
        if np.allclose(A1[i, i], 1, rtol = rtol, atol = atol) or np.allclose(A1[i, i], -1, rtol = rtol, atol = atol):
            for j in range(n):
                if np.allclose(A2[j, j], A1[i, i], rtol = rtol, atol = atol) and np.all(P[j, :] == 0):
                    P[j, i] = 1
                    break
        else:
            for j in range(n - 1):
                if np.allclose(A2[j, j], A1[i, i], rtol = rtol, atol = atol) and np.all(P[j, :] == 0):
                    if np.allclose(A2[j + 1, j], A1[i + 1, i], rtol = rtol, atol = atol):
                        P[j, i] = 1
                        P[j + 1, i + 1] = 1
                    else:
                        P[j, i + 1] = 1
                        P[j + 1, i] = 1
                    i += 1
                    break
    return P
