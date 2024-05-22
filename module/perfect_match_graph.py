import numpy as np
import math
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.linalg import schur


def approximateTu(Tu, epsilon=0.001, increment=1):
    '''
    Tu must be in the block diagonal form
    it will return two orthogonal matrices S and D, S-T should have small norm, D contains all the missing blocks
    '''

    Tu, Ju = schur(Tu, output='real')

    if not np.allclose(Ju, np.eye(Ju.shape[0])):
        print('Warning: Not in block diagonal form, need to do it outside.')

    x = []
    d = len(Tu)
    i = 0
    flag_p1 = 0
    flag_m1 = 0

    pone_index = -1
    mone_index = -1
    P_pm_one = np.eye(d)
    while i < d:
        if (not np.isclose(Tu[i, i], 1)) and (not np.isclose(Tu[i, i], -1)):
            x.append(Tu[i, i])
            i += 2
        elif np.isclose(Tu[i, i], 1):
            if flag_p1 == 0:
                flag_p1 = 1
                pone_index = i
            else:
                flag_p1 = 0
                x.append(1)
                if i - pone_index > 1:
                    print("Warning: The +1s in Schur Form are not consecutive. May need to multiply P_pm_one.")
                    j = i
                    while j < pone_index - 1:
                        P_pm_one[j, j] = 0
                        P_pm_one[j + 1, j] = 1
                        j += 1
                    P_pm_one[pone_index - 1, pone_index - 1] = 0
                    P_pm_one[i, pone_index - 1] = 1
                pone_index = -1
            i += 1
        else:
            if flag_m1 == 0:
                flag_m1 = 1
                mone_index = i
            else:
                flag_m1 = 0
                x.append(-1)
                if i - mone_index > 1:
                    print("Warning: The -1s in Schur form are not consecutive. May need to multiply P_pm_one.")
                    j = i
                    while j < mone_index - 1:
                        P_pm_one[j, j] = 0
                        P_pm_one[j + 1, j] = 1
                        j += 1
                    P_pm_one[mone_index - 1, mone_index - 1] = 0
                    P_pm_one[i, mone_index - 1] = 1
                mone_index = -1
            i += 1
    # print(x, flag_m1, flag_p1)
    n = len(x)
    if 2 * n + flag_m1 + flag_p1 < d:
        print("Warning: potential multiple eigenvalues of 1/-1, n =", n, "flag_p1 =", flag_p1, "flag_m1 =", flag_m1)
    thetas = np.arccos(x)
    theta_min = np.arccos(1 - (epsilon ** 2) / 2)

    bigN = n
    graph = []
    found = 0
    while found == 0:
        graph = []
        for i in range(n):
            adj = [0] * (bigN + 1)
            lower = max(math.ceil((thetas[i] - theta_min) / np.pi * bigN), 1)
            upper = min(math.floor((thetas[i] + theta_min) / np.pi * bigN), bigN - 1)
            for j in range(lower, (upper + 1)):
                adj[j] = 1
            graph.append(adj)
        g = csr_matrix(graph)
        matching = maximum_bipartite_matching(g, perm_type='column')
        if len(matching[matching == -1]) == 0:
            # print("done, dimension=", bigN * 2)
            found = 1
        else:
            bigN += increment
    approx = np.cos(matching / bigN * np.pi)
    j = 0
    i = 0
    S = []
    D = []
    while i < d:
        row1 = [0] * d
        row2 = [0] * d
        if i == pone_index:
            row1[i] = 1
            i += 1
            S.append(row1)
        elif i == mone_index:
            row1[i] = -1
            i += 1
            S.append(row1)
        else:
            a = approx[j]
            b = math.sqrt(1 - a ** 2)
            if Tu[i + 1, i] > 0:
                row1[i] = a
                row1[i + 1] = -b
                S.append(row1)
                row2 = [0] * d
                row2[i] = b
                row2[i + 1] = a
                S.append(row2)
            else:
                row1[i] = a
                row1[i + 1] = b
                S.append(row1)
                row2[i] = -b
                row2[i + 1] = a
                S.append(row2)
            j += 1
            i += 2
    i = 1
    j = 0
    dm = (bigN - 1 - n) * 2 + (2 - flag_m1 - flag_p1)
    while i <= bigN - 1:
        row1 = [0] * dm
        row2 = [0] * dm
        if len(matching[matching == i]) == 0:
            a = np.cos(i / bigN * np.pi)
            b = math.sqrt(1 - a ** 2)
            row1[j] = a
            row1[j + 1] = -b
            row2[j] = b
            row2[j + 1] = a
            j += 2
            D.append(row1)
            D.append(row2)
        i += 1
    # print("len(D)=", len(D), "n=",n,"bigN=",bigN,"flag_p1 =", flag_p1, "flag_m1 =", flag_m1)
    if flag_p1 == 0:
        row1 = [0] * dm
        row1[(bigN - 1 - n) * 2] = 1
        D.append(row1)
    if flag_m1 == 0:
        row1 = [0] * dm
        row1[(bigN - 1 - n) * 2 + (1 - flag_p1)] = -1
        D.append(row1)

    return (np.array(S), np.array(D), P_pm_one)