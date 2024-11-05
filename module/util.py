import numpy as np
import math
import time
from scipy.linalg import block_diag, schur
from numpy import linalg as LA
import mpmath as mp


def round_up_to_even(f):
    return math.ceil(f / 2.) * 2

def spec_radius(W):
    '''
    # This is more stable, uses svd
    '''
    return np.linalg.norm(W, ord=2)

def check_orthogonal(M):
    '''
    Checks if M is orthogonal
    '''

    return np.allclose(M@M.T, np.eye(M.shape[0])) and np.allclose(M.T@M, np.eye(M.shape[0]))

def canonical_projection_mtx(n, p):
    """
    Matrix of canonical projection from K^n to K^p, p < n
    """
    if p > n:
        raise ValueError("p cannot be greater than n.")

    # Creating an n x n zero matrix
    matrix = np.zeros((n, p), dtype=int)

    # Setting the diagonal elements to 1 for the first p rows
    for i in range(p):
        matrix[i][i] = 1

    return matrix

def apply_P(X, P, n_res):
    return (np.transpose(P) @  X)[:n_res]

def rot_angles(A, _check=False, _schur=True, timer=False, single_angle = False, pm1_threshold = 1e-8):
    '''
    Given orthogonal matrix A, do real Schur decomposition to get canonical block diagonal T (and orthogonal Z)
    With T, extract all the rotations in the canonical form

    Note: _schur is faster (the ``proper way").

    Added single angle mode, doesn't return the flipped angles.
    '''

    # Use Real Schur's decomposition to get the block-diagonal canonical form
    # This is the bottle neck run time...O(n^3)
    start = time.time()
    T, _ = schur(A)
    end = time.time()
    if timer:
        print('Schur time: ', end - start)

    # The ``proper way" -- with real Schur decomposition
    if _schur:
        start3 = time.time()

        # Extract indices
        dt = np.diag(T)
        idx_non1 = np.where(np.abs(np.abs(dt) - 1) > pm1_threshold)[0]
        # print(idx_non1)
        # Index for 1
        idx_1 = np.where(np.abs(dt - 1) < pm1_threshold)[0]
        # Index for - 1
        idx_n1 = np.where(np.abs(dt + 1) < pm1_threshold)[0]

        # Get rotational angle of each block
        t_list = []
        angle_list = []

        # print(len(idx_non1))

        # Extra check for validity
        if A.shape[0] % 2 == 0:
            try:
                assert (len(idx_1) == 1 and len(idx_n1) == 1)
                # print('Even case ok')
            except AssertionError:
                print('Upsilon missing in EVEN case')
                # print(len(idx_1), len(idx_n1))
        else:
            try:
                assert (len(idx_1) == 1 and len(idx_n1) == 0)
                # print('Odd case ok')
            except AssertionError:
                print('Upsilon missing in ODD case')
        # print(len(idx_1))
        # print(len(idx_n1))

        if len(idx_non1) >= 2:

            for i in range(0, len(idx_non1) - 1, 2):
                #             print(idx_non1[i], idx_non1[i+1])
                ia = idx_non1[i]
                ib = idx_non1[i + 1] + 1

                # print(ia,ib)

                # The current block
                T_block = T[ia:ib, ia:ib]

                # atan2 is between -pi and pi
                block_angle = math.atan2(T_block[1, 0], T_block[0, 0])

                angle_list.append(block_angle)

                if not(single_angle):
                    angle_list.append(-1 * block_angle)

                t_list.append(T_block)
            # print(len(angle_list))
        # Fill-in the \pm 1's
        # for i in idx_1:
        # print(len(idx_1), len(idx_n1))
        for i in range(len(idx_1)):
            t_list.append(np.array([1]))
        for i in range(len(idx_n1)):
            t_list.append(np.array([-1]))


        if len(idx_1) > 0:
            angle_list.append(0)
            angle_list.append(2*np.pi)
            # Here we don't append -0 which is 2*pi by symmetry.
        # for i in idx_n1:
        if len(idx_n1) > 0:

            angle_list.append(math.pi)

        angle_list = np.array(angle_list)

        end3 = time.time()
        if timer:
            print('Non-eigen time: ', end3 - start3)

        if _check:
            Tf = block_diag(*t_list)

            # Check if they coincide

            einit, _ = LA.eig(T)
            #         print(einit)
            #         print(len(einit))

            idinit = einit.argsort()[::-1]
            einit = einit[idinit]

            ef, _ = LA.eig(Tf)
            #         print(ef)
            #         print(len(ef))

            idf = ef.argsort()[::-1]
            ef = ef[idf]

            #         print(einit)
            #         print(ef)
            # Should be equal

            print('Eigenvalues equal?')
            print(np.allclose(einit, ef))
    angle_list = angle_list % (2 * math.pi)
    if not single_angle:
        return angle_list
    else:
        return np.array(list(dict.fromkeys(angle_list)))


def binary_pi(n_digits):
    """
    Binary expansion of pi, align with https://oeis.org/A004601
    """
    mp.dps = int(n_digits * 3.32) + 10  # Set decimal places needed for binary precision

    # Integer Part
    integer_part = int(mp.pi)
    binary_integer_part = [int(bit) for bit in bin(integer_part)[2:]]  # Convert to list of integers

    # Fraction part
    binary_fractional_part = []
    fractional_part = mp.pi - integer_part

    for _ in range(n_digits):
        fractional_part *= 2
        digit = int(fractional_part)
        binary_fractional_part.append(digit)
        fractional_part -= digit

    # Combine together
    binary_combined = np.array(binary_integer_part + binary_fractional_part, dtype=int)
    return binary_combined