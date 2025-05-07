import numpy as np
from scipy.linalg import qr


def QR_positive(A: np.array):
    """ compute the unique (positive diagonal entries of R) QR decomposition of a matrix

    Args:
        A (np.array): the matrix whose QR decomposition is to be computed

    Returns:
        Q (np.array): the orthonormal matrix Q
        R (np.array): the upper triangular matrix R with positive entries on the diagonal
    """

    assert (A.shape[0] == A.shape[1]) and (len(A.shape) == 2)

    n = A.shape[0]

    Q, R = qr(A)

    # check the diagonal entries
    for i in range(n):
        if R[i, i] < 0: # time to flip
            # flipping the column of R and the row of Q
            R[:, i] *= -1.0
            Q[i, :] *= -1.0


    return Q, R