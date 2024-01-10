import numpy as np


def DCSSOMP(Y, A, L):
    # Get dimensions
    K = Y.shape[1]  # number of channels
    N = Y.shape[0]  # observations per channel
    M = A.shape[1]  # size of sparse vector (M >> N)

    if L <= 0:
        L = N

    # 1. Initialization
    R = Y  # residual
    psi = np.zeros((N, L, K), dtype=complex)
    indices = np.zeros(L, dtype=int)
    columns = np.zeros((N, L, K), dtype=complex)
    betamatrix = np.zeros((L, K), dtype=complex)

    for counter in range(L):
        # 2. Find maximum correlation between residual and columns of A
        cost = np.zeros(M)
        for m in range(M):
            for k in range(K):
                # Calculate cost for each column m of A and each channel k
                cost[m] = cost[m] + np.abs(np.dot(A[:, m, k].conj().T, R[:, k])) / np.linalg.norm(A[:, m, k], 2)

        maxi = np.argmax(cost)
        indices[counter] = maxi

        for k in range(K):
            # 3. Orthogonalize
            columns[:, counter, k] = A[:, maxi, k]
            omega = A[:, maxi, k]
            psi[:, counter, k] = omega

            for counter2 in range(counter):
                # Gram-Schmidt orthogonalization
                psi[:, counter, k] = psi[:, counter, k] - (
                        np.dot(psi[:, counter2, k].conj().T, omega) * psi[:, counter2, k] /
                        np.linalg.norm(psi[:, counter2, k]) ** 2)

            # 4. Update coefficients and residual
            beta = np.dot(psi[:, counter, k].conj().T, R[:, k]) / np.linalg.norm(
                psi[:, counter, k]) ** 2
            betamatrix[counter, k] = beta
            R[:, k] = R[:, k] - beta * psi[:, counter, k]

    # 6. Deorthogonalize
    h = np.zeros((L, K), dtype=complex)
    for k in range(K):
        # QR factorization
        Q, Rqr = np.linalg.qr(columns[:, :, k], mode='reduced')
        h[:, k] = np.linalg.inv(Rqr) @ Q.conj().T @ psi[:, :, k] @ betamatrix[:, k]

    return indices, h
