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

    for counter in range(1, L + 1):
        # 2. Find maximum correlation between residual and columns of A
        cost = np.zeros(M)
        for m in range(M):
            for k in range(K):
                # Calculate cost for each column m of A and each channel k
                cost[m] = cost[m] + np.abs(np.dot(A[:, m, k].conj().T, R[:, k])) / np.linalg.norm(A[:, m, k],2)

        maxi = np.argmax(cost)
        print(maxi)
        indices[counter - 1] = maxi

        for k in range(1, K + 1):
            # 3. Orthogonalize
            columns[:, counter - 1, k - 1] = A[:, maxi, k - 1]
            omega = A[:, maxi, k - 1]
            psi[:, counter - 1, k - 1] = omega

            for counter2 in range(1, counter):
                # Gram-Schmidt orthogonalization
                psi[:, counter - 1, k - 1] = psi[:, counter - 1, k - 1] - (
                        np.dot(psi[:, counter2 - 1, k - 1].conj().T, omega) * psi[:, counter2 - 1, k - 1] /
                        np.linalg.norm(psi[:, counter2 - 1, k - 1]) ** 2)

            # 4. Update coefficients and residual
            beta = np.dot(psi[:, counter - 1, k - 1].conj().T, R[:, k - 1]) / np.linalg.norm(psi[:, counter - 1, k - 1]) ** 2
            betamatrix[counter - 1, k - 1] = beta
            R[:, k - 1] = R[:, k - 1] - beta * psi[:, counter - 1, k - 1]

    # 6. Deorthogonalize
    h = np.zeros((L, K), dtype=complex)
    for k in range(1, K + 1):
        # QR factorization
        Q, Rqr = np.linalg.qr(columns[:, :, k - 1], mode='reduced')
        h[:, k - 1] = np.linalg.inv(Rqr) @ Q.conj().T @ psi[:, :, k - 1] @ betamatrix[:, k - 1]

    return indices, h
