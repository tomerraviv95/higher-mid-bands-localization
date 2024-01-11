import matplotlib.pyplot as plt
import numpy as np

from music import array_response_vector, music

np.random.seed(100)

lamda = 1  # wavelength
kappa = np.pi / lamda  # wave number
L = 5  # number of sources
N = 32  # number of ULA elements
snr = 10  # signal to noise ratio

array = np.arange(N)
Thetas = np.pi * (np.random.rand(L) - 1 / 2)  # random source directions
# Alphas = np.random.randn(L) + np.random.randn(L) * 1j  # random source powers
Alphas = 1 * np.ones(L) # np.sqrt(1 / 2) * Alphas
angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
numAngles = angles.size

numrealization = 1000
H = np.zeros((N, numrealization)) + 1j * np.zeros((N, numrealization))
for iter in range(numrealization):
    htmp = np.zeros(N)
    for i in range(L):
        pha = np.exp(1j * 2 * np.pi * np.random.rand(1))
        htmp = htmp + pha * Alphas[i] * array_response_vector(N, np.sin(Thetas[i]))
    H[:, iter] = htmp + np.sqrt(0.5 / snr) * (np.random.randn(N) + np.random.randn(N) * 1j)
cov = H @ H.conj().transpose()
print(cov.min(), cov.max())
# MUSIC algorithm
DoAsMUSIC, psindB = music(cov=cov, L=L, n_elements=N, variables=np.sin(angles))
print(DoAsMUSIC)
plt.plot(angles, psindB)
plt.plot(angles[DoAsMUSIC], psindB[DoAsMUSIC], 'x')
plt.title('MUSIC')
plt.legend(['pseudo spectrum', 'Estimated DoAs'])
plt.show()
