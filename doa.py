import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as LA
import scipy.signal as ss


# Functions
def array_response_vector(array, theta):
    N = array.shape
    v = np.exp(1j * 2 * np.pi * array * np.sin(theta))
    return v / np.sqrt(N)


def music(CovMat, L, N, array, Angles):
    # CovMat is the signal covariance matrix, L is the number of sources, N is the number of n_elements
    # array holds the positions of antenna elements
    # Angles are the grid of directions in the azimuth angular domain
    _, V = LA.eig(CovMat)
    Qn = V[:, L:N]
    numAngles = Angles.size
    pspectrum = np.zeros(numAngles)
    for i in range(numAngles):
        av = array_response_vector(array, Angles[i])
        pspectrum[i] = 1 / LA.norm(av.conj().T @ Qn @ Qn.conj().T @ av)
        print(pspectrum[i])
    psindB = np.log10(10 * pspectrum / pspectrum.min())
    DoAsMUSIC, _ = ss.find_peaks(psindB, height=1.35, distance=1.5)
    return DoAsMUSIC, pspectrum


np.random.seed(100)

lamda = 1  # wavelength
kappa = np.pi / lamda  # wave number
L = 5  # number of sources
N = 32  # number of ULA elements
snr = 10  # signal to noise ratio

array = np.arange(N)
Thetas = np.pi * (np.random.rand(L) - 1 / 2)  # random source directions
print(Thetas / np.pi * 180)
Alphas = np.random.randn(L) + np.random.randn(L) * 1j  # random source powers
Alphas = np.sqrt(1 / 2) * Alphas
print(Thetas)
print(Alphas)

h = np.zeros(N)
for i in range(L):
    h = h + Alphas[i] * array_response_vector(array, Thetas[i])

Angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
numAngles = Angles.size

hv = np.zeros(numAngles)
for j in range(numAngles):
    av = array_response_vector(array, Angles[j])
    hv[j] = np.abs(np.inner(h, av.conj()))

powers = np.zeros(L)
for j in range(L):
    av = array_response_vector(array, Thetas[j])
    powers[j] = np.abs(np.inner(h, av.conj()))

numrealization = 100
H = np.zeros((N, numrealization)) + 1j * np.zeros((N, numrealization))

for iter in range(numrealization):
    htmp = np.zeros(N)
    for i in range(L):
        pha = np.exp(1j * 2 * np.pi * np.random.rand(1))
        htmp = htmp + pha * Alphas[i] * array_response_vector(array, Thetas[i])
    H[:, iter] = htmp + np.sqrt(0.5 / snr) * (np.random.randn(N) + np.random.randn(N) * 1j)
cov = H @ H.conj().transpose()
print(cov.min(),cov.max())
# MUSIC algorithm
DoAsMUSIC, psindB = music(cov, L, N, array, Angles)
print(DoAsMUSIC)
plt.plot(Angles, psindB)
plt.plot(Angles[DoAsMUSIC], psindB[DoAsMUSIC], 'x')
plt.title('MUSIC')
plt.legend(['pseudo spectrum', 'Estimated DoAs'])
plt.show()
