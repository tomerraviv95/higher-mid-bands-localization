import numpy as np
import scipy.linalg
import scipy.signal
from matplotlib import pyplot as plt

np.random.seed(10)


# Function to get response for a given number of n_elements and spatial frequency
def getResponse(antennas, phi):
    return np.exp(-1j * np.pi * phi * np.arange(antennas)) / antennas ** 0.5


# System parameters
L = 1  # number of paths (including LOS)
Nt = 32  # number of TX n_elements
Nr = Nt  # number of RX n_elements
Nb = 360  # number of beams in dictionary
Ns = 100  # number of beams sent
c = 300  # speed of light meter / us
posRx = np.array([5, 3])  # RX (user) position, TX is assumed to be in [0, 0]
sigma = 0.1  # noise standard deviation
fc = 100  # carrier frequency in MHz

# Generate scatter points
SP = np.random.rand(L - 1, 2) * 20 - 10  # random points uniformly placed in a 20 m x 20 m area

# Compute Channel Parameters for L paths
TOA = np.zeros(L)
AOA = np.zeros(L)
TOA[0] = np.linalg.norm(posRx) / c
AOA[0] = np.arctan2(posRx[1], posRx[0])

for l in range(1, L):
    AOA[l] = np.arctan2(SP[l - 1, 1], SP[l - 1, 0])
    TOA[l] = (np.linalg.norm(SP[l - 1, :]) + np.linalg.norm(posRx - SP[l - 1, :])) / c

h = 1 * np.ones(L)  # some high channel gains
# Create dictionary
aa = np.linspace(-np.pi / 2, np.pi / 2, Nb)  # dictionary of spatial frequencies

# Generate channel
H = np.zeros(Nr, dtype=complex)
for l in range(L):
    H += h[l] * np.exp(-1j * 2 * np.pi * TOA[l] * fc) * getResponse(Nr, np.sin(AOA[l]))

# Generate the observation and beamformers
y = np.zeros((Nr, Ns), dtype=complex)
F = np.zeros(Ns, dtype=complex)
for k in range(Ns):
    F[k] = np.exp(1j * np.random.rand(1) * 2 * np.pi)  # random beamformer
    y[:, k] = np.dot(H, F[k])  + sigma / np.sqrt(2) * (np.random.randn(Nr) + 1j * np.random.randn(Nr))


def music(cov, L, N):
    # cov is the signal covariance matrix, L is the number of sources, N is the number of n_elements
    # array holds the positions of antenna elements
    # Angles are the grid of directions in the azimuth angular domain
    _, V = scipy.linalg.eig(cov)
    pspectrum = np.zeros(aa.size)
    Qn = V[:, L:N]
    for i in range(aa.size):
        av = getResponse(Nr, np.sin(aa[i]))
        pspectrum[i] = 1 / scipy.linalg.norm(Qn.conj().T @ av)
        print(pspectrum[i])
    psindB = np.log10(10 * pspectrum / pspectrum.min())
    DoAsMUSIC, _ = scipy.signal.find_peaks(psindB, height=1.35, distance=1.5)
    return DoAsMUSIC, psindB


cov = y @ y.conj().transpose()
print(cov.min(), cov.max())
print(AOA)
indices, spectrum = music(cov=cov, L=L, N=Nr)
print(indices)
plt.plot(aa, spectrum)
plt.plot(aa[indices], spectrum[indices], 'x')
plt.title('MUSIC')
plt.legend(['pseudo spectrum', 'Estimated DoAs'])
plt.show()
