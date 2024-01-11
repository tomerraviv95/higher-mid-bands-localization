import numpy as np
from matplotlib import pyplot as plt

from music import music, array_response_vector

np.random.seed(10)

# System parameters
L = 5  # number of paths (including LOS)
K = 10  # number of subcarriers
Nt = 32  # number of TX n_elements
Nr = Nt  # number of RX n_elements
Nb = 360  # number of beams in dictionary
Ns = 1000  # number of beams sent
c = 300  # speed of light meter / us
posRx = np.array([5, 3])  # RX (user) position, TX is assumed to be in [0, 0]
sigma = 0.1  # noise standard deviation
fc = 1000  # carrier frequency in MHz
BW = 100  # BW frequency in MHz

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

h = np.sqrt(1 / 2) * (np.random.randn(L) + np.random.randn(L) * 1j)  # random channel gains
# Create dictionary
aa = np.linspace(-np.pi / 2, np.pi / 2, Nb)  # dictionary of spatial frequencies

# Generate the observation and beamformers
y = np.zeros((Nr, K, Ns), dtype=complex)
F = np.zeros(Ns, dtype=complex)
for ns in range(Ns):
    # Generate channel
    H = np.zeros((Nr, K), dtype=complex)
    for l in range(L):
        F = np.exp(1j * np.random.rand(1) * 2 * np.pi)  # random beamformer
        steering_vector = array_response_vector(Nr, np.arange(Nr) * np.sin(AOA[l])).reshape(-1, 1)
        delays_phase_vector = array_response_vector(K, 2 * (fc + np.arange(K) * BW / K) * TOA[l]).reshape(-1, 1)
        H += F * h[l] * delays_phase_vector.T * steering_vector
    y[:, :, ns] = H + sigma / np.sqrt(2) * (np.random.randn(Nr, K) + 1j * np.random.randn(Nr, K))

# angle of arrival
cov = np.cov(y.reshape(Nr, -1), bias=True)
angle_options = np.sin(aa)
indices, spectrum = music(cov=cov, L=L, n_elements=Nr, options=angle_options, basis_vector=np.arange(Nr))
plt.plot(aa, spectrum)
plt.plot(aa[indices], spectrum[indices], 'x')
plt.title('MUSIC for AOA estimation')
plt.xlabel('degree[rad]')
plt.ylabel('MUSIC coefficient')
plt.legend(['spectrum', 'Estimated AOAs'])
plt.show()

# time delay
cov2 = np.cov(np.transpose(y, [1, 0, 2]).reshape(K, -1), bias=True)
time_options = np.linspace(0, 0.1, 1000)
time_basis_vector = 2 * (fc + np.arange(K) * BW / K)
indices2, spectrum2 = music(cov=cov2, L=L, n_elements=K, options=time_options, basis_vector=time_basis_vector)
plt.plot(time_options, spectrum2)
plt.plot(time_options[indices2], spectrum2[indices2], 'x')
plt.title('MUSIC for delay estimation')
plt.xlabel('time[s]')
plt.ylabel('MUSIC coefficient')
plt.legend(['spectrum', 'Estimated delays'])
plt.show()
