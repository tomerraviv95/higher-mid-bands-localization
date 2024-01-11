import matplotlib.pyplot as plt
import numpy as np

from music import array_response_vector, music

np.random.seed(100)

# System parameters
L = 5  # number of paths (including LOS)
Nr = 32  # number of RX n_elements
pilots = 100  # number of pilots
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

Angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
numAngles = Angles.size

# Generate channel
h = 1 * np.ones(L)  # some high channel gains
H = np.zeros(Nr, dtype=complex)
for l in range(L):
    H += h[l] * np.exp(-1j * 2 * np.pi * TOA[l] * fc) * array_response_vector(Nr, np.sin(AOA[l]))

# Generate the observation and beamformers
y = np.zeros((Nr, pilots), dtype=complex)
F = np.zeros(pilots, dtype=complex)
for k in range(pilots):
    F[k] = np.exp(1j * np.random.rand(1) * 2 * np.pi)  # random beamformer
    y[:, k] = np.dot(H, F[k]) + sigma / np.sqrt(2) * (np.random.randn(Nr) + 1j * np.random.randn(Nr))

cov = y @ y.conj().transpose()
print(cov.min(), cov.max())
# MUSIC algorithm
DoAsMUSIC, psindB = music(cov, L, Nr, Angles)
print(DoAsMUSIC)
plt.plot(Angles, psindB)
plt.plot(Angles[DoAsMUSIC], psindB[DoAsMUSIC], 'x')
plt.title('MUSIC')
plt.legend(['pseudo spectrum', 'Estimated DoAs'])
plt.show()
