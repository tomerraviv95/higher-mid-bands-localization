import matplotlib.pyplot as plt
import numpy as np


# Function to get response for a given number of antennas and spatial frequency
def getResponse(antennas, phi):
    return np.exp(-1j * np.pi * phi * np.arange(antennas)) / antennas ** 0.5


# System parameters
L = 4  # number of paths (including LOS)
Rs = 100  # total BW in MHz
N = 10  # number of subcarriers
Nt = 32  # number of TX antennas
Nr = Nt  # number of RX antennas
Nb = Nt * 2  # number of beams in dictionary
Ns = 20  # number of beams sent
c = 300  # speed of light meter / us
Ts = 1 / Rs  # sampling period in us
posRx = np.array([5, 1])  # RX (user) position, TX is assumed to be in [0, 0]
alpha = 0.2  # user orientation
sigma = 0.1  # noise standard deviation

# Generate scatter points
SP = np.random.rand(L - 1, 2) * 20 - 10  # random points uniformly placed in a 20 m x 20 m area

# Compute Channel Parameters for L paths
TOA = np.zeros(L)
AOD = np.zeros(L)
AOA = np.zeros(L)
TOA[0] = np.linalg.norm(posRx) / c
AOD[0] = np.arctan2(posRx[1], posRx[0])
AOA[0] = np.arctan2(posRx[1], posRx[0]) - np.pi - alpha

for l in range(1, L):
    AOD[l] = np.arctan2(SP[l - 1, 1], SP[l - 1, 0])
    AOA[l] = np.arctan2(SP[l - 1, 1] - posRx[1], SP[l - 1, 0] - posRx[0]) - alpha
    TOA[l] = (np.linalg.norm(SP[l - 1, :]) + np.linalg.norm(posRx - SP[l - 1, :])) / c

h = 10 * np.ones(L)  # some high channel gains

# Create dictionary
Ut = np.zeros((Nt, Nb))
Ur = np.zeros((Nr, Nb))
aa = np.arange(-Nb / 2, Nb / 2)
aa = 2 * aa / Nb  # dictionary of spatial frequencies

for m in range(Nb):
    Ut[:, m] = getResponse(Nt, aa[m]) * np.sqrt(Nt)
    Ur[:, m] = getResponse(Nr, aa[m]) * np.sqrt(Nr)

# Generate channel
H = np.zeros((Nr, Nt, N), dtype=complex)
for n in range(N):
    for l in range(L):
        H[:, :, n] += h[l] * np.exp(-1j * 2 * np.pi * TOA[l] * n / (N * Ts)) * np.sqrt(Nr) * \
                      getResponse(Nr, np.sin(AOA[l])) * np.sqrt(Nt) * getResponse(Nt, np.sin(AOD[l])).conj().T

# Visualize the beamspace channel for 1 subcarrier in AOA/AOD space
Hb = np.dot(np.dot(Ur.conj().T, H[:, :, 0]), Ut)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(np.arcsin(aa), np.arcsin(aa), abs(Hb), 50, cmap='binary')
ax.set_xlabel('AOD')
ax.set_ylabel('AOA')
plt.show()

# Generate the observation and beamformers
y = np.zeros((Nr, Ns, N), dtype=complex)
F = np.zeros((Nt, Ns, N), dtype=complex)
for k in range(Ns):
    for n in range(N):
        F[:, k, n] = np.exp(1j * np.random.rand(Nt) * 2 * np.pi)  # random beamformers
        y[:, k, n] = np.dot(H[:, :, n], F[:, k, n]) + sigma / np.sqrt(2) * (
                np.random.randn(Nr) + 1j * np.random.randn(Nr))

# Vectorize and generation of the basis
ybb = np.zeros((Nr * Ns, N), dtype=complex)
Omega = np.zeros((Nr * Ns, Nb * Nb, N), dtype=complex)
for n in range(N):
    ybb[:, n] = y[:, :, n].reshape(Nr * Ns)
    Omega[:, :, n] = np.kron((Ut.conj().T @ F[:, :, n]).conj().T, Ur)
