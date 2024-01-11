import numpy as np
from matplotlib import pyplot as plt

from music import music, array_response_vector, compute_time_options, compute_angle_options

np.random.seed(10)

# System parameters
L = 3  # number of paths (including LOS)
K = 10  # number of subcarriers
Nt = 32  # number of TX n_elements
Nr = Nt  # number of RX n_elements
Nb = 360  # number of beams in dictionary
T_res = 200  # number of times in dictionary
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
ANGLE = False
if ANGLE:
    angle_cov = np.cov(y.reshape(Nr, -1), bias=True)
    angle_values = np.arange(Nr)
    angle_options = compute_angle_options(aa, values=angle_values)
    indices, spectrum = music(cov=angle_cov, L=L, n_elements=Nr, options=angle_options)
    fig = plt.figure()
    plt.plot(aa, spectrum)
    plt.plot(aa[indices], spectrum[indices], 'x')
    plt.title('MUSIC for AOA Estimation')
    plt.xlabel('degree[rad]')
    plt.ylabel('MUSIC coefficient')
    plt.legend(['spectrum', 'Estimated AOAs'])
    plt.savefig('AOA.png', dpi=fig.dpi)
    plt.show()

# time delay
TIME_DELAY = False
if TIME_DELAY:
    fig = plt.figure()
    time_cov = np.cov(np.transpose(y, [1, 0, 2]).reshape(K, -1), bias=True)
    time_values = np.linspace(0, 0.1, T_res)
    time_options = compute_time_options(fc, K, BW, values=time_values)
    indices, spectrum = music(cov=time_cov, L=L, n_elements=K, options=time_options)
    plt.plot(time_values, spectrum)
    plt.plot(time_values[indices], spectrum[indices], 'x')
    plt.title('MUSIC for Delay Estimation')
    plt.xlabel('time[s]')
    plt.ylabel('MUSIC coefficient')
    plt.legend(['spectrum', 'Estimated delays'])
    plt.savefig('delay.png', dpi=fig.dpi)
    plt.show()

# combining both estimates, 2-D AOA & TOA
BOTH = True
if BOTH:
    fig = plt.figure()
    angle_time_cov = np.cov(y.reshape(K * Nr, -1), bias=True)
    angle_values = np.arange(Nr)
    angle_options = compute_angle_options(aa, values=angle_values)
    time_values = np.linspace(0, 0.1, T_res)
    time_options = compute_time_options(fc, K, BW, values=time_values)
    angle_time_options = np.kron(angle_options, time_options)
    indices, spectrum = music(cov=angle_time_cov, L=L, n_elements=Nr * K, options=angle_time_options)
    angle_indices = indices // T_res
    time_indices = indices % T_res
    degs_set, time_set = set(), set()
    filtered_peaks = []
    for angle_ind, time_ind in zip(angle_indices, time_indices):
        if angle_ind not in degs_set and time_ind not in time_set:
            filtered_peaks.append([aa[angle_ind], time_values[time_ind]])
            degs_set.add(angle_ind)
            time_set.add(time_ind)
    print(filtered_peaks)
    print(AOA)
    print(TOA)
