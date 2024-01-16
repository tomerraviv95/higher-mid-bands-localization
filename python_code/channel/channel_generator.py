# Generate scatter points
from collections import namedtuple
from typing import List

import numpy as np

from python_code import conf
from python_code.utils.basis_functions import compute_time_options, compute_angle_options
from python_code.utils.constants import C, ChannelBWType

Channel = namedtuple("Channel", ["SP", "y", "AOA", "TOA"])


def create_scatter_points(L):
    # x must be positive. The array lies on the y-axis and points towards the x-axis.
    # SP = np.random.rand(L - 1, 2) * 20 - 10  # random points uniformly placed in a 20 m x 20 m area
    SP = np.array([[10, 10], [15, -12]])
    return SP


def compute_gt_channel_parameters(ue_pos: List[float], SP: np.ndarray):
    # Compute Channel Parameters for L paths
    TOA = [0 for _ in range(conf.L)]
    AOA = [0 for _ in range(conf.L)]
    TOA[0] = np.linalg.norm(ue_pos) / C
    AOA[0] = np.arctan2(ue_pos[1], ue_pos[0])
    for l in range(1, conf.L):
        AOA[l] = np.arctan2(SP[l - 1, 1], SP[l - 1, 0])
        TOA[l] = (np.linalg.norm(SP[l - 1, :]) + np.linalg.norm(conf.ue_pos - SP[l - 1, :])) / C
    conf.max_time = max(TOA) * 1.2
    return TOA, AOA


def compute_observations(TOA: List[float], AOA: List[float]):
    alpha = np.sqrt(1 / 2) * (np.random.randn(conf.L) + np.random.randn(conf.L) * 1j)
    # Generate the observation and beamformers
    y = np.zeros((conf.Nr, conf.K, conf.Ns), dtype=complex)
    for ns in range(conf.Ns):
        # Generate channel
        h = np.zeros((conf.Nr, conf.K), dtype=complex)
        for l in range(conf.L):
            F = np.exp(1j * np.random.rand(1) * 2 * np.pi)  # random beamformer
            aoa_vector = compute_angle_options(np.array([AOA[l]]), np.arange(conf.Nr)).T
            delays_phase_vector = compute_time_options(conf.fc, conf.K, conf.BW, np.array([TOA[l]]))
            if conf.channel_bandwidth == ChannelBWType.NARROWBAND.name:
                h += F * alpha[l] * np.matmul(aoa_vector, delays_phase_vector)
            elif conf.channel_bandwidth == ChannelBWType.WIDEBAND.name:
                frequency_wideband_vector = compute_time_options(conf.fc, conf.K, conf.BW, np.array([1 / conf.fc]))
                wideband_matrix = np.matmul(aoa_vector, frequency_wideband_vector)
                h += F * alpha[l] * wideband_matrix * delays_phase_vector
            else:
                raise ValueError("No such type of channel BW!")
        ## adding the white Gaussian noise
        noise = conf.sigma / np.sqrt(2) * (np.random.randn(conf.Nr, conf.K) + 1j * np.random.randn(conf.Nr, conf.K))
        y[:, :, ns] = h + noise
    return y


def get_channel():
    SP = create_scatter_points(conf.L)
    TOA, AOA = compute_gt_channel_parameters(np.array(conf.ue_pos), SP)
    y = compute_observations(TOA, AOA)
    channel_instance = Channel(SP=SP, y=y, TOA=TOA, AOA=AOA)
    return channel_instance
