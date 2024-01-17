# Generate scatter points
from collections import namedtuple
from typing import List

import numpy as np

from python_code import conf
from python_code.utils.basis_functions import compute_time_options, compute_angle_options
from python_code.utils.constants import C, ChannelBWType

Channel = namedtuple("Channel", ["scatterers", "y", "AOA", "TOA"])


def create_scatter_points(L):
    # x must be positive. The array lies on the y-axis and points towards the x-axis.
    # scatterers = np.random.rand(L - 1, 2) * 20 - 10  # random points uniformly placed in a 20 m x 20 m area
    scatterers = np.array([[10, 10], [15, -10]])
    return scatterers


def compute_gt_channel_parameters(ue_pos: List[float], scatterers: np.ndarray):
    # Compute Channel Parameters for L paths
    TOA = [0 for _ in range(conf.L)]
    AOA = [0 for _ in range(conf.L)]
    TOA[0] = np.linalg.norm(ue_pos) / C
    AOA[0] = np.arctan2(ue_pos[1], ue_pos[0])
    for l in range(1, conf.L):
        AOA[l] = np.arctan2(scatterers[l - 1, 1], scatterers[l - 1, 0])
        TOA[l] = (np.linalg.norm(scatterers[l - 1, :]) + np.linalg.norm(conf.ue_pos - scatterers[l - 1, :])) / C
    conf.max_time = max(TOA) * 1.2
    return TOA, AOA


def compute_observations(TOA: List[float], AOA: List[float]):
    alpha = np.sqrt(1 / 2) * (np.random.randn(conf.L) + np.random.randn(conf.L) * 1j)
    # Generate the observation and beamformers
    y = np.zeros((conf.Nr, conf.K, conf.Ns), dtype=complex)
    for ns in range(conf.Ns):
        # Generate channel
        h = np.zeros((conf.Nr, conf.K), dtype=complex)
        F = np.exp(1j * np.random.rand(1) * 2 * np.pi)  # random beamformer
        for l in range(conf.L):
            delays_phase_vector = compute_time_options(conf.fc, conf.K, conf.BW, np.array([TOA[l]]))
            if conf.channel_bandwidth == ChannelBWType.NARROWBAND.name:
                aoa_vector = compute_angle_options(np.array([AOA[l]]), np.arange(conf.Nr)).T
                delay_aoa_matrix = np.matmul(aoa_vector, delays_phase_vector)
            elif conf.channel_bandwidth == ChannelBWType.WIDEBAND.name:
                wideband_aoa_mat = []
                for k in range(conf.K):
                    wideband_angle = np.array([AOA[l]]) * (1 + k * conf.BW / conf.fc)
                    wideband_aoa_vec = compute_angle_options(wideband_angle, np.arange(conf.Nr)).T
                    wideband_aoa_mat.append(wideband_aoa_vec)
                wideband_aoa_mat = np.concatenate(wideband_aoa_mat, axis=1)
                delay_aoa_matrix = wideband_aoa_mat * delays_phase_vector
            else:
                raise ValueError("No such type of channel BW!")
            h += F * alpha[l] * delay_aoa_matrix

    ## adding the white Gaussian noise
    noise = conf.sigma / np.sqrt(2) * (np.random.randn(conf.Nr, conf.K) + 1j * np.random.randn(conf.Nr, conf.K))
    y[:, :, ns] = h + noise
    return y

def get_channel():
    scatterers = create_scatter_points(conf.L)
    TOA, AOA = compute_gt_channel_parameters(np.array(conf.ue_pos), scatterers)
    y = compute_observations(TOA, AOA)
    channel_instance = Channel(scatterers=scatterers, y=y, TOA=TOA, AOA=AOA)
    return channel_instance
