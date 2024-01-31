# Generate scatter points
from collections import namedtuple
from typing import List

import numpy as np

from python_code import conf
from python_code.utils.basis_functions import compute_time_options, compute_angle_options, create_wideband_aoa_mat
from python_code.utils.constants import C, ChannelBWType, P_0, DATA_COEF
from python_code.utils.path_loss import compute_path_loss

Channel = namedtuple("Channel", ["scatterers", "y", "AOA", "TOA"])


def compute_gt_channel_parameters(bs_loc: np.ndarray, ue_pos: np.ndarray, scatterers: np.ndarray):
    # Compute Channel Parameters for L paths
    TOA = [0 for _ in range(conf.L)]
    AOA = [0 for _ in range(conf.L)]
    TOA[0] = np.linalg.norm(ue_pos - bs_loc) / C
    AOA[0] = np.arctan2(ue_pos[1] - bs_loc[1], ue_pos[0] - bs_loc[0])
    for l in range(1, conf.L):
        AOA[l] = np.arctan2(scatterers[l - 1, 1] - bs_loc[1], scatterers[l - 1, 0] - bs_loc[0])
        TOA[l] = (np.linalg.norm(bs_loc - scatterers[l - 1]) + np.linalg.norm(conf.ue_pos - scatterers[l - 1])) / C
    # assert that toa are supported, must be smaller than largest distance divided by the speed
    assert all([TOA[l] < conf.K / conf.BW for l in range(len(TOA))])
    return TOA, AOA


def compute_observations(TOA: List[float], AOA: List[float]):
    alpha = P_0 * np.sqrt(1 / 2) * (np.random.randn(conf.L) + np.random.randn(conf.L) * 1j)
    Ns = conf.Nr_x * conf.K * DATA_COEF
    # Generate the observation and beamformers
    y = np.zeros((conf.Nr_x, conf.K, Ns), dtype=complex)
    for ns in range(Ns):
        # Generate channel
        h = np.zeros((conf.Nr_x, conf.K), dtype=complex)
        for l in range(conf.L):
            F = np.exp(1j * np.random.rand(1) * 2 * np.pi)  # random beamformer
            delays_phase_vector = compute_time_options(conf.fc, conf.K, conf.BW, np.array([TOA[l]]))
            if conf.channel_bandwidth == ChannelBWType.NARROWBAND.name:
                aoa_vector = compute_angle_options(np.sin(np.array([AOA[l]])), zoa=1, values=np.arange(conf.Nr_x)).T
                delay_aoa_matrix = np.matmul(aoa_vector, delays_phase_vector)
            elif conf.channel_bandwidth == ChannelBWType.WIDEBAND.name:
                raise ValueError("Wideband is currently no supported!!")
                wideband_aoa_mat = create_wideband_aoa_mat(np.array([AOA[l]]), conf.K, conf.BW, conf.fc, conf.Nr_x,
                                                           stack_axis=1)
                delay_aoa_matrix = wideband_aoa_mat * delays_phase_vector
            else:
                raise ValueError("No such type of channel BW!")
            channel_gain = alpha[l] / compute_path_loss(TOA[l])
            h += F * channel_gain * delay_aoa_matrix
        ## adding the white Gaussian noise
        noise = conf.sigma / np.sqrt(2) * (np.random.randn(conf.Nr_x, conf.K) + 1j * np.random.randn(conf.Nr_x, conf.K))
        y[:, :, ns] = h + noise
    return y


def get_2d_channel(bs_loc, ue_pos, scatterers):
    TOA, AOA = compute_gt_channel_parameters(bs_loc, ue_pos, scatterers)
    print(f"Distance to user {TOA[0] * C}[m], TOA[us]: {round(TOA[0], 3)}")
    y = compute_observations(TOA, AOA)
    channel_instance = Channel(scatterers=scatterers, y=y, TOA=TOA, AOA=AOA)
    return channel_instance
