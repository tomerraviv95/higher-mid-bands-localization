# Generate scatter points
from collections import namedtuple
from typing import List

import numpy as np

from python_code import conf
from python_code.utils.basis_functions import compute_angle_options, create_wideband_aoa_mat, compute_time_options
from python_code.utils.constants import C, ChannelBWType, P_0

Channel = namedtuple("Channel", ["scatterers", "y", "AOA", "TOA", "ZOA"])


def compute_gt_channel_parameters(bs_loc: np.ndarray, ue_pos: np.ndarray, scatterers: np.ndarray):
    # Compute Channel Parameters for L paths
    TOA = [0 for _ in range(conf.L)]
    AOA = [0 for _ in range(conf.L)]
    ZOA = [0 for _ in range(conf.L)]
    TOA[0] = np.linalg.norm(ue_pos - bs_loc) / C
    AOA[0] = np.arctan2(ue_pos[1] - bs_loc[1], ue_pos[0] - bs_loc[0])
    ZOA[0] = np.arctan2(np.linalg.norm(bs_loc[:2] - ue_pos[:2]), bs_loc[2] - ue_pos[2])
    for l in range(1, conf.L):
        AOA[l] = np.arctan2(scatterers[l - 1, 1] - bs_loc[1], scatterers[l - 1, 0] - bs_loc[0])
        ZOA[l] = np.arctan2(np.linalg.norm(scatterers[l - 1, :2] - bs_loc[:2]), bs_loc[2] - scatterers[l - 1, 2])
        TOA[l] = (np.linalg.norm(bs_loc - scatterers[l - 1]) + np.linalg.norm(conf.ue_pos - scatterers[l - 1])) / C
    return TOA, AOA, ZOA


def compute_observations(TOA: List[float], AOA: List[float], ZOA: List[float]):
    alpha = P_0 * np.sqrt(1 / 2) * (np.random.randn(conf.L) + np.random.randn(conf.L) * 1j)
    # Generate the observation and beamformers
    y = np.zeros((conf.Nr_y, conf.Nr_x, conf.K, conf.Ns), dtype=complex)
    for ns in range(conf.Ns):
        # Generate channel
        h = np.zeros((conf.Nr_y, conf.Nr_x, conf.K), dtype=complex)
        for l in range(conf.L):
            F = np.exp(1j * np.random.rand(1) * 2 * np.pi)  # random beamformer
            delays_phase_vector = compute_time_options(conf.fc, conf.K, conf.BW, np.array([TOA[l]]))
            if conf.channel_bandwidth == ChannelBWType.NARROWBAND.name:
                aoa_vector_y = compute_angle_options(np.sin(np.array([AOA[l]])).reshape(-1, 1),
                                                     np.sin(np.array([ZOA[l]])), np.arange(conf.Nr_y)).T
                aoa_vector_x = compute_angle_options(np.sin(np.array([AOA[l]])).reshape(-1, 1),
                                                     np.cos(np.array([ZOA[l]])), np.arange(conf.Nr_x)).T
                aoa_matrix = np.expand_dims(aoa_vector_y @ aoa_vector_x.T, axis=-1)
                delay_aoa_matrix = aoa_matrix @ delays_phase_vector
            elif conf.channel_bandwidth == ChannelBWType.WIDEBAND.name:
                raise ValueError("Wideband is currently no supported!!")
                wideband_aoa_mat = create_wideband_aoa_mat(np.array([AOA[l]]), conf.K, conf.BW, conf.fc, conf.Nr,
                                                           stack_axis=1)
                delay_aoa_matrix = wideband_aoa_mat * delays_phase_vector
            else:
                raise ValueError("No such type of channel BW!")
            channel_gain = alpha[l] / compute_path_loss(TOA[l])
            h += F * channel_gain * delay_aoa_matrix
        ## adding the white Gaussian noise
        noise_real = np.random.randn(conf.Nr_y, conf.Nr_x, conf.K)
        noise_img = 1j * np.random.randn(conf.Nr_y, conf.Nr_x, conf.K)
        noise = conf.sigma / np.sqrt(2) * (noise_real + noise_img)
        y[:, :, :, ns] = h + noise
    return y


def get_3d_channel(bs_loc, ue_pos, scatterers):
    TOA, AOA, ZOA = compute_gt_channel_parameters(bs_loc, ue_pos, scatterers)
    y = compute_observations(TOA, AOA, ZOA)
    channel_instance = Channel(scatterers=scatterers, y=y, TOA=TOA, AOA=AOA, ZOA=ZOA)
    return channel_instance
