# Generate scatter points
from collections import namedtuple
from typing import List

import numpy as np

from python_code import conf
from python_code.utils.basis_functions import array_response_vector
from python_code.utils.constants import C

Channel = namedtuple("Channel", ["SP", "y", "AOA", "TOA"])


def create_scatter_points(L):
    # SP = np.random.rand(L - 1, 2) * 20 - 10  # random points uniformly placed in a 20 m x 20 m area
    SP = np.array([[8, 4], [4, 1]])
    return SP


def compute_gt_channel_parameters(posRx, SP):
    # Compute Channel Parameters for L paths
    TOA = [0 for _ in range(conf.L)]
    AOA = [0 for _ in range(conf.L)]
    TOA[0] = np.linalg.norm(posRx) / C
    AOA[0] = np.arctan2(posRx[1], posRx[0])

    for l in range(1, conf.L):
        AOA[l] = np.arctan2(SP[l - 1, 1], SP[l - 1, 0])
        TOA[l] = (np.linalg.norm(SP[l - 1, :]) + np.linalg.norm(conf.posRx - SP[l - 1, :])) / C
    return TOA, AOA


def compute_observations(TOA: List[float], AOA: List[float]):
    h = np.sqrt(1 / 2) * (np.random.randn(conf.L) + np.random.randn(conf.L) * 1j)  # random channel gains
    # Generate the observation and beamformers
    y = np.zeros((conf.Nr, conf.K, conf.Ns), dtype=complex)
    for ns in range(conf.Ns):
        # Generate channel
        H = np.zeros((conf.Nr, conf.K), dtype=complex)
        for l in range(conf.L):
            F = np.exp(1j * np.random.rand(1) * 2 * np.pi)  # random beamformer
            steering_vector = array_response_vector(conf.Nr, np.arange(conf.Nr) * np.sin(AOA[l])).reshape(-1, 1)
            delays_phase_vector = array_response_vector(conf.K,
                                                        2 * (conf.fc + np.arange(conf.K) * conf.BW / conf.K) * TOA[
                                                            l]).reshape(-1, 1)
            H += F * h[l] * delays_phase_vector.T * steering_vector
        y[:, :, ns] = H + conf.sigma / np.sqrt(2) * (
                np.random.randn(conf.Nr, conf.K) + 1j * np.random.randn(conf.Nr, conf.K))
    return y


def get_channel():
    SP = create_scatter_points(conf.L)
    TOA, AOA = compute_gt_channel_parameters(np.array(conf.posRx), SP)
    y = compute_observations(TOA, AOA)
    channel_instance = Channel(SP=SP,y=y,TOA=TOA,AOA=AOA)
    return channel_instance
