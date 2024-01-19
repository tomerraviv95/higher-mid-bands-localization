import math
from collections import namedtuple

import numpy as np
from matplotlib import pyplot as plt

from python_code import conf
from python_code.estimation.beam_sweeper import BeamSweeper
from python_code.estimation.music import MUSIC
from python_code.utils.basis_functions import compute_angle_options, compute_time_options, create_wideband_aoa_mat
from python_code.utils.constants import AlgType

algs = {AlgType.BEAMSWEEPER: BeamSweeper(2), AlgType.MUSIC: MUSIC(1.4)}
ALG_TYPE = AlgType.MUSIC

Estimation = namedtuple("Estimation", ["AOA", "TOA", "ZOA"], defaults=(None,) * 3)
NEIGHBORS_THRESH = 10


class AngleEstimator:
    def __init__(self):
        self.aoa_angles_dict = np.linspace(-np.pi / 2, np.pi / 2, conf.Nb)  # dictionary of spatial frequencies
        self.Na = 90
        self.zoa_angles_dict = np.linspace(0, np.pi / 2, self.Na)
        self._angle_options = np.zeros((conf.Nr, conf.Nr, self.Na * conf.Nb), dtype=complex)
        for i in range(conf.Nb):
            aoa = self.aoa_angles_dict[i]
            for j in range(self.Na):
                zoa = self.zoa_angles_dict[j]
                aoa_vector_y = compute_angle_options(np.array([aoa]).reshape(-1, 1),
                                                     np.sin(np.array([zoa])), np.arange(conf.Nr)).T
                aoa_vector_x = compute_angle_options(np.array([aoa]).reshape(-1, 1),
                                                     np.cos(np.array([zoa])), np.arange(conf.Nr)).T
                self._angle_options[:, :, i * self.Na + j] = aoa_vector_y @ aoa_vector_x.T
        self._angle_options = self._angle_options.reshape(32 * 32, -1).T
        self.algorithm = algs[ALG_TYPE]

    def filter_neighbor_peaks(self, aoa_indices, zoa_indices):
        aoa_zoa_set = set()
        for aoa, zoa in zip(aoa_indices, zoa_indices):
            is_added = True
            # iterate through each aoa,zoa already in the set. if one existing pair that is close to the current exists
            # then don't add it
            for n_aoa, n_zoa in aoa_zoa_set:
                if abs(n_aoa - aoa) + abs(n_zoa - zoa) < NEIGHBORS_THRESH:
                    is_added = False
            if is_added:
                aoa_zoa_set.add((aoa, zoa))
        aoa_list, zoa_list = zip(*aoa_zoa_set)
        return list(aoa_list), list(zoa_list)

    def estimate(self, y):
        indices, self._spectrum = self.algorithm.run(y=y, basis_vectors=self._angle_options,
                                                     n_elements=conf.Nr * conf.Nr)
        print(self._spectrum, indices)
        print(self._spectrum[indices])
        aoa_indices = indices // (self.Na)
        zoa_indices = indices % (self.Na)
        self._aoa_indices, self._zoa_indices = self.filter_neighbor_peaks(aoa_indices, zoa_indices)
        estimation = Estimation(AOA=self.aoa_angles_dict[self._aoa_indices],
                                ZOA=self.zoa_angles_dict[self._zoa_indices])
        fig = plt.figure()
        plt.contourf(self.aoa_angles_dict, self.zoa_angles_dict,
                     self._spectrum.reshape(self.Na, conf.Nb, order='F'), cmap='magma')
        ax = plt.gca()
        ax.set_xlabel('AOA[us]')
        ax.set_ylabel('ZOA[rad]')
        plt.plot(estimation.AOA, estimation.ZOA, 'ro')
        plt.savefig('AOA_and_ZOA.png', dpi=fig.dpi)
        plt.show()
        return estimation


class WidebandAngleEstimator:
    def __init__(self):
        self.angles_dict = np.linspace(-np.pi / 2, np.pi / 2, conf.Nb)  # dictionary of spatial frequencies
        self._angle_options = create_wideband_aoa_mat(self.angles_dict, conf.K, conf.BW, conf.fc, conf.Nr,
                                                      stack_axis=1).reshape(conf.Nr * conf.K, -1).T
        self.algorithm = algs[ALG_TYPE]

    def estimate(self, y):
        self._indices, self._spectrum = self.algorithm.run(y=y, basis_vectors=self._angle_options,
                                                           n_elements=conf.Nr * conf.K)
        estimator = Estimation(AOA=self.angles_dict[self._indices])
        return estimator


class TimeEstimator:
    def __init__(self):
        self.times_dict = np.linspace(0, conf.max_time, conf.T_res)
        self._time_options = compute_time_options(conf.fc, conf.K, conf.BW, values=self.times_dict)
        self.algorithm = algs[ALG_TYPE]

    def estimate(self, y):
        self._indices, self._spectrum = self.algorithm.run(y=np.transpose(y, [1, 0, 2]), n_elements=conf.K,
                                                           basis_vectors=self._time_options)
        estimator = Estimation(TOA=self.times_dict[self._indices])
        return estimator


class AngleTimeEstimator:
    def __init__(self):
        self.angle_estimator = AngleEstimator()
        self.time_estimator = TimeEstimator()
        self.angle_time_options = np.kron(self.angle_estimator._angle_options, self.time_estimator._time_options)
        self.algorithm = algs[ALG_TYPE]

    def estimate(self, y):
        indices, self._spectrum = self.algorithm.run(y=y, n_elements=conf.Nr * conf.K,
                                                     basis_vectors=self.angle_time_options)
        # filter nearby detected peaks
        aoa_toa_set = self.filter_peaks(indices)
        aoa_list, toa_list = zip(*aoa_toa_set)
        estimator = Estimation(AOA=aoa_list, TOA=toa_list)
        return estimator

    def filter_peaks(self, indices):
        aoa_indices = indices // conf.T_res
        toa_indices = indices % conf.T_res
        aoa_toa_set = set()
        for unique_toa_ind in np.unique(toa_indices):
            toa = self.time_estimator.times_dict[unique_toa_ind]
            avg_aoa_ind = int(np.mean(aoa_indices[toa_indices == unique_toa_ind]))
            aoa = self.angle_estimator.angles_dict[avg_aoa_ind]
            if not self.set_contains(aoa_toa_set, (aoa, toa)):
                aoa_toa_set.add((aoa, toa))
        return aoa_toa_set

    @staticmethod
    def set_contains(aoa_toa_set, aoa_toa_tuple):
        for c_aoa, c_toa in aoa_toa_set:
            if abs(aoa_toa_tuple[0] - c_aoa) < 5 * math.pi / 180 and abs(aoa_toa_tuple[1] - c_toa) < 0.02:
                return True
        return False
