import numpy as np

from python_code import conf
from python_code.estimation import Estimation
from python_code.estimation.algs import ALG_TYPE, ALGS_DICT
from python_code.estimation.angle import AngleEstimator2D, AngleEstimator3D
from python_code.estimation.time import TimeEstimator2D, TimeEstimator3D
from python_code.utils.peaks_filtering import merge, filter_peaks

PROXIMITY_THRESH = 15


class AngleTimeEstimator2D:
    def __init__(self):
        self.angle_estimator = AngleEstimator2D()
        self.time_estimator = TimeEstimator2D()
        self.angle_time_options = np.kron(self.angle_estimator._angle_options, self.time_estimator._time_options)
        self.algorithm = ALGS_DICT[ALG_TYPE]

    def estimate(self, y):
        indices, self._spectrum, L_hat = self.algorithm.run(y=y, n_elements=conf.Nr_x * conf.K,
                                                            basis_vectors=self.angle_time_options)
        aoa_indices = indices // conf.T_res
        toa_indices = indices % conf.T_res
        merged = np.array(merge(aoa_indices, toa_indices))
        peaks = filter_peaks(merged, L_hat)
        self._aoa_indices = peaks[:, 0]
        self._toa_indices = peaks[:, 1]
        estimator = Estimation(AOA=self.angle_estimator.angles_dict[self._aoa_indices],
                               TOA=self.time_estimator.times_dict[self._toa_indices])
        return estimator


class AngleTimeEstimator3D:
    def __init__(self):
        self.angle_estimator = AngleEstimator3D()
        self.time_estimator = TimeEstimator3D()
        self.angle_time_options = np.kron(self.angle_estimator._angle_options, self.time_estimator._time_options)
        self.algorithm = ALGS_DICT[ALG_TYPE]

    def estimate(self, y):
        indices, self._spectrum,L_hat = self.algorithm.run(y=y, n_elements=conf.Nr_x * conf.Nr_y * conf.K,
                                                     basis_vectors=self.angle_time_options, do_one_calc=False)
        print(indices)
        # filter nearby detected peaks
        aoa_zoa_toa_set = self.filter_peaks(indices)
        aoa_list, zoa_list, toa_list = zip(*aoa_zoa_toa_set)
        if type(aoa_list) is not list:
            aoa_list, zoa_list, toa_list = [aoa_list], [zoa_list], [toa_list]
        estimator = Estimation(AOA=[self.angle_estimator.aoa_angles_dict[aoa_ind] for aoa_ind in aoa_list],
                               ZOA=[self.angle_estimator.zoa_angles_dict[zoa_ind] for zoa_ind in zoa_list],
                               TOA=[self.time_estimator.times_dict[toa_ind] for toa_ind in toa_list])
        return estimator

    def filter_peaks(self, indices):
        angle_indices = indices // conf.T_res
        aoa_indices = angle_indices // (conf.zoa_res)
        zoa_indices = angle_indices % (conf.zoa_res)
        toa_indices = indices % conf.T_res
        aoa_toa_zoa_set = set()
        for aoa_ind, zoa_ind, toa_ind in zip(aoa_indices, zoa_indices, toa_indices):
            to_add = True
            for aoa_ind2, zoa_ind2, toa_ind2 in aoa_toa_zoa_set:
                if sum([abs(aoa_ind2 - aoa_ind), abs(zoa_ind2 - zoa_ind), abs(toa_ind2 - toa_ind)]) < PROXIMITY_THRESH:
                    to_add = False
            if to_add:
                aoa_toa_zoa_set.add((aoa_ind, zoa_ind, toa_ind))
        return aoa_toa_zoa_set
