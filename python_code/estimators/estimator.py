import numpy as np

from python_code import conf
from python_code.estimators.beam_sweeper import BeamSweeper
from python_code.estimators.music import MUSIC
from python_code.utils.basis_functions import compute_angle_options, compute_time_options, create_wideband_aoa_mat
from python_code.utils.constants import AlgType

algs = {AlgType.BEAMSWEEPER: BeamSweeper(2), AlgType.MUSIC: MUSIC(1.4)}
ALG_TYPE = AlgType.MUSIC

class AngleEstimator:
    def __init__(self):
        self.angles_dict = np.linspace(-np.pi / 2, np.pi / 2, conf.Nb)  # dictionary of spatial frequencies
        self._angle_options = compute_angle_options(self.angles_dict, values=np.arange(conf.Nr))
        self.algorithm = algs[ALG_TYPE]

    def estimate(self, y):
        self._indices, self._spectrum = self.algorithm.run(y=y, basis_vectors=self._angle_options, n_elements=conf.Nr)
        return self.angles_dict[self._indices]


class WidebandAngleEstimator:
    def __init__(self):
        self.angles_dict = np.linspace(-np.pi / 2, np.pi / 2, conf.Nb)  # dictionary of spatial frequencies
        self._angle_options = create_wideband_aoa_mat(self.angles_dict, conf.K, conf.BW, conf.fc, conf.Nr,
                                                      stack_axis=1).reshape(conf.Nr * conf.K, -1).T
        self.algorithm = algs[ALG_TYPE]

    def estimate(self, y):
        self._indices, self._spectrum = self.algorithm.run(y=y, basis_vectors=self._angle_options,
                                                       n_elements=conf.Nr * conf.K)
        return self.angles_dict[self._indices]


class TimeEstimator:
    def __init__(self):
        self.times_dict = np.linspace(0, conf.max_time, conf.T_res)
        self._time_options = compute_time_options(conf.fc, conf.K, conf.BW, values=self.times_dict)
        self.algorithm = algs[ALG_TYPE]

    def estimate(self, y):
        self._indices, self._spectrum = self.algorithm.run(y=np.transpose(y, [1, 0, 2]), n_elements=conf.K,
                                                       basis_vectors=self._time_options)
        return self.times_dict[self._indices]


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
        filtered_peaks = self.filter_nearby_peaks(indices)
        return filtered_peaks

    def filter_nearby_peaks(self, indices):
        angle_indices = indices // conf.T_res
        time_indices = indices % conf.T_res
        filtered_peaks = []
        for unique_time_ind in np.unique(time_indices):
            unique_time = self.time_estimator.times_dict[unique_time_ind]
            avg_angle_ind = int(np.mean(angle_indices[time_indices == unique_time_ind]))
            ang_angle = self.angle_estimator.angles_dict[avg_angle_ind]
            filtered_peaks.append([ang_angle, unique_time])
        filtered_peaks = np.array(filtered_peaks)
        return filtered_peaks