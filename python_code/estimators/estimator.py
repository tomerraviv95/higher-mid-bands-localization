import numpy as np

from python_code import conf
from python_code.estimators.music import music
from python_code.utils.basis_functions import compute_angle_options, compute_time_options, create_wideband_aoa_mat


class AngleEstimator:
    def __init__(self):
        self.angles_dict = np.linspace(-np.pi / 2, np.pi / 2, conf.Nb)  # dictionary of spatial frequencies
        self._angle_options = compute_angle_options(self.angles_dict, values=np.arange(conf.Nr))
        self.algorithm = music

    def estimate(self, y):
        cov = np.cov(y.reshape(conf.Nr, -1), bias=True)
        self._indices, self._spectrum = self.algorithm(cov=cov, n_elements=conf.Nr, options=self._angle_options)
        return self.angles_dict[self._indices]


class WidebandAngleEstimator(AngleEstimator):
    def __init__(self):
        super(WidebandAngleEstimator, self).__init__()
        self._angle_options = create_wideband_aoa_mat(self.angles_dict, conf.K, conf.BW, conf.fc, conf.Nr,
                                                      stack_axis=1).T

    def estimate(self, y):
        for k in range(conf.K):
            cov = np.cov(y[:, k], bias=True)
            cur_options = self._angle_options[k * len(self.angles_dict):(k + 1) * len(self.angles_dict)]
            indices, spectrum = self.algorithm(cov=cov, n_elements=conf.Nr, options=cur_options)
            print(indices)
        exit()
        return self.angles_dict[self._indices]


class TimeEstimator:
    def __init__(self):
        self.times_dict = np.linspace(0, conf.max_time, conf.T_res)
        self._time_options = compute_time_options(conf.fc, conf.K, conf.BW, values=self.times_dict)
        self.algorithm = music

    def estimate(self, y):
        cov = np.cov(np.transpose(y, [1, 0, 2]).reshape(conf.K, -1), bias=True)
        self._indices, self._spectrum = self.algorithm(cov=cov, n_elements=conf.K, options=self._time_options)
        return self.times_dict[self._indices]


class AngleTimeEstimator:
    def __init__(self):
        self.angle_estimator = AngleEstimator()
        self.time_estimator = TimeEstimator()
        self.angle_time_options = np.kron(self.angle_estimator._angle_options, self.time_estimator._time_options)
        self.algorithm = music

    def estimate(self, y):
        angle_time_cov = np.cov(y.reshape(conf.K * conf.Nr, -1), bias=True)
        indices, self._spectrum = self.algorithm(cov=angle_time_cov, n_elements=conf.Nr * conf.K,
                                                 options=self.angle_time_options)
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


class WidebandAngleTimeEstimator(AngleTimeEstimator):
    def __init__(self):
        super(WidebandAngleTimeEstimator, self).__init__()
        self.modify_angle_options_by_wideband()

    def modify_angle_options_by_wideband(self):
        angles_dict = self.angle_estimator.angles_dict
        angle_options = create_wideband_aoa_mat(angles_dict, conf.K, conf.BW, conf.fc, conf.Nr, stack_axis=0).T
        time_options = np.tile(self.time_estimator._time_options, [1, conf.Nr])
        self.angle_time_options = np.einsum('nk,lk->nlk', angle_options, time_options).reshape(-1,
                                                                                               angle_options.shape[1])
