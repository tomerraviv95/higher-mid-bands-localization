import numpy as np
import scipy.signal
from scipy.ndimage.measurements import label

from python_code import conf


class CaponBeamforming:
    def __init__(self, thresh: float):
        self.thresh = thresh

    def run(self, y: np.ndarray, basis_vectors: np.ndarray, n_elements: int, one_dimensional=True):
        cov = np.cov(y.reshape(n_elements, -1), bias=True)
        inv_cov = np.linalg.inv(cov)
        inv_cov = inv_cov / np.linalg.norm(inv_cov)
        norm_values = np.linalg.norm((basis_vectors.conj() @ inv_cov) * basis_vectors, axis=1)
        norm_values = 1 / norm_values
        if one_dimensional:
            indices, _ = scipy.signal.find_peaks(norm_values, height=self.thresh)
            return indices, norm_values, len(indices)
        norm_values = norm_values.reshape(-1, int(conf.K / (conf.BW * conf.T_res)))
        labeled, ncomponents = label(norm_values > self.thresh,
                                     structure=np.ones((3, 3), dtype=np.int))  # this defines the connection filter)
        indices = []
        for component in range(1, ncomponents + 1):
            component_indices = np.array(np.where(labeled == component)).T
            max, ind = 0, None
            for component_indx in component_indices:
                if norm_values[component_indx[0]][component_indx[1]] > max:
                    max = norm_values[component_indx[0]][component_indx[1]]
                    ind = component_indx
            indices.append(ind)
        return np.array(indices), norm_values, len(indices)
