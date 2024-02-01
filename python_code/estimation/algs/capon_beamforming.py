import numpy as np
import scipy.signal
from scipy.ndimage.measurements import label


class CaponBeamforming:
    def __init__(self, thresh: float):
        self.thresh = thresh

    def run(self, y: np.ndarray, basis_vectors: np.ndarray, n_elements: int, second_dim: int = None,
            third_dim: int = None, batches=1):
        cov = np.cov(y.reshape(n_elements, -1), bias=True)
        cov = np.linalg.inv(cov)
        cov = cov / np.linalg.norm(cov)
        if batches == 1:
            norm_values = np.linalg.norm((basis_vectors.conj() @ cov) * basis_vectors, axis=1)
        else:
            norm_values = np.zeros(basis_vectors.shape[0])
            batch_size = basis_vectors.shape[0] // batches
            for i in range(0, basis_vectors.shape[0], batch_size):
                norm_values[i:i + batch_size] = np.linalg.norm((basis_vectors[i:i + batch_size].conj() @ cov) *
                                                               basis_vectors[i:i + batch_size], axis=1)
        norm_values = 1 / norm_values
        # treat the spectrum as 1d if the second dim is None
        if second_dim is None:
            indices, _ = scipy.signal.find_peaks(norm_values, height=self.thresh)
            return indices, norm_values, len(indices)
        elif third_dim is None:
            norm_values = norm_values.reshape(-1, second_dim)
            labeled, ncomponents = label(norm_values > self.thresh, structure=np.ones((3, 3), dtype=np.int))
        else:
            norm_values = norm_values.reshape(-1, second_dim, third_dim)
            labeled, ncomponents = label(norm_values > self.thresh,
                                         structure=np.ones((3, 3, 3), dtype=np.int))

        def get_current_component(norm_values, component_indx):
            if third_dim is None:
                return norm_values[component_indx[0]][component_indx[1]]
            return norm_values[component_indx[0]][component_indx[1]][component_indx[2]]

        indices = []
        for component in range(1, ncomponents + 1):
            component_indices = np.array(np.where(labeled == component)).T
            max, ind = 0, None
            for component_indx in component_indices:
                cur_comp = get_current_component(norm_values, component_indx)
                if cur_comp > max:
                    max = cur_comp
                    ind = component_indx
            indices.append(ind)
        return np.array(indices), norm_values, len(indices)
