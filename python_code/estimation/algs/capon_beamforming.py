import numpy as np
import scipy.signal
from scipy.ndimage.measurements import label


class CaponBeamforming:
    """
    The Capon Beamformer.
    Lorenz, R.G. and Boyd, S.P., 2005. Robust minimum variance beamforming.
    IEEE transactions on signal processing, 53(5), pp.1684-1696.
    """

    def __init__(self, thresh: float):
        self.thresh = thresh

    def run(self, y: np.ndarray, basis_vectors: np.ndarray, n_elements: int, second_dim: int = None,
            third_dim: int = None, batches=1):
        """
        y is channel observations
        basis vectors are the set of beamforming vectors in the dictionary
        n_elements is the dimension of the covariance matrix, .e.g antennas num for AOA, subcarriers num for TOA
        if second_dim is None then do a 1 dimensional peak search
        if second_dim is not None but third_dim is None do a 2 dimensions peak search
        if both are not None, do a 3 dimensions peak search
        batches allow for batching the computations of the beamforming objective to avoid memory crash
        return a tuple of the [max_indices, spectrum, L_hat] where L_hat is the estimated number of paths
        """
        # compute inverse covariance matrix
        cov = np.cov(y.reshape(n_elements, -1), bias=True)
        cov = np.linalg.inv(cov)
        cov = cov / np.linalg.norm(cov)
        # compute the Capon spectrum values for each basis vector
        norm_values = self._compute_capon_spectrum(basis_vectors, batches, cov)
        # treat the spectrum as 1d if the second dim is None
        if second_dim is None:
            indices, _ = scipy.signal.find_peaks(norm_values, height=self.thresh)
            return indices, norm_values, len(indices)
        # treat the spectrum as 2d if the third dim is None
        elif third_dim is None:
            norm_values = norm_values.reshape(-1, second_dim)
            labeled, ncomponents = label(norm_values > self.thresh, structure=np.ones((3, 3), dtype=np.int))
        # treat the spectrum as 3d
        else:
            norm_values = norm_values.reshape(-1, second_dim, third_dim)
            labeled, ncomponents = label(norm_values > self.thresh,
                                         structure=np.ones((3, 3, 3), dtype=np.int))

        def get_current_component(norm_values, component_indx):
            if third_dim is None:
                return norm_values[component_indx[0]][component_indx[1]]
            return norm_values[component_indx[0]][component_indx[1]][component_indx[2]]

        # in case of 2d or 3d spectrum, finding regions of peaks then taking the maximum in each region
        # as the representative for that region
        indices = []
        for component in range(1, ncomponents + 1):
            # get the region indices
            component_indices = np.array(np.where(labeled == component)).T
            max, ind = 0, None
            # look for the maximum value and indices in that region
            for component_indx in component_indices:
                cur_comp = get_current_component(norm_values, component_indx)
                if cur_comp > max:
                    max = cur_comp
                    ind = component_indx
            # add the corresponding index of the maximum value
            indices.append(ind)
        return np.array(indices), norm_values, len(indices)

    def _compute_capon_spectrum(self, basis_vectors: np.ndarray, batches: int, cov: np.ndarray):
        # compute with a single batch - no memory issues
        if batches == 1:
            norm_values = np.linalg.norm((basis_vectors.conj() @ cov) * basis_vectors, axis=1)
        # do batched computations to avoid memory crash
        else:
            norm_values = np.zeros(basis_vectors.shape[0])
            batch_size = basis_vectors.shape[0] // batches
            for i in range(0, basis_vectors.shape[0], batch_size):
                norm_values[i:i + batch_size] = np.linalg.norm((basis_vectors[i:i + batch_size].conj() @ cov) *
                                                               basis_vectors[i:i + batch_size], axis=1)
        norm_values = 1 / norm_values
        return norm_values
