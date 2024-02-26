import numpy as np
import torch

from python_code import DEVICE
from python_code.utils.constants import MAX_COMP

STEPS = 100


class CaponBeamforming:
    """
    The Capon Beamformer.
    Lorenz, R.G. and Boyd, S.P., 2005. Robust minimum variance beamforming.
    IEEE transactions on signal processing, 53(5), pp.1684-1696.
    """

    def __init__(self, thresh: float):
        self.thresh = thresh

    def run(self, y: np.ndarray, basis_vectors: np.ndarray, second_dim: int = None,
            use_gpu=False):
        """
        y is channel observations
        basis vectors are the set of beamforming vectors in the dictionary
        n_elements is the dimension of the covariance matrix, .e.g antennas num for AOA, subcarriers num for TOA
        if second_dim is None then do a 1 dimensional peak search
        if second_dim is not None do a 2 dimensions peak search
        batches allow for batching the computations of the beamforming objective to avoid memory crash
        return a tuple of the [max_indices, spectrum, L_hat] where L_hat is the estimated number of paths
        """
        # compute the Capon spectrum values for each basis vector
        norm_values = self._compute_beamforming_spectrum(basis_vectors, use_gpu, y)
        # find the peaks in the spectrum
        if second_dim is not None:
            norm_values = norm_values.reshape(-1, second_dim)
            maximum_ind = np.array(np.unravel_index(np.argmax(norm_values, axis=None), norm_values.shape))
            return np.array([maximum_ind]), norm_values, 0
            maximum_inds = np.array(np.unravel_index(np.argsort(norm_values, axis=None)[::-1], norm_values.shape))
            maximum_inds = maximum_inds[:, :5].T
            maximum_inds = maximum_inds[maximum_inds[:, 1].argsort()]
            for maximum_ind in maximum_inds:
                print(maximum_ind)
                if norm_values[maximum_ind[0], maximum_ind[1]] > self.thresh:
                    return np.array([maximum_ind]), norm_values, 0
            return np.array([]), norm_values, 0
        maximum_ind = np.argmax(norm_values)
        return np.array([maximum_ind]), norm_values, 0


    def _compute_beamforming_spectrum(self, basis_vectors: np.ndarray, use_gpu: bool, y: np.ndarray):
        # compute with cpu - no cpu/memory issues
        if not use_gpu:
            norm_values = np.zeros(basis_vectors.shape[0])
            for i in range(0, basis_vectors.shape[0], STEPS):
                cur_basis_vectors = basis_vectors[i:i + STEPS]
                norm_values[i:i + STEPS] = np.linalg.norm((cur_basis_vectors.conj() @ cov) * cur_basis_vectors,
                                                          axis=1)
            norm_values = 1 / norm_values
        # do calculations on GPU - much faster for big matrices
        else:
            y_tensor = torch.tensor(y).to(DEVICE).type(torch.cfloat)
            left_steering = basis_vectors[0].unsqueeze(0).type(torch.cfloat)
            right_steering = basis_vectors[1].T.type(torch.cfloat)
            norm_values = torch.zeros([left_steering.shape[1], right_steering.shape[1]]).to(DEVICE).type(torch.cfloat)
            result_left = torch.matmul(left_steering, y_tensor)
            for sample_ind in range(result_left.shape[0]):
                cur_left_result = result_left[sample_ind]
                norm_values += torch.matmul(cur_left_result, right_steering)
            norm_values /= result_left.shape[0]
            norm_values = torch.abs(norm_values)
            # norm_values = torch.matmul(basis_vectors[0],y_tensor,basis_vectors[1])
            # res0, max_batches = basis_vectors(batch_ind_start=0, batch_ind_end=1)
            # batch_size = res0.shape[0]
            # norm_values = torch.zeros(batch_size * max_batches).to(DEVICE)
            # for i in range(0, max_batches):
            #     cur_basis_vectors = basis_vectors(batch_ind_start=i, batch_ind_end=i + 1)[0]
            #     multiplication = torch.matmul(cur_basis_vectors.conj(),cov)
            #     norm_values[i * batch_size:(i + 1) * batch_size] = torch.linalg.norm(
            #         multiplication,
            #         dim=1)
            norm_values = norm_values.cpu().numpy()
        return norm_values
