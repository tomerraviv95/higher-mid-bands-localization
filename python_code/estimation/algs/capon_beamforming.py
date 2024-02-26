import numpy as np
import torch

from python_code import DEVICE

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
            maximum_ind = np.array(np.unravel_index(np.argmax(norm_values, axis=None), norm_values.shape))
        else:
            maximum_ind = np.argmax(norm_values)
        return np.array([maximum_ind]), norm_values, 0

    def _compute_beamforming_spectrum(self, basis_vectors: np.ndarray, use_gpu: bool, y: np.ndarray):
        # compute with cpu - no cpu/memory issues
        if not use_gpu:
            aoas = basis_vectors[0]
            toas = basis_vectors[1]
            left_matmul = np.einsum('ij,jmk->imk', aoas.conj(), y)
            right_matmul = np.einsum('ijk,jm->imk', left_matmul, toas.conj().T)
            norm_values = np.linalg.norm(right_matmul, axis=2)
        # do calculations on GPU - much faster for big matrices
        else:
            y = torch.tensor(y).to(DEVICE)
            aoas = basis_vectors[0]
            toas = basis_vectors[1]
            left_matmul = torch.einsum('ij,jmk->imk', aoas.conj(), y)
            right_matmul = torch.einsum('ijk,jm->imk', left_matmul, toas.conj().T)
            norm_values = torch.linalg.norm(right_matmul, axis=2)
        return norm_values
