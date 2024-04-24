from typing import List

import numpy as np
import scipy.linalg
import scipy.signal
import torch

from python_code import DEVICE
from python_code.utils.constants import ALG_THRESHOLD

EIGEN_THRESH = 0.1


def cluster(evs: np.ndarray) -> int:
    """
    Estimates multiplicity of smallest eigenvalue.
    evs - the eigenvalues in descending order.
    returns - the number of eigenvalues that belong to the noise subspace.
    """
    # simplest clustering method: with threshold
    return evs[np.where(abs(evs) / abs(evs)[0] < EIGEN_THRESH)].shape[0]


class MUSIC:
    """
    The batched multi-frequency implementation of MUSIC algorithm
    Based on the code in this repo https://github.com/DA-MUSIC/ICASSP22
    """

    def run(self, y: List[np.ndarray], basis_vectors: List[np.ndarray], second_dim: bool = False,
            use_gpu: bool = False):
        if not second_dim:
            print("Only 2D MUSIC (both AOA and TOA) is currently supported")
            raise ValueError("Not supported")
        K = len(y)
        peak, chosen_k = None, None
        norm_values_list = []
        for k in range(K):
            # for each sub-band
            cur_y, cur_basis_vectors = y[k], basis_vectors[k]
            cur_y = cur_y.reshape(-1, cur_y.shape[2])
            noise_mat = self.calculate_noise_subspace_matrix(cur_y)
            if not use_gpu:
                # if GPU is not available simply compute the 2d combination of the vectors
                steering_vectors = np.kron(cur_basis_vectors[0], cur_basis_vectors[1])
                multiplication = steering_vectors @ noise_mat.conj()
                # and straightforward compute the spectrum
                music_coef = 1 / scipy.linalg.norm(multiplication, axis=0)
                norm_values += np.log10(10 * music_coef / music_coef.min())
            else:
                # if GPU is available run a batched computation of the spectrum
                # using the GPU to compute the multiplication fast
                aoa_steering_vectors = cur_basis_vectors[0].type(torch.cfloat).to(DEVICE)
                toa_steering_vectors = cur_basis_vectors[1].type(torch.cfloat).to(DEVICE)
                conj_noise_mat = torch.tensor(noise_mat).type(torch.cfloat).to(DEVICE).conj()
                max_batches = aoa_steering_vectors.shape[0]
                batch_size = toa_steering_vectors.shape[0]
                music_coef = torch.zeros([max_batches, batch_size]).to(DEVICE)
                for i in range(max_batches):
                    # this part is the MUSIC spectrum calculation
                    cur_aoa = aoa_steering_vectors[i].reshape(1, -1)
                    aoa_toa_combinations = torch.kron(cur_aoa, toa_steering_vectors)
                    multiplication = aoa_toa_combinations @ conj_noise_mat
                    music_coef[i] = 1 / torch.linalg.norm(multiplication, dim=1)
                norm_values_tensor = torch.log10(10 * music_coef / music_coef.min())
                norm_values = norm_values_tensor.cpu().numpy()
            norm_values_list.append(norm_values)
            maximum_ind = np.array(np.unravel_index(np.argmax(norm_values, axis=None), norm_values.shape))
            # only if peak is not noisy
            if norm_values[maximum_ind[0], maximum_ind[1]] > ALG_THRESHOLD * np.mean(norm_values):
                peak = maximum_ind
                chosen_k = k

        if chosen_k is None:
            peak = maximum_ind
            chosen_k = K - 1
        self.k = chosen_k
        return np.array([peak]), norm_values_list[self.k]

    def calculate_noise_subspace_matrix(self, cur_y: np.ndarray) -> np.ndarray:
        cov = np.cov(cur_y, bias=True)
        eigenvalues, eigenvectors = scipy.linalg.eig(cov)
        smallest_eigenvalue_times = cluster(eigenvalues)
        l_hat = len(cov) - smallest_eigenvalue_times
        noise_subspace_mat = eigenvectors[:, l_hat:]
        return noise_subspace_mat
