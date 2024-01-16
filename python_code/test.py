import numpy as np
from matplotlib import pyplot as plt

from music import music
from python_code import conf
from python_code.channel.channel_generator import get_channel
from python_code.utils.basis_functions import compute_angle_options, compute_time_options

plt.style.use('dark_background')

if __name__ =="__main__":
    channel_instance = get_channel()
    y = channel_instance.y
    # Create dictionary
    aa = np.linspace(-np.pi / 2, np.pi / 2, conf.Nb)  # dictionary of spatial frequencies

    np.random.seed(conf.seed)

    # angle of arrival
    ANGLE = True
    if ANGLE:
        angle_cov = np.cov(y.reshape(conf.Nr, -1), bias=True)
        angle_values = np.arange(conf.Nr)
        angle_options = compute_angle_options(aa, values=angle_values)
        indices, spectrum = music(cov=angle_cov, L=conf.L, n_elements=conf.Nr, options=angle_options)
        fig = plt.figure()
        plt.plot(aa, spectrum, color="cyan")
        plt.plot(aa[indices], spectrum[indices], 'ro')
        plt.title('MUSIC for AOA Estimation')
        plt.xlabel('degree[rad]')
        plt.ylabel('MUSIC coefficient')
        plt.legend(['spectrum', 'Estimated AOAs'])
        plt.savefig('AOA.png', dpi=fig.dpi)
        plt.show()

    # time delay
    TIME_DELAY = True
    if TIME_DELAY:
        fig = plt.figure()
        time_cov = np.cov(np.transpose(y, [1, 0, 2]).reshape(conf.K, -1), bias=True)
        time_values = np.linspace(0, conf.max_time, conf.T_res)
        time_options = compute_time_options(conf.fc, conf.K, conf.BW, values=time_values)
        indices, spectrum = music(cov=time_cov, L=conf.L, n_elements=conf.K, options=time_options)
        plt.plot(time_values, spectrum, color="orange")
        plt.plot(time_values[indices], spectrum[indices], 'ro')
        plt.title('MUSIC for Delay Estimation')
        plt.xlabel('time[us]')
        plt.ylabel('MUSIC coefficient')
        plt.legend(['spectrum', 'Estimated delays'])
        plt.savefig('delay.png', dpi=fig.dpi)
        plt.show()

    # combining both estimates, 2-D AOA & TOA
    BOTH = True
    if BOTH:
        fig = plt.figure()
        angle_time_cov = np.cov(y.reshape(conf.K * conf.Nr, -1), bias=True)
        angle_values = np.arange(conf.Nr)
        angle_options = compute_angle_options(aa, values=angle_values)
        time_values = np.linspace(0, conf.max_time, conf.T_res)
        time_options = compute_time_options(conf.fc, conf.K, conf.BW, values=time_values)
        angle_time_options = np.kron(angle_options, time_options)
        indices, spectrum = music(cov=angle_time_cov, L=conf.L, n_elements=conf.Nr * conf.K, options=angle_time_options)
        angle_indices = indices // conf.T_res
        time_indices = indices % conf.T_res
        filtered_peaks = []
        for uniq_time in np.unique(time_indices):
            avg_angle = int(np.mean(angle_indices[time_indices == uniq_time]))
            filtered_peaks.append([aa[avg_angle], time_values[uniq_time]])
        filtered_peaks = np.array(filtered_peaks)
        fig = plt.figure()
        plt.contourf(time_values, aa, spectrum.reshape(conf.Nb, conf.T_res), cmap='magma')
        ax = plt.gca()
        ax.set_xlabel('time[us]')
        ax.set_ylabel('AOA[rad]')
        plt.savefig('AOA_and_delay.png', dpi=fig.dpi)
        plt.plot(filtered_peaks[:, 1], filtered_peaks[:, 0], 'ro')
        plt.savefig('AOA_and_delay_est.png', dpi=fig.dpi)
        plt.show()
