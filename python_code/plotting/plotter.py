from matplotlib import pyplot as plt

from python_code import conf
from python_code.utils.constants import Channel, C, Estimation, DEG

plt.style.use('dark_background')

T_WINDOW, A_WINDOW = 10, 15


def plot_angle(estimator, estimation: Estimation):
    fig = plt.figure()
    plt.plot(estimator.aoa_angles_dict, estimator._spectrum, color="cyan")
    plt.plot(estimation.AOA, estimator._spectrum[estimator._indices], 'ro')
    plt.title('Spectrum for AOA Estimation')
    plt.xlabel(f'Degree[rad]')
    plt.ylabel('Spectrum coefficient')
    plt.legend(['spectrum', 'estimated AOAs'])
    plt.savefig('AOA_2d.png', dpi=fig.dpi)
    plt.show()


def plot_time(estimator, estimation: Estimation):
    fig = plt.figure()
    plt.plot(estimator.times_dict, estimator._spectrum, color="orange")
    plt.plot(estimation.TOA, estimator._spectrum[estimator._indices], 'ro')
    plt.title('Spectrum for Delay Estimation')
    plt.xlabel('TIME[us]')
    plt.ylabel('Spectrum coefficient')
    plt.legend(['spectrum', 'estimated delays'])
    plt.savefig('delay.png', dpi=fig.dpi)
    plt.show()


def plot_angle_time(estimator, estimation: Estimation):
    fig = plt.figure()
    aoa_dict = estimator.angle_estimator.aoa_angles_dict
    if estimator.time_estimator.multi_band:
        times_dict = estimator.time_estimator.times_dict[estimator.k]
    else:
        times_dict = estimator.time_estimator.times_dict
    plt.contourf(times_dict, aoa_dict, estimator._spectrum.reshape(len(aoa_dict), len(times_dict), order='F'),
                 map='magma')
    ax = plt.gca()
    ax.set_xlabel('TIME[us]')
    ax.set_ylabel('AOA[rad]')
    plt.plot(estimation.TOA, estimation.AOA, 'ro')
    plt.axis([estimation.TOA - T_WINDOW * conf.T_res, estimation.TOA + T_WINDOW * conf.T_res,
              estimation.AOA - A_WINDOW * DEG, estimation.AOA + A_WINDOW * DEG])
    plt.savefig('AOA_and_delay_est.png', dpi=fig.dpi)
    plt.show()


def print_channel(bs_ue_channel: Channel):
    if bs_ue_channel.ZOA is not None:
        zoa_string = f", ZOA[rad]: {round(bs_ue_channel.ZOA[0], 3)}"
    else:
        zoa_string = ""
    print(f"Distance to user {bs_ue_channel.TOA[0] * C}[m], "
          f"TOA[us]: {round(bs_ue_channel.TOA[0], 3)}, "
          f"AOA[rad]: {round(bs_ue_channel.AOA[0], 3)}" + zoa_string)
