from matplotlib import pyplot as plt

from python_code import conf
from python_code.estimation.estimator import Estimation

plt.style.use('dark_background')


def plot_angle_2d(estimator, estimation: Estimation):
    fig = plt.figure()
    plt.plot(estimator.angles_dict, estimator._spectrum, color="cyan")
    plt.plot(estimation.AOA, estimator._spectrum[estimator._indices], 'ro')
    plt.title('MUSIC for AOA Estimation')
    plt.xlabel('degree[rad]')
    plt.ylabel('MUSIC coefficient')
    plt.legend(['spectrum', 'Estimated AOAs'])
    plt.savefig('AOA.png', dpi=fig.dpi)
    plt.show()


def plot_angles_3d(estimator, estimation: Estimation):
    fig = plt.figure()
    plt.contourf(estimator.aoa_angles_dict, estimator.zoa_angles_dict,
                 estimator._spectrum.reshape(conf.zoa_res, conf.aoa_res, order='F'), cmap='magma')
    plt.plot(estimation.AOA, estimation.ZOA, 'ro')
    ax = plt.gca()
    ax.set_xlabel('AOA[us]')
    ax.set_ylabel('ZOA[rad]')
    plt.savefig('AOA_and_ZOA.png', dpi=fig.dpi)
    plt.show()


def plot_time(estimator, estimation: Estimation):
    fig = plt.figure()
    plt.plot(estimator.times_dict, estimator._spectrum, color="orange")
    plt.plot(estimation.TOA, estimator._spectrum[estimator._indices], 'ro')
    plt.title('MUSIC for Delay Estimation')
    plt.xlabel('TIME[us]')
    plt.ylabel('MUSIC coefficient')
    plt.legend(['spectrum', 'Estimated delays'])
    plt.savefig('delay.png', dpi=fig.dpi)
    plt.show()


def plot_angle_time_2d(estimator, estimation: Estimation):
    fig = plt.figure()
    plt.contourf(estimator.time_estimator.times_dict, estimator.angle_estimator.angles_dict,
                 estimator._spectrum.reshape(conf.aoa_res, conf.T_res), cmap='magma')
    ax = plt.gca()
    ax.set_xlabel('TIME[us]')
    ax.set_ylabel('AOA[rad]')
    plt.plot(estimation.TOA, estimation.AOA, 'ro')
    plt.savefig('AOA_and_delay_est.png', dpi=fig.dpi)
    plt.show()
