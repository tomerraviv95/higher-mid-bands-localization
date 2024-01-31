import math

from matplotlib import pyplot as plt

from python_code import conf
from python_code.estimation.angle_time import Estimation

plt.style.use('dark_background')


def plot_angle_2d(estimator, estimation: Estimation):
    fig = plt.figure()
    plt.plot(estimator.aoa_angles_dict, estimator._spectrum, color="cyan")
    plt.plot(estimation.AOA, estimator._spectrum[estimator._indices], 'ro')
    plt.title('Spectrum for AOA Estimation')
    plt.xlabel(f'Degree[rad]')
    plt.ylabel('Spectrum coefficient')
    plt.legend(['spectrum', 'estimated AOAs'])
    plt.savefig('AOA_2d.png', dpi=fig.dpi)
    plt.show()


def plot_angles_3d(estimator, estimation: Estimation):
    fig = plt.figure()
    plt.contourf(estimator.zoa_angles_dict, estimator.aoa_angles_dict,
                 estimator._spectrum.reshape(len(estimator.aoa_angles_dict), len(estimator.zoa_angles_dict),
                                             order='F'),
                 cmap='magma')
    plt.plot(estimation.ZOA, estimation.AOA, 'ro')
    ax = plt.gca()
    ax.set_ylabel('AOA[rad]')
    ax.set_xlabel('ZOA[rad]')
    plt.savefig('AOA_3d.png', dpi=fig.dpi)
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
    plt.contourf(estimator.time_estimator.times_dict, estimator.angle_estimator.aoa_angles_dict,
                 estimator._spectrum.reshape(math.pi // conf.aoa_res, conf.T_res), cmap='magma')
    ax = plt.gca()
    ax.set_xlabel('TIME[us]')
    ax.set_ylabel('AOA[rad]')
    plt.plot(estimation.TOA, estimation.AOA, 'ro')
    plt.savefig('AOA_and_delay_est.png', dpi=fig.dpi)
    plt.show()


def plot_angle_time_3d(estimator, estimation: Estimation):
    from mpl_toolkits.mplot3d import Axes3D
    ax = Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Creating plot
    ax.scatter(estimation.TOA, estimation.AOA, estimation.ZOA, alpha=1, color='red', s=50)
    ax.set_xlabel('TIME[us]')
    ax.set_ylabel('AOA[rad]')
    ax.set_zlabel('ZOA[rad]')
    plt.savefig('AOA_ZOA_and_delay_est.png', dpi=fig.dpi)
    plt.show()
