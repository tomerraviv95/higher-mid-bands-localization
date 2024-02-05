from matplotlib import pyplot as plt

from python_code.utils.constants import Channel, C, Estimation

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
    plt.title('Spectrum for Delay Estimation')
    plt.xlabel('TIME[us]')
    plt.ylabel('Spectrum coefficient')
    plt.legend(['spectrum', 'estimated delays'])
    plt.savefig('delay.png', dpi=fig.dpi)
    plt.show()


def plot_angle_time_2d(estimator, estimation: Estimation):
    fig = plt.figure()
    plt.contourf(estimator.time_estimator.times_dict, estimator.angle_estimator.aoa_angles_dict,
                 estimator._spectrum.reshape(len(estimator.angle_estimator.aoa_angles_dict),
                                             len(estimator.time_estimator.times_dict), order='F'),
                 map='magma')
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


def print_channel(bs_ue_channel: Channel):
    if bs_ue_channel.ZOA is not None:
        zoa_string = f", ZOA[rad]: {round(bs_ue_channel.ZOA[0], 3)}"
    else:
        zoa_string = ""
    print(f"Distance to user {bs_ue_channel.TOA[0] * C}[m], "
          f"TOA[us]: {round(bs_ue_channel.TOA[0], 3)}, "
          f"AOA[rad]: {round(bs_ue_channel.AOA[0], 3)}" + zoa_string)
