from matplotlib import pyplot as plt

from python_code import conf

plt.style.use('dark_background')


def plot_angle(estimator, est_values):
    fig = plt.figure()
    plt.plot(estimator.angles_dict, estimator._spectrum, color="cyan")
    plt.plot(est_values, estimator._spectrum[estimator._indices], 'ro')
    plt.title('MUSIC for AOA Estimation')
    plt.xlabel('degree[rad]')
    plt.ylabel('MUSIC coefficient')
    plt.legend(['spectrum', 'Estimated AOAs'])
    plt.savefig('AOA.png', dpi=fig.dpi)
    plt.show()


def plot_time(estimator, est_values):
    fig = plt.figure()
    plt.plot(estimator.times_dict, estimator._spectrum, color="orange")
    plt.plot(est_values, estimator._spectrum[estimator._indices], 'ro')
    plt.title('MUSIC for Delay Estimation')
    plt.xlabel('TIME[us]')
    plt.ylabel('MUSIC coefficient')
    plt.legend(['spectrum', 'Estimated delays'])
    plt.savefig('delay.png', dpi=fig.dpi)
    plt.show()


def plot_angle_time(estimator, est_values):
    fig = plt.figure()
    plt.contourf(estimator.time_estimator.times_dict, estimator.angle_estimator.angles_dict,
                 estimator._spectrum.reshape(conf.Nb, conf.T_res), cmap='magma')
    ax = plt.gca()
    ax.set_xlabel('TIME[us]')
    ax.set_ylabel('AOA[rad]')
    plt.savefig('AOA_and_delay.png', dpi=fig.dpi)
    plt.plot(est_values[:, 1], est_values[:, 0], 'ro')
    plt.savefig('AOA_and_delay_est.png', dpi=fig.dpi)
    plt.show()
