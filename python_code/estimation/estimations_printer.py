from python_code import conf
from python_code.plotting.plotter import plot_angle_time_2d, plot_angle_time_3d, plot_time, plot_angle_2d, \
    plot_angles_3d
from python_code.utils.constants import DimensionType


def angle_printer(bs_ue_channel, estimation, estimator):
    if estimation.AOA is not None:
        print(f"Estimated AOA: {sorted(estimation.AOA)}, GT AOA: {sorted(bs_ue_channel.AOA)}")
        if conf.dimensions == DimensionType.Three.name:
            print(f"Estimated ZOA: {sorted(estimation.ZOA)}, GT ZOA: {sorted(bs_ue_channel.ZOA)}")
        if conf.plot_estimation_results:
            if conf.dimensions == DimensionType.Three.name:
                plot_angles_3d(estimator, estimation)
            else:
                plot_angle_2d(estimator, estimation)


def time_printer(bs_ue_channel, estimation, estimator):
    if estimation.TOA is not None:
        print(f"Estimated TOA: {sorted(estimation.TOA)}, GT TOA: {sorted(bs_ue_channel.TOA)}")
        if conf.plot_estimation_results:
            plot_time(estimator, estimation)


def aoa_time_printer(bs_ue_channel, estimation, estimator):
    if estimation.AOA is not None and estimation.TOA is not None:
        if conf.dimensions == DimensionType.Two.name:
            print(f"Estimated AOA: {sorted(estimation.AOA)}, GT AOA: {sorted(bs_ue_channel.AOA)}")
            print(f"Estimated TOA: {sorted(estimation.TOA)}, GT TOA: {sorted(bs_ue_channel.TOA)}")
            if conf.plot_estimation_results:
                plot_angle_time_2d(estimator, estimation)
        if conf.dimensions == DimensionType.Three.name:
            print(f"Estimated AOA: {sorted(estimation.AOA)}, GT AOA: {sorted(bs_ue_channel.AOA)}")
            print(f"Estimated ZOA: {sorted(estimation.ZOA)}, GT ZOA: {sorted(bs_ue_channel.ZOA)}")
            print(f"Estimated TOA: {sorted(estimation.TOA)}, GT TOA: {sorted(bs_ue_channel.TOA)}")
            if conf.plot_estimation_results:
                plot_angle_time_3d(estimator, estimation)
