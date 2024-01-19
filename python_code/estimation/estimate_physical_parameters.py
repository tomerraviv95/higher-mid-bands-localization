import numpy as np

from python_code import conf
from python_code.channel.channel_generator_2d import get_2d_channel
from python_code.channel.channel_generator_3d import get_3d_channel
from python_code.estimation.angle import AngleEstimator3D, AngleEstimator2D, WidebandAngleEstimator
from python_code.estimation.angle_time import AngleTimeEstimator2D, AngleTimeEstimator3D
from python_code.estimation.time import TimeEstimator2D, TimeEstimator3D
from python_code.plotting.plotter import plot_angle_2d, plot_time, plot_angle_time_2d, plot_angles_3d
from python_code.utils.constants import EstimatorType, DimensionType

estimators = {
    EstimatorType.ANGLE: {DimensionType.Three.name: AngleEstimator3D, DimensionType.Two.name: AngleEstimator2D},
    EstimatorType.WIDE_ANGLE: WidebandAngleEstimator,
    EstimatorType.TIME: {DimensionType.Three.name: TimeEstimator3D, DimensionType.Two.name: TimeEstimator2D},
    EstimatorType.ANGLE_TIME: {DimensionType.Three.name: AngleTimeEstimator3D,
                               DimensionType.Two.name: AngleTimeEstimator2D}}


def estimate_physical_parameters(ue_pos, bs_locs, scatterers, estimator_type):
    estimations = []
    # for each bs
    for i, bs_loc in enumerate(bs_locs):
        # get the channel
        if conf.dimensions == DimensionType.Three.name:
            bs_ue_channel = get_3d_channel(np.array(bs_loc), ue_pos, scatterers)
        else:
            bs_ue_channel = get_2d_channel(np.array(bs_loc), ue_pos, scatterers)
        conf.max_time = max(bs_ue_channel.TOA) * 1.2
        estimator = estimators[estimator_type][conf.dimensions]()
        # estimate delay / AOA / ZOA parameters for the current bs
        estimation = estimator.estimate(bs_ue_channel.y)
        estimations.append(estimation)
        print(f"BS #{i} - {bs_loc}")
        # estimating the angle only
        if estimator_type in [EstimatorType.ANGLE, EstimatorType.WIDE_ANGLE]:
            print(f"Estimated AOA: {sorted(estimation.AOA)}, GT AOA: {sorted(bs_ue_channel.AOA)}")
            if conf.dimensions == DimensionType.Three.name:
                print(f"Estimated ZOA: {sorted(estimation.ZOA)}, GT ZOA: {sorted(bs_ue_channel.ZOA)}")
            if conf.plot_estimation_results:
                if conf.dimensions == DimensionType.Three.name:
                    plot_angles_3d(estimator, estimation)
                else:
                    plot_angle_2d(estimator, estimation)
        # estimating the time delay only
        elif estimator_type == EstimatorType.TIME:
            print(f"Estimated TOA: {sorted(estimation.TOA)}, GT TOA: {sorted(bs_ue_channel.TOA)}")
            if conf.plot_estimation_results:
                plot_time(estimator, estimation)
        # combining both estimates, AOA & TOA
        elif estimator_type in [EstimatorType.ANGLE_TIME]:
            if len(estimation.AOA) > 0:
                if conf.dimensions == DimensionType.Two.name:
                    print(f"Estimated AOA: {sorted(estimation.AOA)}, GT AOA: {sorted(bs_ue_channel.AOA)}")
                    print(f"Estimated TOA: {sorted(estimation.TOA)}, GT TOA: {sorted(bs_ue_channel.TOA)}")
                    if conf.plot_estimation_results:
                        plot_angle_time_2d(estimator, estimation)
                if conf.dimensions == DimensionType.Three.name:
                    print(f"Estimated AOA: {sorted(estimation.AOA)}, GT AOA: {sorted(bs_ue_channel.AOA)}")
                    print(f"Estimated AOA: {sorted(estimation.ZOA)}, GT AOA: {sorted(bs_ue_channel.ZOA)}")
                    print(f"Estimated TOA: {sorted(estimation.TOA)}, GT TOA: {sorted(bs_ue_channel.TOA)}")
        else:
            raise ValueError("No such estimator type exists!!")
    return estimations
