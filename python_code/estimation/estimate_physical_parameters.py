import numpy as np

from python_code import conf
from python_code.channel.channel_generator import get_channel
from python_code.estimation.estimator import AngleEstimator, WidebandAngleEstimator, TimeEstimator, AngleTimeEstimator
from python_code.plotting.plotter import plot_angle, plot_time, plot_angle_time
from python_code.utils.constants import EstimatorType

estimators = {EstimatorType.ANGLE: AngleEstimator,
              EstimatorType.WIDE_ANGLE: WidebandAngleEstimator,
              EstimatorType.TIME: TimeEstimator,
              EstimatorType.ANGLE_TIME: AngleTimeEstimator}

def estimate_physical_parameters(ue_pos, bs_locs, scatterers, estimator_type):
    estimations = []
    for i, bs_loc in enumerate(bs_locs):
        bs_ue_channel = get_channel(np.array(bs_loc), ue_pos, scatterers)
        conf.max_time = max(bs_ue_channel.TOA) * 1.2
        estimator = estimators[estimator_type]()
        estimation = estimator.estimate(bs_ue_channel.y)
        estimations.append(estimation)
        print(f"BS #{i} - {bs_loc}")
        # estimating the angle only
        if estimator_type in [EstimatorType.ANGLE, EstimatorType.WIDE_ANGLE]:
            print(f"Estimated: {sorted(estimation.AOA)}, GT: {sorted(bs_ue_channel.AOA)}")
            if conf.plot_estimation_results:
                plot_angle(estimator, estimation)
        # estimating the time delay only
        elif estimator_type == EstimatorType.TIME:
            print(f"Estimated: {sorted(estimation.TOA)}, GT: {sorted(bs_ue_channel.TOA)}")
            if conf.plot_estimation_results:
                plot_time(estimator, estimation)
        # combining both estimates, AOA & TOA
        elif estimator_type in [EstimatorType.ANGLE_TIME]:
            if len(estimation.AOA) > 0:
                print(f"Estimated: {sorted(estimation.AOA)}, GT: {sorted(bs_ue_channel.AOA)}")
                print(f"Estimated: {sorted(estimation.TOA)}, GT: {sorted(bs_ue_channel.TOA)}")
                if conf.plot_estimation_results:
                    plot_angle_time(estimator, estimation)
        else:
            raise ValueError("No such estimator type exists!!")
    return estimations
