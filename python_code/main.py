import numpy as np

from python_code import conf
from python_code.channel.channel_generator import get_channel
from python_code.estimators.estimator import AngleEstimator, TimeEstimator, AngleTimeEstimator, \
    WidebandAngleTimeEstimator, WidebandAngleEstimator
from python_code.plotting.plotter import plot_angle, plot_time, plot_angle_time
from python_code.utils.constants import EstimatorType

estimators = {EstimatorType.ANGLE: AngleEstimator,
              EstimatorType.WANGLE: WidebandAngleEstimator,
              EstimatorType.TIME: TimeEstimator,
              EstimatorType.ANGLE_TIME: AngleTimeEstimator,
              EstimatorType.WANGLE_TIME: WidebandAngleTimeEstimator}

np.random.seed(conf.seed)

if __name__ == "__main__":
    channel_instance = get_channel()
    y = channel_instance.y
    estimator_type = EstimatorType.ANGLE
    estimator = estimators[estimator_type]()
    est_values = estimator.estimate(y)
    # estimating the angle only
    print(estimator_type.name)
    if estimator_type in [EstimatorType.ANGLE, EstimatorType.WANGLE]:
        print(sorted(est_values), sorted(channel_instance.AOA))
        plot_angle(estimator, est_values)
    # estimating the time delay only
    elif estimator_type == EstimatorType.TIME:
        print(sorted(est_values), sorted(channel_instance.TOA))
        plot_time(estimator, est_values)
    # combining both estimates, AOA & TOA
    elif estimator_type in [EstimatorType.ANGLE_TIME, EstimatorType.WANGLE_TIME]:
        if est_values.size > 0:
            print(sorted(est_values[:, 0]), sorted(channel_instance.AOA))
            print(sorted(est_values[:, 1]), sorted(channel_instance.TOA))
            plot_angle_time(estimator, est_values)
    else:
        raise ValueError("No such estimator type exists!!")
