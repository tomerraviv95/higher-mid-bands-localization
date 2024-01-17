import numpy as np

from python_code import conf
from python_code.channel.channel_generator import get_channel, create_scatter_points
from python_code.estimators.estimator import AngleEstimator, TimeEstimator, AngleTimeEstimator, WidebandAngleEstimator
from python_code.plotting.plotter import plot_angle, plot_time, plot_angle_time
from python_code.utils.constants import EstimatorType

estimators = {EstimatorType.ANGLE: AngleEstimator,
              EstimatorType.WANGLE: WidebandAngleEstimator,
              EstimatorType.TIME: TimeEstimator,
              EstimatorType.ANGLE_TIME: AngleTimeEstimator}

np.random.seed(conf.seed)

if __name__ == "__main__":
    # bs locs of type [x,y]. x must be positive. The array lies on the y-axis and points towards the x-axis.
    bs_locs = [[0, 5], [0, -5]]
    scatterers = create_scatter_points(conf.L)
    ue_pos = np.array(conf.ue_pos)
    estimator_type = EstimatorType.ANGLE_TIME
    print(estimator_type.name)
    for bs_loc in bs_locs:
        bs_ue_channel = get_channel(np.array(bs_loc), ue_pos, scatterers)
        conf.max_time = max(bs_ue_channel.TOA) * 1.2
        estimator = estimators[estimator_type]()
        estimation = estimator.estimate(bs_ue_channel.y)
        print(f"BS #{bs_loc}")
        # estimating the angle only
        if estimator_type in [EstimatorType.ANGLE, EstimatorType.WANGLE]:
            print(f"Estimated: {sorted(estimation.AOA)}, GT: {sorted(bs_ue_channel.AOA)}")
            plot_angle(estimator, estimation)
        # estimating the time delay only
        elif estimator_type == EstimatorType.TIME:
            print(f"Estimated: {sorted(estimation.TOA)}, GT: {sorted(bs_ue_channel.TOA)}")
            plot_time(estimator, estimation)
        # combining both estimates, AOA & TOA
        elif estimator_type in [EstimatorType.ANGLE_TIME, EstimatorType.WANGLE_TIME]:
            if estimation.size > 0:
                print(f"Estimated: {sorted(estimation[:, 0])}, GT: {sorted(bs_ue_channel.TOA)}")
                print(sorted(estimation.AOA), sorted(bs_ue_channel.AOA))
                print(sorted(estimation.TOA), sorted(bs_ue_channel.TOA))
                plot_angle_time(estimator, estimation)
        else:
            raise ValueError("No such estimator type exists!!")
