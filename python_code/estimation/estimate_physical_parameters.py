from python_code import conf
from python_code.channel import get_channel
from python_code.estimation.angle import AngleEstimator3D, AngleEstimator2D
from python_code.estimation.angle_time import AngleTimeEstimator2D, AngleTimeEstimator3D
from python_code.estimation.estimations_printer import aoa_time_printer, time_printer, angle_printer
from python_code.estimation.time import TimeEstimator2D, TimeEstimator3D
from python_code.utils.constants import EstimatorType, DimensionType

estimators = {
    EstimatorType.ANGLE: {DimensionType.Three.name: AngleEstimator3D, DimensionType.Two.name: AngleEstimator2D},
    EstimatorType.TIME: {DimensionType.Three.name: TimeEstimator3D, DimensionType.Two.name: TimeEstimator2D},
    EstimatorType.ANGLE_TIME: {DimensionType.Three.name: AngleTimeEstimator3D,
                               DimensionType.Two.name: AngleTimeEstimator2D}}


def estimate_physical_parameters(ue_pos, bs_locs, scatterers, estimator_type):
    estimations = []
    # for each bs
    for i, bs_loc in enumerate(bs_locs):
        # generate the channel
        bs_ue_channel = get_channel(bs_loc, ue_pos, scatterers)
        conf.max_time = max(bs_ue_channel.TOA) * 1.2
        # choose the estimator based on the desired type
        estimator = estimators[estimator_type][conf.dimensions]()
        # estimate delay / AOA / ZOA parameters for the current bs
        estimation = estimator.estimate(bs_ue_channel.y)
        estimations.append(estimation)
        print(f"BS #{i} - {bs_loc}")
        # print the angle result + graph only
        if estimator_type == EstimatorType.ANGLE:
            angle_printer(bs_ue_channel, estimation, estimator)
        # print the delay result + graph only
        elif estimator_type == EstimatorType.TIME:
            time_printer(bs_ue_channel, estimation, estimator)
        # print the angle + delay, result + graph only
        elif estimator_type == EstimatorType.ANGLE_TIME:
            aoa_time_printer(bs_ue_channel, estimation, estimator)
        else:
            raise ValueError("No such estimator type exists!!")
    return estimations
