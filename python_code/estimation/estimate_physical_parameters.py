from python_code import conf
from python_code.channel import get_channel
from python_code.estimation.angle import AngleEstimator3D, AngleEstimator2D
from python_code.estimation.angle_time import AngleTimeEstimator2D, AngleTimeEstimator3D
from python_code.estimation.estimations_combining import combine_estimations
from python_code.estimation.estimations_printer import aoa_time_printer, time_printer, angle_printer
from python_code.estimation.time import TimeEstimator2D, TimeEstimator3D
from python_code.utils.constants import EstimatorType, DimensionType, C

estimators = {
    EstimatorType.ANGLE: {DimensionType.Three.name: AngleEstimator3D, DimensionType.Two.name: AngleEstimator2D},
    EstimatorType.TIME: {DimensionType.Three.name: TimeEstimator3D, DimensionType.Two.name: TimeEstimator2D},
    EstimatorType.ANGLE_TIME: {DimensionType.Three.name: AngleTimeEstimator3D,
                               DimensionType.Two.name: AngleTimeEstimator2D}}


def estimate_physical_parameters(ue_pos, bs_locs, scatterers, estimator_type, bands):
    estimations = []
    # for each bs
    for i, bs_loc in enumerate(bs_locs):
        # for each frequency sub-band
        for j, band in enumerate(bands):
            # generate the channel
            bs_ue_channel = get_channel(bs_loc, ue_pos, scatterers, band)
            if j == 0:
                print(f"Distance to user {bs_ue_channel.TOA[0] * C}[m], "
                      f"TOA[us]: {round(bs_ue_channel.TOA[0], 3)}, "
                      f"AOA to user {round(bs_ue_channel.AOA[0], 3)}[rad]")
            # choose the estimator based on the desired type
            estimator = estimators[estimator_type][conf.dimensions](band)
            # estimate delay / AOA / ZOA parameters for the current bs
            estimation = estimator.estimate(bs_ue_channel.y)
            estimations.append(estimation)
        estimation = combine_estimations(estimations)
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
