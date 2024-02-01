import numpy as np
from sklearn.metrics import mean_squared_error

from python_code import conf
from python_code.channel.bs_scatterers import create_bs_locs, create_scatter_points
from python_code.estimation.estimate_physical_parameters import estimate_physical_parameters
from python_code.optimization import optimize_to_estimate_position
from python_code.utils.bands_manipulation import get_bands_from_conf
from python_code.utils.constants import EstimatorType, DimensionType, C

np.random.seed(conf.seed)
estimations_strings_dict = {'angle': EstimatorType.ANGLE,
                            'time': EstimatorType.TIME,
                            'both': EstimatorType.ANGLE_TIME}


def main():
    # bs locs of type [x,y]. x must be above the x-location of at least one BS.
    # The array lies on the y-axis and points towards the x-axis.
    if conf.dimensions == DimensionType.Two.name:
        assert len(conf.ue_pos) == 2
        print("x-axis up, y-axis right")
    else:
        assert len(conf.ue_pos) == 3
        print("Right handed 3D axes")
    estimator_type = estimations_strings_dict[conf.est_type]
    bs_locs = create_bs_locs(conf.B)
    scatterers = create_scatter_points(conf.L)
    ue_pos = np.array(conf.ue_pos)
    bands = get_bands_from_conf()
    print(f"UE: {ue_pos}, Scatterers: {[str(scatter) for scatter in scatterers]}")
    print(estimator_type.name)
    print(f"Max distance supported by setup: {max([C / band.BW * band.K for band in bands])}[m]")
    print(f"Calculating from #{len(bands)} band")
    # ------------------------------------- #
    # Physical Parameters' Estimation Phase #
    # ------------------------------------- #
    estimations = estimate_physical_parameters(ue_pos, bs_locs, scatterers, estimator_type, bands)
    # must estimate both angle and time to estimate locations in this code version (theoretically you could use one)
    if estimator_type != EstimatorType.ANGLE_TIME:
        print("Position estimation is not implemented for only time/angle measurements")
        exit()
    # ------------------------- #
    # Position Estimation Phase #
    # ------------------------- #
    est_ue_pos = optimize_to_estimate_position(bs_locs, estimations)
    print(f"Estimated Position: {est_ue_pos}")
    rmse = mean_squared_error(ue_pos, est_ue_pos, squared=False)
    print(f"RMSE: {rmse}")
    return rmse


if __name__ == "__main__":
    main()
