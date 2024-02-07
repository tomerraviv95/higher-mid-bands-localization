import numpy as np
from sklearn.metrics import mean_squared_error

from python_code import conf
from python_code.channel import get_channel
from python_code.estimation import estimations_strings_dict
from python_code.estimation.estimate_physical_parameters import estimate_physical_parameters
from python_code.optimization import optimize_to_estimate_position
from python_code.plotting.estimations_printer import printer_main
from python_code.plotting.plotter import print_channel
from python_code.utils.bands_manipulation import get_bands_from_conf
from python_code.utils.constants import EstimatorType, DimensionType, C

np.random.seed(conf.seed)


def main():
    # BS locs of type [x,y,z]. x must be above the x-location of at least one BS.
    # In 2D - The array lies on the y-axis and points towards the x-axis.
    # In 3D - The array lies on the x-axis and y-axis and points towards the z-axis.
    ue_pos = np.array(conf.ue_pos)
    if conf.dimensions == DimensionType.Two.name:
        assert len(ue_pos) == 2
        print("x-axis up, y-axis right")
    else:
        assert len(ue_pos) == 3
        print("Right handed 3D axes")
    bands = get_bands_from_conf()
    estimator_type = estimations_strings_dict[conf.est_type]
    print(f"UE: {ue_pos}")
    print(estimator_type.name)
    print(f"Max distance supported by setup: {max([C / band.BW * band.K for band in bands])}[m]")
    print(f"Calculating from #{len(bands)} band")
    # ------------------------------------- #
    # Physical Parameters' Estimation Phase #
    # ------------------------------------- #
    estimations, per_band_y, bs_locs = [], [], []
    # for each bs
    for b in range(conf.B):
        # for each frequency sub-band
        for j, band in enumerate(bands):
            # generate the channel
            bs_ue_channel = get_channel(b, conf.ue_pos, band)
            if j == 0:
                print_channel(bs_ue_channel)
                bs_locs.append(bs_ue_channel.bs)
            # append to list
            per_band_y.append(bs_ue_channel.y)
        estimation, estimator = estimate_physical_parameters[conf.band_type](per_band_y, bands, estimator_type)
        estimations.append(estimation)
        printer_main(bs_ue_channel, estimation, estimator, estimator_type)
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
