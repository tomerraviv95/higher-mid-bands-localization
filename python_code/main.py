import numpy as np
from sklearn.metrics import mean_squared_error

from python_code import conf
from python_code.channel.generate_channel import get_channel
from python_code.estimation import estimations_strings_dict
from python_code.estimation.estimate_physical_parameters import estimate_physical_parameters
from python_code.optimization.position_optimizer import optimize_to_estimate_position
from python_code.plotting.estimations_printer import printer_main
from python_code.plotting.plotter import print_channel
from python_code.utils.bands_manipulation import get_bands_from_conf
from python_code.utils.constants import EstimatorType, C


def main():
    # random seed for run
    np.random.seed(conf.seed)
    # BS locs of type [x,y]. x must be above the x-location of at least one BS.
    # The array lies on the y-axis and points towards the x-axis.
    ue_pos = np.array(conf.ue_pos)
    assert len(ue_pos) == 2
    print("x-axis up, y-axis right")
    bands = get_bands_from_conf()
    estimator_type = estimations_strings_dict[conf.est_type]
    print(f"UE: {ue_pos}")
    print(estimator_type.name)
    print(f"Max distance supported by setup: {max([C / band.BW * band.K for band in bands])}[m]")
    print(f"Calculating from #{len(bands)} band - {[band.fc for band in bands]}")
    # ------------------------------------- #
    # Physical Parameters' Estimation Phase #
    # ------------------------------------- #
    estimations, per_band_y, bs_locs, bs_ue_channels = [], [], [], []
    # for each bs
    for b in range(1, conf.B + 1):
        # for each frequency sub-band
        for j, band in enumerate(bands):
            # generate the channel
            bs_ue_channel = get_channel(b, conf.ue_pos, band)
            if j == 0:
                print_channel(bs_ue_channel)
                bs_locs.append(bs_ue_channel.bs)
            # append to list
            per_band_y.append(bs_ue_channel.y)
            bs_ue_channels.append(bs_ue_channel)
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
    est_ue_pos = optimize_to_estimate_position(np.array(bs_locs), estimations)
    print(f"Estimated Position: {est_ue_pos}")
    rmse = mean_squared_error(ue_pos, est_ue_pos, squared=False)
    aoa_rmse = abs(estimations[0].AOA[0]-bs_ue_channels[0].AOA[0])
    toa_rmse = abs(estimations[0].TOA[0]-bs_ue_channels[0].TOA[0])
    print(f"RMSE: {rmse}, AOA Abs Error: {aoa_rmse}, TOA Abs Error: {toa_rmse}")
    return rmse, aoa_rmse, toa_rmse


if __name__ == "__main__":
    main()
