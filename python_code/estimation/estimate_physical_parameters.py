from typing import List, Tuple

from python_code import conf
from python_code.estimation import estimators
from python_code.estimation.estimations_combining import combine_estimations
from python_code.utils.bands_manipulation import Band
from python_code.utils.constants import EstimatorType, Channel, Estimation, BandType


def separate_bands_estimator(per_band_y: List[Channel], bands: List[Band], estimator_type: EstimatorType) -> Tuple[
    Estimation, object]:
    per_band_estimations = []
    # for each frequency sub-band
    for j, band in enumerate(bands):
        # choose the estimator based on the desired type
        estimator = estimators[estimator_type]([band])
        # estimate delay / AOA / ZOA parameters for the current bs
        estimation = estimator.estimate(per_band_y[j])
        per_band_estimations.append(estimation)
    # combine the estimations from the different bands together
    estimation = combine_estimations(per_band_estimations, bands, estimator_type)
    return estimation, estimator


def sim_bands_estimator(per_band_y: List[Channel], bands: List[Band], estimator_type: EstimatorType) -> Tuple[
    Estimation, object]:
    # choose the estimator based on the desired type
    estimator = estimators[estimator_type](bands)
    # estimate delay / AOA / ZOA parameters for the current bs based on all measurements from all bands
    estimation = estimator.estimate(per_band_y)
    return estimation, estimator


estimate_physical_parameters = {BandType.SINGLE.name: separate_bands_estimator,
                                BandType.MULTI.name: sim_bands_estimator}
