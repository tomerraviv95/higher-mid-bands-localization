from typing import List, Tuple

from python_code.estimation import estimators
from python_code.utils.bands_manipulation import Band
from python_code.utils.constants import EstimatorType, Channel, Estimation, BandType


def single_band_estimator(per_band_y: List[Channel], bands: List[Band], estimator_type: EstimatorType) -> Tuple[
    Estimation, object]:
    # choose the estimator based on the desired type
    estimator = estimators[estimator_type]([bands[0]])
    # estimate delay / AOA / delay & AOA parameters for the current bs
    estimation = estimator.estimate(per_band_y[0])
    return estimation, estimator


def multiple_bands_estimator(per_band_y: List[Channel], bands: List[Band], estimator_type: EstimatorType) -> Tuple[
    Estimation, object]:
    # choose the estimator based on the desired type
    estimator = estimators[estimator_type](bands)
    # estimate delay / AOA / delay & AOA parameters for the current bs
    estimation = estimator.estimate(per_band_y)
    return estimation, estimator


estimate_physical_parameters = {BandType.SINGLE.name: single_band_estimator,
                                BandType.MULTI.name: multiple_bands_estimator}
