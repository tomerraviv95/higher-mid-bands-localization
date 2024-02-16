from python_code.estimation.parameters.angle import AngleEstimator
from python_code.estimation.parameters.angle_time import AngleTimeEstimator
from python_code.estimation.parameters.time import TimeEstimator
from python_code.utils.constants import EstimatorType

estimators = {EstimatorType.ANGLE: AngleEstimator, EstimatorType.TIME: TimeEstimator,
              EstimatorType.ANGLE_TIME: AngleTimeEstimator}

estimations_strings_dict = {'angle': EstimatorType.ANGLE,
                            'time': EstimatorType.TIME,
                            'both': EstimatorType.ANGLE_TIME}
