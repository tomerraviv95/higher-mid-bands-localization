from python_code.estimation.parameters_2d.angle import AngleEstimator2D
from python_code.estimation.parameters_2d.angle_time import AngleTimeEstimator2D
from python_code.estimation.parameters_2d.time import TimeEstimator2D
from python_code.estimation.parameters_3d.angle import AngleEstimator3D
from python_code.estimation.parameters_3d.angle_time import AngleTimeEstimator3D
from python_code.estimation.parameters_3d.time import TimeEstimator3D
from python_code.utils.constants import DimensionType, EstimatorType

estimators = {
    EstimatorType.ANGLE: {DimensionType.Three.name: AngleEstimator3D, DimensionType.Two.name: AngleEstimator2D},
    EstimatorType.TIME: {DimensionType.Three.name: TimeEstimator3D, DimensionType.Two.name: TimeEstimator2D},
    EstimatorType.ANGLE_TIME: {DimensionType.Three.name: AngleTimeEstimator3D,
                               DimensionType.Two.name: AngleTimeEstimator2D}}

estimations_strings_dict = {'angle': EstimatorType.ANGLE,
                            'time': EstimatorType.TIME,
                            'both': EstimatorType.ANGLE_TIME}
