from typing import List

from python_code.estimation import Estimation


def combine_estimations(estimations: List[Estimation]):
    if len(estimations) == 1:
        return estimations[0]
    raise ValueError("Not implemented the combination until now!")
