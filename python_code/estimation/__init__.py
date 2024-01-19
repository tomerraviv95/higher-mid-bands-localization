from collections import namedtuple

Estimation = namedtuple("Estimation", ["AOA", "TOA", "ZOA"], defaults=(None,) * 3)
