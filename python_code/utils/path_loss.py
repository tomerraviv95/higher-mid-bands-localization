import math

from python_code import conf


def compute_path_loss(toa):
    loss_db = 20 * math.log10(toa) + 20 * math.log10(conf.fc) + 20 * math.log10(4 * math.pi)
    return 10 ** (loss_db / 20)
