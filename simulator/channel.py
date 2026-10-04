"""Radio channel: path loss, SINR and the resulting data rate of a link."""
import math
import random

from .config import (BREAKPOINT_M, DATA_RATE_MBPS, FULL_RATE_SINR_DB, LOSS_EXPONENT_BEYOND,
                     MIN_SUCCESS, NOISE_FLOOR_DBM, PATH_LOSS_1M_DB, SINR_NOISE_DB, TX_POWER_DBM)


def distance(a, b):
    """Euclidean distance between two (x, y) points in metres."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


# Formula (TGax enterprise model, no walls):
#   PL(d) = PL(1 m) + 20 log10(min(max(d, 1), Bp)) + [d > Bp] * 10 n log10(d / Bp)
def path_loss_db(d):
    loss = PATH_LOSS_1M_DB + 20 * math.log10(max(1.0, min(d, BREAKPOINT_M)))
    if d > BREAKPOINT_M:
        loss += 10 * LOSS_EXPONENT_BEYOND * math.log10(d / BREAKPOINT_M)
    return loss


# Formulas:
#   SINR = P_signal - 10 log10(10^(P_interference / 10) + 10^(N0 / 10)) + noise,  noise ~ N(0, sigma^2)
#   rate = R_max * min(1, max(eta_min, SINR / SINR_full))
def data_rate(signal_m, interference_m, rng=random):
    """Rate [Mb/s] of a link whose transmitter is signal_m away while one interferer is interference_m away."""
    signal_dbm = TX_POWER_DBM - path_loss_db(signal_m)
    interference_dbm = TX_POWER_DBM - path_loss_db(interference_m)
    noise_dbm = 10 * math.log10(10 ** (interference_dbm / 10) + 10 ** (NOISE_FLOOR_DBM / 10))
    sinr = signal_dbm - noise_dbm + rng.gauss(0, SINR_NOISE_DB)
    return DATA_RATE_MBPS * max(MIN_SUCCESS, min(1.0, sinr / FULL_RATE_SINR_DB))
