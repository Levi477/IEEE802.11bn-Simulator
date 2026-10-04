"""A method = bandit algorithm + urgency type. This module runs one method cycle by cycle."""
import random

import numpy as np

from .bandits import ALGORITHMS
from .channel import data_rate, distance
from .config import FRAME_BYTES, PARAMS, TXOP_MS
from .hmab import HierarchicalMAB
from .topology import APS, STATION_POS, STATIONS
from .traffic import Traffic
from .urgency import MODES, urgency_map

# Random is only a no-urgency baseline; every learning algorithm is run with all four urgency types.
METHOD_KEYS = ["random|none"] + [f"{alg}|{mode}" for mode in MODES for alg in ("egreedy", "ucb", "thompson")]
METRICS = ("rate", "utility", "u", "arrived", "dropped", "throughput", "mean_u")


def method_label(key, short=False):
    algorithm, mode = key.split("|")
    name = ALGORITHMS[algorithm][0]
    return name if short or mode == "none" else f"{name} + {MODES[mode].split()[0].lower()} urgency"


def methods_of(mode):
    return [k for k in METHOD_KEYS if k.endswith("|" + mode)]


def draw_cycle(rng):
    """Which AP wins the TXOP and which station its head-of-line frame is for (both uniform, as with DCF)."""
    ap = rng.choice(list(APS))
    return ap, rng.choice(list(STATIONS[ap]))


class Method:
    def __init__(self, key, seed=None):
        self.key = key
        self.algorithm, self.mode = key.split("|")
        self.label = method_label(key)
        self.mab = HierarchicalMAB(ALGORITHMS[self.algorithm][1])
        self.traffic = Traffic(seed)

    def urgency(self):
        return urgency_map(self.mode, self.traffic)

    # Formulas:
    #   utility    = r + w * u(partner station)
    #   throughput = delivered frames * frame size / TXOP duration
    def run_cycle(self, anchor_ap, anchor_sta):
        traffic = self.traffic
        traffic.start_cycle()
        urgency = self.urgency()
        a_sta, ap, sta, rate = self.mab.play(anchor_ap, anchor_sta, urgency)
        served_u = urgency[sta] if urgency else 0.0
        a_pos = STATION_POS[a_sta]
        traffic.serve(a_sta, data_rate(distance(APS[anchor_ap], a_pos), distance(APS[ap], a_pos), traffic.phy))
        traffic.serve(sta, rate)
        traffic.end_cycle()
        return {"anchor_ap": anchor_ap, "anchor_sta": a_sta, "ap": ap, "sta": sta, "rate": rate,
                "u": served_u, "utility": rate + PARAMS["urgency_weight"] * served_u,
                "drop": 100 * traffic.dropped / max(traffic.arrived, 1),
                "throughput": traffic.delivered * FRAME_BYTES * 8 / (TXOP_MS * 1e3),
                "arrived": traffic.arrived, "dropped": traffic.dropped,
                "mean_u": np.mean([traffic.ap_level(a) for a in APS])}


def simulate(key, seed, cycles):
    """One run of one method. Every method sees the same TXOP winners and packet arrivals for a given seed."""
    rng = random.Random(seed)
    scenario = [draw_cycle(rng) for _ in range(cycles)]
    random.seed(seed)
    method = Method(key, seed)
    method.traffic.record_from = cycles // 2
    out = {k: np.zeros(cycles) for k in METRICS}
    for t, (anchor_ap, anchor_sta) in enumerate(scenario):
        info = method.run_cycle(anchor_ap, anchor_sta)
        for k in METRICS:
            out[k][t] = info[k]
    out["delay_hist"] = method.traffic.delay_hist
    out["ac_arrived"], out["ac_dropped"] = method.traffic.ac_arrived, method.traffic.ac_dropped
    return out


def smoothed(series, key, window):
    """Moving average. The drop percentage is a ratio of sums, not an average of per-cycle percentages."""
    ones = np.ones(window)
    if key == "drop":
        dropped = np.convolve(series["dropped"], ones, "valid")
        return 100 * dropped / np.maximum(np.convolve(series["arrived"], ones, "valid"), 1)
    return np.convolve(series[key], ones, "valid") / window
