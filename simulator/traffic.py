"""Packet traffic: queues, retries, deadline drops, delay statistics and the per-AP drop urgency."""
import random

import numpy as np

from .config import (ACCESS_CATEGORIES, DATA_RATE_MBPS, FRAMES_PER_TXOP, MAX_RETRIES, PARAMS, TXOP_MS,
                     URGENCY_HOLD)
from .topology import AP_OF, APS, STATION_APP, STATION_POS
from .urgency import LEVELS, quantize

MAX_DEADLINE = max(ac["deadline"] for ac in ACCESS_CATEGORIES.values())


# Formula: CDF(d) = packets delivered within delay d / packets that arrived  (in %).
# Dropped packets are never delivered, so the curve ends at the delivered share instead of 100 %.
def delay_cdf(delay_hist, arrived):
    """delay_hist[k] = packets delivered k cycles after arriving. Returns (delay in ms, CDF in %)."""
    delay_ms = (np.arange(len(delay_hist)) + 1) * TXOP_MS
    return delay_ms, 100 * np.cumsum(delay_hist) / max(arrived, 1)


class Traffic:
    """Queues of every station. A queue is a list of batches [arrival cycle, packets, retries so far]."""

    def __init__(self, seed=None):
        self.arrivals = np.random.default_rng(seed)    # packet arrivals (same for every method)
        self.phy = random.Random(seed)                 # channel noise of the sharing link
        self.queue = {s: [] for s in STATION_POS}
        self.ap_value = dict.fromkeys(APS, 0.5)        # every AP starts at best-effort urgency
        self.last_drop = dict.fromkeys(APS, None)
        self.window_in = dict.fromkeys(APS, 0)
        self.window_drop = dict.fromkeys(APS, 0)
        self.cycle = 0
        self.record_from = 0                           # delay/drop statistics start at this cycle
        self.delay_hist = {ac: np.zeros(MAX_DEADLINE + 1) for ac in ACCESS_CATEGORIES}
        self.ac_arrived = dict.fromkeys(ACCESS_CATEGORIES, 0)
        self.ac_dropped = dict.fromkeys(ACCESS_CATEGORIES, 0)

    def ap_level(self, ap):
        return quantize(self.ap_value[ap])

    def queued(self, station):
        return sum(batch[1] for batch in self.queue[station])

    def delay_cdf(self, categories):
        """Delay CDF of the packets of the given access categories since statistics started."""
        hist = sum(self.delay_hist[ac] for ac in categories)
        return delay_cdf(hist, sum(self.ac_arrived[ac] for ac in categories))

    def start_cycle(self):
        """New packets arrive at every station; the count is Poisson with the access category's mean."""
        self.arrived = self.dropped = self.delivered = 0
        means = [ACCESS_CATEGORIES[STATION_APP[s]]["packets"] for s in STATION_POS]
        for station, n in zip(STATION_POS, self.arrivals.poisson(means)):
            if n:
                self.queue[station].append([self.cycle, int(n), 0])
                self.arrived += n
                self.window_in[AP_OF[station]] += n
                if self.cycle >= self.record_from:
                    self.ac_arrived[STATION_APP[station]] += n

    def _drop(self, station, n):
        self.dropped += n
        self.window_drop[AP_OF[station]] += n
        if self.cycle >= self.record_from:
            self.ac_dropped[STATION_APP[station]] += n

    # Formula: of the frames sent, round(sent * rate / R_max) are delivered; the rest are retried,
    # and dropped once they have failed MAX_RETRIES times. At most FRAMES_PER_TXOP frames are sent.
    def serve(self, station, rate):
        budget, success = FRAMES_PER_TXOP, rate / DATA_RATE_MBPS
        retry, rest = [], []
        for arrival, count, tries in self.queue[station]:
            sent = min(count, budget)
            budget -= sent
            if sent:
                ok = round(sent * success)
                self.delivered += ok
                if ok and self.cycle >= self.record_from:
                    self.delay_hist[STATION_APP[station]][self.cycle - arrival] += ok
                if sent - ok:
                    if tries + 1 >= MAX_RETRIES:
                        self._drop(station, sent - ok)
                    else:
                        retry.append([arrival, sent - ok, tries + 1])
            if count - sent:
                rest.append([arrival, count - sent, tries])
        self.queue[station] = retry + rest

    def end_cycle(self):
        """Drop packets past their deadline, then update the AP urgency every URGENCY_HOLD cycles."""
        for station, batches in self.queue.items():
            deadline = ACCESS_CATEGORIES[STATION_APP[station]]["deadline"]
            expired = sum(b[1] for b in batches if self.cycle - b[0] >= deadline)
            if expired:
                self._drop(station, expired)
                self.queue[station] = [b for b in batches if self.cycle - b[0] < deadline]
        self.cycle += 1
        if self.cycle % URGENCY_HOLD == 0:
            self.update_ap_urgency()

    # Formula (per AP, once per hold window):
    #   reference = max(TARGET_DROP, dropped_network / arrived_network)
    #   value     = clip(value + DROP_STEP * (dropped_AP / arrived_AP - reference), 0.25, 1.0)
    #   level     = value rounded to the nearest access-category level
    def update_ap_urgency(self):
        network = sum(self.window_drop.values()) / max(sum(self.window_in.values()), 1)
        reference = max(PARAMS["target_drop"], network)
        for ap in APS:
            if self.window_in[ap]:
                ratio = min(1.0, self.window_drop[ap] / self.window_in[ap])
                self.last_drop[ap] = ratio
                value = self.ap_value[ap] + PARAMS["drop_step"] * (ratio - reference)
                self.ap_value[ap] = min(LEVELS[-1], max(LEVELS[0], value))
            self.window_in[ap] = self.window_drop[ap] = 0
