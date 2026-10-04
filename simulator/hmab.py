"""Hierarchical multi-armed bandit (H-MAB): level 1 picks the partner AP, level 2 picks its station."""
import random

from .channel import data_rate, distance
from .config import PARAMS
from .topology import APS, STATIONS


class HierarchicalMAB:
    """One level-1 agent per sharing pair (AP, station) and one level-2 agent per (sharing pair, partner AP),
    as in the H-MAB paper, where the sharing pair is called P0."""

    def __init__(self, bandit_class):
        self.bandit_class = bandit_class
        self.level_1 = {}
        self.level_2 = {}

    def _agent(self, table, key, arms):
        if key not in table:
            table[key] = self.bandit_class(arms)
        return table[key]

    # Formula (urgency bonus, zero when urgency is off):
    #   level 1:  bonus(AP)      = w * max urgency of the AP's stations
    #   level 2:  bonus(station) = w * urgency of the station
    def play(self, anchor_ap, anchor_sta, urgency):
        """Run one TXOP and learn from it. Returns (sharing station, partner AP, partner station, rate)."""
        weight = PARAMS["urgency_weight"] if urgency else 0.0
        if urgency:
            anchor_sta = pick_sharing_station(anchor_ap, anchor_sta, urgency)
        pair = (anchor_ap, anchor_sta)

        ap_agent = self._agent(self.level_1, pair, [ap for ap in APS if ap != anchor_ap])
        ap = ap_agent.choose({a: weight * max(urgency[s] for s in STATIONS[a]) if urgency else 0.0
                              for a in ap_agent.q})
        sta_agent = self._agent(self.level_2, pair + (ap,), list(STATIONS[ap]))
        sta = sta_agent.choose({s: weight * urgency[s] if urgency else 0.0 for s in sta_agent.q})

        position = STATIONS[ap][sta]
        rate = data_rate(distance(APS[ap], position), distance(APS[anchor_ap], position))
        ap_agent.update(ap, rate)
        sta_agent.update(sta, rate)
        return anchor_sta, ap, sta, rate


# Formula (EDCA-like contention inside the sharing AP): P(station) = u_s / sum of u over backlogged stations.
# Higher categories win more often but, as in real EDCA, not always, so lower categories are not starved.
def pick_sharing_station(anchor_ap, default, urgency):
    backlogged = [s for s in STATIONS[anchor_ap] if urgency[s] > 0]
    if not backlogged:
        return default
    return random.choices(backlogged, weights=[urgency[s] for s in backlogged])[0]
