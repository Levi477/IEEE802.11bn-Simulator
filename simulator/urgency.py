"""The four urgency types: none, application (per station), packet drop (per AP) and combined."""
from .config import ACCESS_CATEGORIES
from .topology import AP_OF, STATION_APP, STATION_POS

MODES = {"none": "No urgency", "app": "Application urgency", "drop": "Packet-drop urgency",
         "combined": "Combined urgency"}

LEVELS = sorted(ac["urgency"] for ac in ACCESS_CATEGORIES.values())          # [0.25, 0.5, 0.75, 1.0]
LEVEL_NAME = {ac["urgency"]: name for name, ac in ACCESS_CATEGORIES.items()}  # 1.0 -> "AC_VO"
APP_URGENCY = {s: ACCESS_CATEGORIES[STATION_APP[s]]["urgency"] for s in STATION_POS}


def quantize(x):
    """Round x to the nearest of the four access-category levels (ties go to the more urgent level)."""
    return min(LEVELS, key=lambda level: (abs(level - x), -level))


# Formulas:
#   application:  u_s = level of the station's access category
#   packet drop:  u_s = level of the station's AP (see Traffic.update_ap_urgency)
#   combined:     u_s = quantize((application + packet drop) / 2)
def station_level(mode, traffic, station):
    if mode == "app":
        return APP_URGENCY[station]
    if mode == "drop":
        return traffic.ap_level(AP_OF[station])
    return quantize((APP_URGENCY[station] + traffic.ap_level(AP_OF[station])) / 2)


def urgency_map(mode, traffic):
    """Urgency of every station for this cycle, or None when urgency is off.

    A station with an empty queue has nothing to send, so its urgency is 0.
    """
    if mode == "none":
        return None
    return {s: station_level(mode, traffic, s) if traffic.queue[s] else 0.0 for s in STATION_POS}
