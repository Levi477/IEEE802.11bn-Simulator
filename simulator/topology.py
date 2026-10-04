"""Network layout: where the APs and stations are and which application each station runs.

Edit this file to try a different deployment. Positions are in metres.
"""

# Four APs on the corners of a 20 m square, the test layout of the H-MAB paper (Wojnar et al., 2025).
APS = {
    "AP_1": (0, 0),
    "AP_2": (20, 0),
    "AP_3": (0, 20),
    "AP_4": (20, 20),
}

# Stations sit on the outer diagonal of their AP, 2.83 m (near) and 7.07 m (far) away.
# AP_4 has one extra station so that the APs are not all identical.
STATIONS = {
    "AP_1": {"STA_1A": (-2, -2), "STA_1B": (-5, -5)},
    "AP_2": {"STA_2A": (22, -2), "STA_2B": (25, -5)},
    "AP_3": {"STA_3A": (-2, 22), "STA_3B": (-5, 25)},
    "AP_4": {"STA_4A": (22, 22), "STA_4B": (25, 25), "STA_4C": (27, 22)},
}

# Application (EDCA access category) of every station. The mix puts every category
# on at least two APs, so no single AP holds all the urgent traffic.
STATION_APP = {
    "STA_1A": "AC_VO", "STA_1B": "AC_BE",
    "STA_2A": "AC_VI", "STA_2B": "AC_BK",
    "STA_3A": "AC_BE", "STA_3B": "AC_VO",
    "STA_4A": "AC_BK", "STA_4B": "AC_VI", "STA_4C": "AC_BE",
}

STATION_POS = {name: pos for stations in STATIONS.values() for name, pos in stations.items()}
AP_OF = {station: ap for ap, stations in STATIONS.items() for station in stations}


def check_topology():
    """Stop early with a clear message if the layout above is inconsistent."""
    if len(APS) < 2:
        raise ValueError("APS needs at least two access points")
    if set(STATIONS) != set(APS):
        raise ValueError(f"STATIONS and APS must list the same APs (mismatch: {set(STATIONS) ^ set(APS)})")
    names = [s for stations in STATIONS.values() for s in stations]
    if len(names) != len(set(names)):
        raise ValueError("station names must be unique across all APs")
    for ap, stations in STATIONS.items():
        if not stations:
            raise ValueError(f"{ap} has no stations")
    missing = set(STATION_POS) - set(STATION_APP)
    if missing:
        raise ValueError(f"STATION_APP has no application for {sorted(missing)}")
