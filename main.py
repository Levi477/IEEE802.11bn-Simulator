import math
import random
import matplotlib.pyplot as plt

TX_POWER_DBM = 16.02         # Transmit power
NOISE_FLOOR_DBM = -93.97     # Ambient noise
TXOP_MS = 5.484              # Time window for transmission in ms
FRAME_BYTES = 1500           # Size of one A-MPDU sub-frame
DATA_RATE_MBPS = 120.0       # Theoretical max data rate
WALLS_IN_PATH = 1            # Assume 1 wall between interfering nodes

access_points = {
    "AP_1": {"x": 0.0, "y": 0.0},
    "AP_2": {"x": 15.0, "y": 0.0},
    "AP_3": {"x": 0.0, "y": 15.0}
}

stations = {
    "AP_2": [
        {"name": "STA_2A_Close", "x": 17.0, "y": 0.0},
        {"name": "STA_2B_Far", "x": 30.0, "y": 0.0}
    ],
    "AP_3": [
        {"name": "STA_3A_Close", "x": 0.0, "y": 17.0},
        {"name": "STA_3B_Far", "x": 0.0, "y": 30.0}
    ]
}

def get_distance(node1, node2):
    return math.hypot(node2["x"] - node1["x"], node2["y"] - node1["y"])

def calc_path_loss(distance, walls):
    loss = 40.05 + (20 * math.log10(5.0 / 2.4)) + (20 * math.log10(max(1.0, min(distance, 10.0)))) + (7 * walls)
    if distance > 10.0:
        loss += 35 * math.log10(distance / 10.0)
    return loss

def calc_effective_mbps(signal_distance, interference_distance):
    loss = calc_path_loss(signal_distance, walls=0)
    interference_dbm = TX_POWER_DBM - calc_path_loss(interference_distance, WALLS_IN_PATH)

    interference_mw = 10 ** (interference_dbm / 10.0)
    noise_mw = 10 ** (NOISE_FLOOR_DBM / 10.0)
    total_interference_db = 10 * math.log10(interference_mw + noise_mw)
    sinr = TX_POWER_DBM - (loss + total_interference_db) + random.gauss(0, 2)
    success_prob = max(0.1, min(1.0, sinr / 35.0))
    bytes_per_ms = (DATA_RATE_MBPS * 1000) / 8
    max_frames = math.ceil((bytes_per_ms * TXOP_MS) / FRAME_BYTES)

    delivered_bytes = max_frames * FRAME_BYTES * success_prob
    return (delivered_bytes * 8) / (TXOP_MS * 1000)

level_1_memory = {"AP_2": 0.0, "AP_3": 0.0}
level_2_memory = {
    "AP_2": {"STA_2A_Close": 0.0, "STA_2B_Far": 0.0},
    "AP_3": {"STA_3A_Close": 0.0, "STA_3B_Far": 0.0}
}
explore_chance = 0.1
anchor_ap = access_points["AP_1"]

time_log = []
mbps_log = []
TOTAL_CYCLES = 50

for cycle in range(1, TOTAL_CYCLES + 1):

    # LEVEL 1
    if random.random() < explore_chance:
        chosen_ap_name = random.choice(list(level_1_memory.keys()))
    else:
        chosen_ap_name = max(level_1_memory, key=level_1_memory.get)

    active_ap = access_points[chosen_ap_name]

    # LEVEL 2
    station_options = level_2_memory[chosen_ap_name]
    if random.random() < explore_chance:
        chosen_sta_name = random.choice(list(station_options.keys()))
    else:
        chosen_sta_name = max(station_options, key=station_options.get)

    active_sta = next(s for s in stations[chosen_ap_name] if s["name"] == chosen_sta_name)

    sig_dist = get_distance(active_ap, active_sta)
    int_dist = get_distance(anchor_ap, active_sta)
    reward_mbps = calc_effective_mbps(sig_dist, int_dist)

    level_1_memory[chosen_ap_name] = (level_1_memory[chosen_ap_name] + reward_mbps) / 2.0
    level_2_memory[chosen_ap_name][chosen_sta_name] = (level_2_memory[chosen_ap_name][chosen_sta_name] + reward_mbps) / 2.0

    time_s = (cycle * TXOP_MS) / 1000.0
    time_log.append(time_s)
    mbps_log.append(reward_mbps)

plt.figure(figsize=(8, 5))

plt.plot(time_log, mbps_log, color='#004488', marker='o', markersize=3, linewidth=1, label="Effective Data Rate")

plt.title("Effective Data Rate over Time")
plt.xlabel("Time [s]")
plt.ylabel("Effective data rate [Mb/s]")
plt.grid(True, linestyle=":", alpha=0.7)
plt.legend()
plt.tight_layout()

plt.savefig("data_rate_vs_time.png")
