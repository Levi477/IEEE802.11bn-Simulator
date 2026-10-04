"""All tunable constants, each with the reason or assumption behind its value."""
import math

# ---------------------------------------------------------------- radio model
# 40 mW, a typical indoor AP transmit power.
TX_POWER_DBM = 16.02

# Thermal noise over a 20 MHz channel (-101 dBm) plus a ~7 dB receiver noise figure.
NOISE_FLOOR_DBM = -93.97

# 5 GHz band, the band used in the TGax enterprise path-loss model.
CARRIER_GHZ = 5.0

# TGax enterprise model: free-space loss at 1 m for the carrier frequency.
PATH_LOSS_1M_DB = 40.05 + 20 * math.log10(CARRIER_GHZ / 2.4)

# TGax model: beyond 10 m the loss grows with exponent 3.5 instead of 2.
BREAKPOINT_M = 10.0
LOSS_EXPONENT_BEYOND = 3.5

# Random SINR fluctuation (standard deviation), as in the H-MAB paper.
SINR_NOISE_DB = 2.0

# Assumption: one fixed MCS delivering 120 Mb/s. The delivered fraction grows linearly with
# SINR and is full at 35 dB; it never falls below 10 %, the share a robust MCS still gets through.
DATA_RATE_MBPS = 120.0
FULL_RATE_SINR_DB = 35.0
MIN_SUCCESS = 0.1

# Maximum PPDU (and TXOP) duration allowed in IEEE 802.11ax.
TXOP_MS = 5.484

# One MPDU carries an Ethernet-sized (MTU) packet.
FRAME_BYTES = 1500

# Frames that fit into one TXOP at the full data rate: ceil(rate * TXOP / frame size) = 55.
FRAMES_PER_TXOP = math.ceil(DATA_RATE_MBPS * 1e3 * TXOP_MS / (FRAME_BYTES * 8))

# -------------------------------------------------------------------- traffic
# The four IEEE 802.11 EDCA access categories. Their urgency levels (1.0, 0.75, 0.5, 0.25) are the only
# urgency values used anywhere in the simulator. Arrival rates and deadlines are assumptions:
# voice is light but very delay sensitive, video is heavy and moderately sensitive,
# best effort and background tolerate long delays. One cycle (TXOP) is 5.484 ms.
ACCESS_CATEGORIES = {
    "AC_VO": {"name": "Voice", "urgency": 1.0, "packets": 2.0, "deadline": 8},
    "AC_VI": {"name": "Video", "urgency": 0.75, "packets": 8.0, "deadline": 15},
    "AC_BE": {"name": "Best effort", "urgency": 0.5, "packets": 6.0, "deadline": 30},
    "AC_BK": {"name": "Background", "urgency": 0.25, "packets": 6.0, "deadline": 60},
}

# A failed frame is retried this many times before it is dropped. Real 802.11 allows more retries;
# 3 keeps queues short in this simple model.
MAX_RETRIES = 3

# -------------------------------------------------------------- learning
# Exponential moving average step: Q <- Q + ALPHA * (r - Q). 1/2 is the rule of the original script.
ALPHA = 0.5

# Tuned on separate random seeds by maximising the mean data rate over a 33 s run.
EPSILON = 0.1      # epsilon-greedy: chance of picking a random arm
UCB_C = 18.0       # UCB: size of the exploration bonus [Mb/s]
TS_SIGMA = 40.0    # Thompson sampling: spread of the sampled rate [Mb/s]

# Mb/s of data rate the controller gives up per unit of urgency. 60 lets one urgency level (0.25)
# outweigh a 15 Mb/s rate difference.
URGENCY_WEIGHT = 60.0

# ----------------------------------------------------- packet-drop urgency
# An AP's urgency value moves by DROP_STEP x (its drop ratio - reference drop ratio) every
# URGENCY_HOLD cycles. The reference is the network drop ratio, but never below TARGET_DROP.
DROP_STEP = 0.5
TARGET_DROP = 0.05
URGENCY_HOLD = 50   # 50 cycles = 0.27 s, long enough to measure a drop ratio

# Values the live sliders change while the simulation runs.
PARAMS = {"epsilon": EPSILON, "ucb_c": UCB_C, "ts_sigma": TS_SIGMA, "urgency_weight": URGENCY_WEIGHT,
          "drop_step": DROP_STEP, "target_drop": TARGET_DROP}

# ------------------------------------------------------------ outputs
GRAPH_RUNS = 20        # independent runs averaged in every graph
GRAPH_CYCLES = 6000    # 6000 cycles = 32.9 s of simulated time
CDF_ALGORITHM = "thompson"   # algorithm shown per access category in the delay CDF
RESULTS_DIR = "results"

START_SPEED = 30                 # live view: simulated cycles per second at start
GRAPH_WINDOW, SMOOTHING = 2000, 50   # live view: cycles on screen, moving-average length
