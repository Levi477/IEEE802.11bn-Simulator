import math
import random
import sys
import time

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.widgets import Button, RadioButtons, Slider


# Topology, positions in metres. Every AP needs at least one station, station names must be unique.
APS = {
    "AP_1": (0, 0),
    "AP_2": (20, 0),
    "AP_3": (0, 20),
    "AP_4": (20, 20),
}
STATIONS = {
    "AP_1": {"STA_1A": (-2, -2), "STA_1B": (-5, -5)},
    "AP_2": {"STA_2A": (22, -2), "STA_2B": (25, -5)},
    "AP_3": {"STA_3A": (-2, 22), "STA_3B": (-5, 25)},
    "AP_4": {"STA_4A": (22, 22), "STA_4B": (25, 25), "STA_4C": (27, 22)},
}

# Radio model
TX_POWER_DBM = 16.02
NOISE_FLOOR_DBM = -93.97
DATA_RATE_MBPS = 120.0                            # best possible data rate of one link
PATH_LOSS_1M_DB = 40.05 + 20 * math.log10(5.0 / 2.4)   # loss at 1 m (5 GHz)
BREAKPOINT_M = 10.0                               # beyond this distance the loss grows faster
LOSS_EXPONENT_BEYOND = 3.5                        # path-loss exponent beyond the breakpoint
SINR_NOISE_DB = 2.0                               # random noise added to the SINR (std dev)
FULL_RATE_SINR_DB = 35.0                          # SINR at which a link reaches its full rate
MIN_SUCCESS = 0.1                                 # a link never drops below 10% of its full rate
TXOP_MS = 5.484
FRAME_BYTES = 1500
FRAMES_PER_TXOP = math.ceil(DATA_RATE_MBPS * 1e3 * TXOP_MS / (FRAME_BYTES * 8))

# Packet traffic
PACKETS_PER_CYCLE = 6.0
PACKET_DEADLINE = 20
MAX_RETRIES = 3

# Algorithm settings
EPSILON = 0.1           # ε-greedy and Priority: chance of trying a random arm
UCB_C = 18.0            # UCB: size of the exploration bonus [Mb/s]
URGENCY_WEIGHT = 60.0   # Priority: Mb/s of bias per unit of urgency
START_SPEED = 30        # simulated cycles per second

# Dynamic urgency
INITIAL_URGENCY = 0.5
DROP_STEP = 0.5
TARGET_DROP = 0.05
URGENCY_HOLD = 50

# Graphs made by "python live.py --graphs"
GRAPH_RUNS = 20
GRAPH_CYCLES = 6000


METHODS = {"random": "Random", "egreedy": "ε-greedy", "ucb": "UCB",
           "priority": "ε-greedy + urgency", "priority_ucb": "UCB + urgency",
           "dyn_egreedy": "ε-greedy + dynamic urgency", "dyn_ucb": "UCB + dynamic urgency"}
GROUPS = {"No urgency": ("random", "egreedy", "ucb"),
          "Random urgency": ("priority", "priority_ucb"),
          "Dynamic urgency": ("dyn_egreedy", "dyn_ucb")}
GROUP_OF = {m: g for g, ms in GROUPS.items() for m in ms}
URGENT = {"priority", "priority_ucb", "dyn_egreedy", "dyn_ucb"}
DYNAMIC = {"dyn_egreedy", "dyn_ucb"}
UCB_LIKE = {"ucb", "priority_ucb", "dyn_ucb"}
METHOD_COLORS = {"random": "#8a8a86", "egreedy": "#2a78d6", "ucb": "#eb6834",
                 "priority": "#2a78d6", "priority_ucb": "#eb6834",
                 "dyn_egreedy": "#2a78d6", "dyn_ucb": "#eb6834"}
METRIC = {"No urgency": ("rate", "Data rate [Mb/s]"),
          "Random urgency": ("utility", "Utility r + w·u [Mb/s]"),
          "Dynamic urgency": ("drop", "Dropped packets [%]")}
ANCHOR, PARTNER, INTERFERENCE, IDLE = "#4a3aa7", "#008300", "#e34948", "#c9c8c0"
GRAPH_WINDOW, SMOOTHING = 2000, 50   # the graph shows the last 2000 cycles, 50-cycle moving average

PARAMS = {"epsilon": EPSILON, "ucb_c": UCB_C, "urgency_weight": URGENCY_WEIGHT,
          "drop_step": DROP_STEP, "target_drop": TARGET_DROP}
STATION_POS = {name: xy for stas in STATIONS.values() for name, xy in stas.items()}


def check_topology():
    if len(APS) < 2:
        raise ValueError("APS needs at least two access points")
    if set(STATIONS) != set(APS):
        raise ValueError(f"STATIONS and APS must list the same APs (mismatch: {set(STATIONS) ^ set(APS)})")
    names = [s for stas in STATIONS.values() for s in stas]
    if len(names) != len(set(names)):
        raise ValueError("station names must be unique across all APs")
    for ap, stas in STATIONS.items():
        if not stas:
            raise ValueError(f"{ap} has no stations")


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def path_loss(d):
    loss = PATH_LOSS_1M_DB + 20 * math.log10(max(1.0, min(d, BREAKPOINT_M)))
    if d > BREAKPOINT_M:
        loss += 10 * LOSS_EXPONENT_BEYOND * math.log10(d / BREAKPOINT_M)
    return loss


def data_rate(signal_dist, interference_dist, rng=random):
    signal_dbm = TX_POWER_DBM - path_loss(signal_dist)
    interference_dbm = TX_POWER_DBM - path_loss(interference_dist)
    noise_dbm = 10 * math.log10(10 ** (interference_dbm / 10) + 10 ** (NOISE_FLOOR_DBM / 10))
    sinr = signal_dbm - noise_dbm + rng.gauss(0, SINR_NOISE_DB)
    return DATA_RATE_MBPS * max(MIN_SUCCESS, min(1.0, sinr / FULL_RATE_SINR_DB))


def draw_cycle(rng):
    ap = rng.choice(list(APS))
    return ap, rng.choice(list(STATIONS[ap])), {s: rng.random() for s in STATION_POS}


def pick(method, arms, bonus):
    if method == "random":
        return random.choice(list(arms))
    if method in UCB_LIKE:
        untried = [a for a in arms if arms[a][1] == 0]
        if untried:
            return random.choice(untried)
        total = sum(n for _, n in arms.values())
        return max(arms, key=lambda a: arms[a][0] + bonus[a]
                   + PARAMS["ucb_c"] * math.sqrt(math.log(total) / arms[a][1]))
    if random.random() < PARAMS["epsilon"]:
        return random.choice(list(arms))
    return max(arms, key=lambda a: (arms[a][0] + bonus[a], random.random()))


def update(arm, reward):
    arm[0] = (arm[0] + reward) / 2
    arm[1] += 1


def step(method, memory, anchor_ap, anchor_sta, urgency):
    level_1, level_2 = memory
    weight = PARAMS["urgency_weight"] if method in URGENT else 0.0
    if method in URGENT:
        anchor_sta = max(STATIONS[anchor_ap], key=urgency.get)
    pair = (anchor_ap, anchor_sta)

    arms = level_1.setdefault(pair, {ap: [0.0, 0] for ap in APS if ap != anchor_ap})
    ap = pick(method, arms, {a: weight * max(urgency[s] for s in STATIONS[a]) for a in arms})
    arms_2 = level_2.setdefault(pair + (ap,), {s: [0.0, 0] for s in STATIONS[ap]})
    sta = pick(method, arms_2, {s: weight * urgency[s] for s in arms_2})

    position = STATIONS[ap][sta]
    rate = data_rate(dist(APS[ap], position), dist(APS[anchor_ap], position))
    update(arms[ap], rate)
    update(arms_2[sta], rate)
    return anchor_sta, ap, sta, rate


class Traffic:
    def __init__(self, seed):
        self.arrivals = np.random.default_rng(seed)
        self.phy = random.Random(seed)
        self.queue = {s: [] for s in STATION_POS}
        self.u = dict.fromkeys(STATION_POS, INITIAL_URGENCY)
        self.last_drop = dict.fromkeys(STATION_POS, None)
        self.window_in = dict.fromkeys(STATION_POS, 0)
        self.window_drop = dict.fromkeys(STATION_POS, 0)
        self.cycle = 0

    def start_cycle(self):
        self.arrived = self.dropped = self.delivered = 0
        for s, n in zip(STATION_POS, self.arrivals.poisson(PACKETS_PER_CYCLE, len(STATION_POS))):
            if n:
                self.queue[s].append([self.cycle, int(n), 0])
                self.arrived += n
                self.window_in[s] += n

    def lose(self, sta, n):
        self.dropped += n
        self.window_drop[sta] += n

    def serve(self, sta, rate):
        budget, success = FRAMES_PER_TXOP, rate / DATA_RATE_MBPS
        retry, rest = [], []
        for batch in self.queue[sta]:
            sent = min(batch[1], budget)
            budget -= sent
            if sent:
                ok = round(sent * success)
                self.delivered += ok
                if sent - ok:
                    if batch[2] + 1 >= MAX_RETRIES:
                        self.lose(sta, sent - ok)
                    else:
                        retry.append([batch[0], sent - ok, batch[2] + 1])
            if batch[1] - sent:
                rest.append([batch[0], batch[1] - sent, batch[2]])
        self.queue[sta] = retry + rest

    def end_cycle(self):
        for s, q in self.queue.items():
            expired = sum(b[1] for b in q if self.cycle - b[0] >= PACKET_DEADLINE)
            if expired:
                self.lose(s, expired)
                self.queue[s] = [b for b in q if self.cycle - b[0] < PACKET_DEADLINE]
        self.cycle += 1
        if self.cycle % URGENCY_HOLD:
            return
        for s in STATION_POS:
            if self.window_in[s]:
                ratio = min(1.0, self.window_drop[s] / self.window_in[s])
                self.last_drop[s] = ratio
                self.u[s] = min(1.0, max(0.0, self.u[s] + PARAMS["drop_step"] * (ratio - PARAMS["target_drop"])))
            self.window_in[s] = self.window_drop[s] = 0


HISTORY = ("rate", "utility", "u", "arrived", "dropped", "throughput")


def smoothed(history, key):
    window = np.ones(SMOOTHING)
    if key == "drop":
        dropped = np.convolve(history["dropped"], window, "valid")
        return 100 * dropped / np.maximum(np.convolve(history["arrived"], window, "valid"), 1)
    return np.convolve(history[key], window, "valid") / SMOOTHING


def new_state(method, seed=None):
    return {"memory": ({}, {}), "traffic": Traffic(seed)}


def run_cycle(method, state, anchor_ap, anchor_sta, random_urgency):
    traffic = state["traffic"]
    traffic.start_cycle()
    urgency = traffic.u if method in DYNAMIC else random_urgency
    a_sta, ap, sta, rate = step(method, state["memory"], anchor_ap, anchor_sta, urgency)
    served_u = urgency[sta] if method in URGENT else 0.0
    a_pos = STATION_POS[a_sta]
    traffic.serve(a_sta, data_rate(dist(APS[anchor_ap], a_pos), dist(APS[ap], a_pos), traffic.phy))
    traffic.serve(sta, rate)
    traffic.end_cycle()
    return {"anchor_ap": anchor_ap, "anchor_sta": a_sta, "ap": ap, "sta": sta, "rate": rate,
            "u": served_u, "utility": rate + PARAMS["urgency_weight"] * served_u,
            "drop": 100 * traffic.dropped / max(traffic.arrived, 1),
            "throughput": traffic.delivered * FRAME_BYTES * 8 / (TXOP_MS * 1e3),
            "arrived": traffic.arrived, "dropped": traffic.dropped}


class Live:
    def __init__(self):
        self.selected, self.paused = "egreedy", False
        self.build_figure()
        self.reset()
        self.animation = FuncAnimation(self.fig, self.tick, interval=30, cache_frame_data=False)

    def reset(self, _=None):
        self.rng = random.Random()
        seed = random.randrange(1 << 30)
        self.states = {m: new_state(m, seed) for m in METHODS}
        self.history = {m: {k: [] for k in HISTORY} for m in METHODS}
        self.counts = {m: dict.fromkeys(STATION_POS, 0) for m in METHODS}
        self.last = {}
        self.cycle, self.progress, self.clock = 0, 0.0, time.perf_counter()
        self.advance()

    def advance(self):
        anchor_ap, anchor_sta, self.random_urgency = draw_cycle(self.rng)
        for m in METHODS:
            info = run_cycle(m, self.states[m], anchor_ap, anchor_sta, self.random_urgency)
            self.last[m] = info
            for key in HISTORY:
                self.history[m][key].append(info[key])
            self.counts[m][info["sta"]] += 1
        self.cycle += 1

    def tick(self, _):
        now = time.perf_counter()
        dt, self.clock = min(now - self.clock, 0.1), now
        if not self.paused:
            self.progress += self.speed.val * dt
            while self.progress >= 1:
                self.progress -= 1
                self.advance()
        self.draw()

    def urgency_shown(self):
        if self.selected in DYNAMIC:
            return self.states[self.selected]["traffic"].u
        if self.selected in URGENT:
            return self.random_urgency
        return None

    def draw(self):
        info = self.last[self.selected]
        anchor_ap, anchor_sta, ap, sta, rate = info["anchor_ap"], info["anchor_sta"], info["ap"], info["sta"], info["rate"]
        urgency = self.urgency_shown()
        traffic = self.states[self.selected]["traffic"]
        self.ap_dots.set_facecolor([ANCHOR if n == anchor_ap else PARTNER if n == ap else IDLE for n in APS])
        self.sta_dots.set_sizes([40 + 260 * urgency[s] if urgency else 110 for s in STATION_POS])
        self.sta_dots.set_facecolor([ANCHOR if s == anchor_sta else PARTNER if s == sta else IDLE
                                     for s in STATION_POS])
        for s, text in self.sta_text.items():
            queued = sum(b[1] for b in traffic.queue[s])
            label = f"{s}\nq={queued}" if urgency is None else f"{s}\nq={queued} u={urgency[s]:.2f}"
            if traffic.last_drop[s]:
                label += f"\ndrop {100 * traffic.last_drop[s]:.0f}%"
            text.set_text(label)

        def link(line, ap_name, sta_name, width):
            line.set_data(*zip(APS[ap_name], STATION_POS[sta_name]))
            line.set_linewidth(width)

        link(self.anchor_link, anchor_ap, anchor_sta, 3)
        link(self.partner_link, ap, sta, max(1, 10 * rate / DATA_RATE_MBPS))
        link(self.interference_link, anchor_ap, sta, 2)
        for ring, name, color in ((self.anchor_ring, anchor_ap, ANCHOR), (self.partner_ring, ap, PARTNER)):
            ring.set_center(APS[name])
            ring.set_radius(self.span * (0.04 + 0.30 * self.progress))
            ring.set_edgecolor(color)
            ring.set_alpha(1 - self.progress)
        self.ax_net.set_title(f"{METHODS[self.selected]}  ·  cycle {self.cycle:,}\n"
                              f"anchor {anchor_ap} → {anchor_sta}     partner {ap} → {sta}     {rate:.0f} Mb/s"
                              f"     {info['drop']:.0f}% dropped", fontsize=12)

        group = GROUP_OF[self.selected]
        key, ylabel = METRIC[group]
        n, curves = self.cycle, []
        for m, line in self.lines.items():
            line.set_visible(m in GROUPS[group])
            h = {k: np.array(v[-GRAPH_WINDOW:]) for k, v in self.history[m].items()}
            if m in GROUPS[group] and len(h["rate"]) >= SMOOTHING:
                smooth = smoothed(h, key)
                line.set_data(n - len(smooth) + 1 + np.arange(len(smooth)), smooth)
                curves.append(smooth)
        if group != self.shown_group:
            self.shown_group = group
            self.ax_rate.set_ylabel(ylabel)
            self.ax_rate.set_title(f"{group}: {SMOOTHING}-cycle moving average", loc="left", fontweight="bold")
            self.ax_rate.legend(handles=[self.lines[m] for m in GROUPS[group]], loc="upper center",
                                ncol=3, fontsize=9, frameon=False)
        self.ax_rate.set_xlim(max(0, n - GRAPH_WINDOW), max(n, 300))
        if curves:
            low, high = min(c.min() for c in curves), max(c.max() for c in curves)
            pad = max(2.0, 0.08 * (high - low))
            self.ax_rate.set_ylim(low - pad, high + 0.25 * (high - low) + pad)

        share = [100 * self.counts[self.selected][s] / n for s in STATION_POS]
        for bar, s, value in zip(self.bars, STATION_POS, share):
            bar.set_width(value)
            bar.set_color(PARTNER if s == sta else "#9db8d9")
        self.ax_bars.set_xlim(0, max(20, 1.15 * max(share)))

    def build_figure(self):
        plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
        self.fig = plt.figure(figsize=(15, 9))
        self.fig.canvas.manager.set_window_title("Multi-AP bandit, live")

        xs = [p[0] for p in list(APS.values()) + list(STATION_POS.values())]
        ys = [p[1] for p in list(APS.values()) + list(STATION_POS.values())]
        self.span = max(max(xs) - min(xs), max(ys) - min(ys), 10)
        pad = 0.18 * self.span
        ax = self.ax_net = self.fig.add_axes([0.02, 0.36, 0.50, 0.58])
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)
        ax.set_aspect("equal")
        ax.axis("off")
        self.anchor_ring = ax.add_patch(Circle((0, 0), 1, fill=False, lw=2.5))
        self.partner_ring = ax.add_patch(Circle((0, 0), 1, fill=False, lw=2.5))
        self.interference_link, = ax.plot([], [], color=INTERFERENCE, ls="--", zorder=1)
        self.anchor_link, = ax.plot([], [], color=ANCHOR, zorder=2)
        self.partner_link, = ax.plot([], [], color=PARTNER, zorder=2)
        self.ap_dots = ax.scatter(*zip(*APS.values()), marker="^", s=380, edgecolor="#0b0b0b", zorder=4)
        self.sta_dots = ax.scatter(*zip(*STATION_POS.values()), s=100, edgecolor="#0b0b0b", zorder=4)
        for name, pos in APS.items():
            ax.annotate(name, pos, xytext=(0, -16), textcoords="offset points", ha="center", va="top",
                        fontweight="bold", fontsize=12)
        halo = [pe.withStroke(linewidth=3, foreground="white")]
        self.sta_text = {}
        for ap, stations in STATIONS.items():
            for name, (x, y) in stations.items():
                right = x >= APS[ap][0]
                self.sta_text[name] = ax.annotate(
                    "", (x, y), xytext=(6 if right else -6, 7), textcoords="offset points", zorder=6,
                    ha="left" if right else "right", va="bottom", fontsize=8.5, color="#52514e",
                    linespacing=1.1, path_effects=halo)
        ax.legend(handles=[
            Line2D([], [], color=ANCHOR, lw=3, label="anchor: won the TXOP"),
            Line2D([], [], color=PARTNER, lw=3, label="partner: picked by the bandit (thicker = faster)"),
            Line2D([], [], color=INTERFERENCE, lw=2, ls="--", label="interference from the anchor"),
            Line2D([], [], marker="o", ls="", color=IDLE, mec="#0b0b0b", label="dot size = urgency u")],
            loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=9.5, frameon=False)

        ax = self.ax_rate = self.fig.add_axes([0.60, 0.62, 0.37, 0.31])
        self.lines = {m: ax.plot([], [], color=METHOD_COLORS[m], lw=2.2, ls="--" if m == "random" else "-",
                                 label=METHODS[m])[0] for m in METHODS}
        ax.set_xlabel("Cycle")
        ax.grid(alpha=0.3)
        self.shown_group = None

        ax = self.ax_bars = self.fig.add_axes([0.66, 0.29, 0.31, 0.25])
        self.bars = ax.barh(range(len(STATION_POS)), np.zeros(len(STATION_POS)), color="#9db8d9")
        ax.set_yticks(range(len(STATION_POS)), list(STATION_POS), fontsize=max(5, min(9, 80 / len(STATION_POS))))
        ax.invert_yaxis()
        ax.set_xlabel("% of cycles chosen as partner (animated method)")

        radio_ax = self.fig.add_axes([0.02, 0.01, 0.22, 0.26], frameon=False)
        radio_ax.set_title("Animate", loc="left", fontsize=11, fontweight="bold")
        self.radio = RadioButtons(radio_ax, [f"{GROUP_OF[m]}: {METHODS[m]}" for m in METHODS], active=1,
                                  label_props={"fontsize": [9] * len(METHODS)})
        self.radio.on_clicked(lambda label: setattr(
            self, "selected", next(m for m in METHODS if label == f"{GROUP_OF[m]}: {METHODS[m]}")))

        def slider(y, label, low, high, value, **kw):
            return Slider(self.fig.add_axes([0.42, y, 0.22, 0.025]), label, low, high, valinit=value, **kw)

        sliders = {
            "epsilon": slider(0.225, "ε  ", 0, 1, EPSILON),
            "ucb_c": slider(0.185, "UCB c [Mb/s]  ", 0, max(DATA_RATE_MBPS / 2, 1.5 * UCB_C), UCB_C),
            "urgency_weight": slider(0.145, "urgency weight w  ", 0, max(2 * DATA_RATE_MBPS, 1.5 * URGENCY_WEIGHT),
                                     URGENCY_WEIGHT),
            "drop_step": slider(0.105, "drop step  ", 0, 2, DROP_STEP),
            "target_drop": slider(0.065, "target drop  ", 0, 1, TARGET_DROP),
        }
        for key, s in sliders.items():
            s.on_changed(lambda v, key=key: PARAMS.update({key: v}))
        self.speed = slider(0.025, "speed [cycles/s]  ", 1, 500, START_SPEED, valstep=1)
        self.sliders = (*sliders.values(), self.speed)

        self.pause_button = Button(self.fig.add_axes([0.70, 0.09, 0.11, 0.06]), "Pause")
        self.reset_button = Button(self.fig.add_axes([0.83, 0.09, 0.11, 0.06]), "Reset")
        self.pause_button.on_clicked(self.toggle_pause)
        self.reset_button.on_clicked(self.reset)

    def toggle_pause(self, _):
        self.paused = not self.paused
        self.pause_button.label.set_text("Play" if self.paused else "Pause")


def simulate(method, seed, cycles):
    rng = random.Random(seed)
    scenario = [draw_cycle(rng) for _ in range(cycles)]
    random.seed(seed)
    state = new_state(method, seed)
    out = {k: np.zeros(cycles) for k in HISTORY + ("mean_u",)}
    for t, (anchor_ap, anchor_sta, rand_u) in enumerate(scenario):
        info = run_cycle(method, state, anchor_ap, anchor_sta, rand_u)
        for k in HISTORY:
            out[k][t] = info[k]
        out["mean_u"][t] = np.mean(list(state["traffic"].u.values()))
    return out


def make_graphs():
    plt.switch_backend("Agg")
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.3, "legend.frameon": False})
    time_s = np.arange(SMOOTHING, GRAPH_CYCLES + 1) * TXOP_MS / 1000
    results = {m: [simulate(m, seed, GRAPH_CYCLES) for seed in range(GRAPH_RUNS)] for m in METHODS}
    mean = {m: {k: np.mean([r[k] for r in results[m]], axis=0) for k in HISTORY + ("mean_u",)} for m in METHODS}

    def steady(m, key):
        if key == "drop":
            return 100 * mean[m]["dropped"][-1000:].sum() / mean[m]["arrived"][-1000:].sum()
        return mean[m][key][-1000:].mean()

    def panel(ax, methods, key, ylabel, title):
        for m in methods:
            ax.plot(time_s, smoothed(mean[m], key), color=METHOD_COLORS[m], lw=2.2,
                    ls="--" if m == "random" else "-", label=f"{METHODS[m]}: {steady(m, key):.2f}")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.legend(loc="upper right" if key == "drop" else "center right", fontsize=10)

    note = f"average of {GRAPH_RUNS} runs, {SMOOTHING}-cycle moving average, legend value = last 1000 cycles"

    def save(fig, name, title):
        fig.suptitle(f"{title} ({note})", x=0.01, ha="left", fontsize=12)
        fig.tight_layout()
        fig.savefig(name, dpi=150, bbox_inches="tight")
        plt.close(fig)

    group = GROUPS["No urgency"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    panel(axes[0], group, "rate", "Data rate [Mb/s]", "Data rate")
    panel(axes[1], group, "drop", "Dropped packets [%]", "Packets dropped")
    save(fig, "graph_no_urgency.png", "Without urgency")

    group = GROUPS["Random urgency"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    panel(axes[0], group, "utility", "Utility r + w·u [Mb/s]", "Utility")
    panel(axes[1], group, "u", "Urgency of the served partner station", "Urgency served")
    panel(axes[2], group, "drop", "Dropped packets [%]", "Packets dropped")
    save(fig, "graph_urgency.png", f"With random urgency, w = {URGENCY_WEIGHT:g}")

    group = GROUPS["Dynamic urgency"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    panel(axes[0], group, "rate", "Data rate [Mb/s]", "Data rate")
    panel(axes[1], group, "drop", "Dropped packets [%]", "Packets dropped")
    panel(axes[2], group, "mean_u", "Mean urgency of all stations", "Urgency level")
    save(fig, "graph_dynamic_urgency.png", f"With dynamic urgency from dropped packets, w = {URGENCY_WEIGHT:g}, "
                                           f"step {DROP_STEP:g}, target {TARGET_DROP:g}")

    print(f"{'method':<28}{'rate':>8}{'utility':>9}{'u served':>10}{'dropped %':>11}{'delivered Mb/s':>16}")
    for m in METHODS:
        print(f"{METHODS[m]:<28}{steady(m, 'rate'):8.2f}{steady(m, 'utility'):9.2f}"
              f"{steady(m, 'u'):10.3f}{steady(m, 'drop'):11.2f}{steady(m, 'throughput'):16.2f}")
    print("wrote graph_no_urgency.png, graph_urgency.png, graph_dynamic_urgency.png")


if __name__ == "__main__":
    check_topology()
    if "--graphs" in sys.argv:
        make_graphs()
    else:
        app = Live()
        plt.show()
