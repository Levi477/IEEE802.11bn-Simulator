"""Live window: animates one method on the network and plots all methods of the same urgency type."""
import random
import time

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.widgets import Button, RadioButtons, Slider

from .bandits import ALGORITHMS
from .config import (ACCESS_CATEGORIES, DATA_RATE_MBPS, DROP_STEP, EPSILON, GRAPH_WINDOW, PARAMS, SMOOTHING,
                     START_SPEED, TARGET_DROP, TS_SIGMA, TXOP_MS, UCB_C, URGENCY_WEIGHT)
from .methods import METHOD_KEYS, METRICS, Method, draw_cycle, method_label, methods_of, smoothed
from .style import AC_COLORS, ALG_COLORS, ANCHOR, IDLE, INK, INK_SOFT, INTERFERENCE, PANEL, PARTNER
from .topology import APS, STATION_APP, STATION_POS, STATIONS
from .traffic import MAX_DEADLINE
from .urgency import LEVEL_NAME, MODES, station_level

PLOT_METRIC = {"none": ("rate", "Data rate [Mb/s]"), "app": ("utility", "Utility r + w·u [Mb/s]"),
               "drop": ("drop", "Dropped packets [%]"), "combined": ("utility", "Utility r + w·u [Mb/s]")}
HALO = [pe.withStroke(linewidth=3, foreground="white")]
VIEWS = {"time": "Over time", "cdf_alg": "Delay CDF: algorithms", "cdf_ac": "Delay CDF: access categories"}
KPI_WINDOW = 200   # cards average the last 200 cycles (1.1 s)


class LiveView:
    def __init__(self):
        self.algorithm, self.mode, self.paused = "ucb", "none", False
        self.fig = plt.figure(figsize=(16, 9.2), facecolor="white")
        self.fig.canvas.manager.set_window_title("IEEE 802.11bn C-SR simulator")
        self.fig.text(0.015, 0.965, "Multi-AP coordinated spatial reuse", fontsize=17, fontweight="bold", color=INK)
        self.fig.text(0.015, 0.94, "Hierarchical bandit picks the partner AP and station for every TXOP",
                      fontsize=11, color=INK_SOFT)
        self._build_network()
        self._build_plot()
        self._build_cards()
        self._build_bars()
        self._build_controls()
        self.reset()
        self.animation = FuncAnimation(self.fig, self.tick, interval=30, cache_frame_data=False)

    @property
    def selected(self):
        return "random|none" if self.algorithm == "random" else f"{self.algorithm}|{self.mode}"

    # ------------------------------------------------------------ simulation
    def reset(self, _=None):
        self.rng = random.Random()
        seed = random.randrange(1 << 30)          # same packet arrivals for every method
        self.methods = {k: Method(k, seed) for k in METHOD_KEYS}
        self.history = {k: {m: [] for m in METRICS} for k in METHOD_KEYS}
        self.partner_count = {k: dict.fromkeys(STATION_POS, 0) for k in METHOD_KEYS}
        self.cycle, self.progress, self.clock = 0, 0.0, time.perf_counter()
        self.advance()

    def advance(self):
        """One cycle for every method, all with the same TXOP winner."""
        anchor_ap, anchor_sta = draw_cycle(self.rng)
        self.last = {}
        for key, method in self.methods.items():
            info = method.run_cycle(anchor_ap, anchor_sta)
            self.last[key] = info
            for m in METRICS:
                self.history[key][m].append(info[m])
            self.partner_count[key][info["sta"]] += 1
        self.cycle += 1

    def tick(self, _):
        now = time.perf_counter()
        dt, self.clock = min(now - self.clock, 0.1), now
        if not self.paused:
            self.progress += self.speed.val * dt     # progress through the current TXOP, 0..1
            while self.progress >= 1:
                self.progress -= 1
                self.advance()
        self.draw()

    # --------------------------------------------------------------- drawing
    def draw(self):
        key = self.selected
        method, info = self.methods[key], self.last[key]
        self._draw_network(method, info)
        self._draw_plot(key)
        self._draw_cards(key, method)
        self._draw_bars(key, info)

    def _draw_network(self, method, info):
        mode, traffic = method.mode, method.traffic
        a_ap, a_sta, ap, sta, rate = info["anchor_ap"], info["anchor_sta"], info["ap"], info["sta"], info["rate"]
        levels = {s: station_level(mode, traffic, s) for s in STATION_POS} if mode != "none" else None

        self.ap_dots.set_facecolor([ANCHOR if a == a_ap else PARTNER if a == ap else IDLE for a in APS])
        self.sta_dots.set_sizes([60 + 300 * levels[s] if levels else 140 for s in STATION_POS])
        self.sta_dots.set_edgecolor([ANCHOR if s == a_sta else PARTNER if s == sta else "white" for s in STATION_POS])
        self.sta_dots.set_linewidth([3.5 if s in (a_sta, sta) else 1.0 for s in STATION_POS])
        for s, text in self.sta_text.items():
            label = f"{s}  {STATION_APP[s][3:]}\nqueue {traffic.queued(s)}"
            text.set_text(label + (f"  u={levels[s]:.2f}" if levels else ""))
        for a, text in self.ap_text.items():
            label = a + (f"  ·  {LEVEL_NAME[traffic.ap_level(a)][3:]}" if mode in ("drop", "combined") else "")
            if traffic.last_drop[a] is not None:
                label += f"\ndrop {100 * traffic.last_drop[a]:.0f}%"
            text.set_text(label)

        for line, (src, dst), width in ((self.anchor_link, (a_ap, a_sta), 3),
                                        (self.partner_link, (ap, sta), max(1, 10 * rate / DATA_RATE_MBPS)),
                                        (self.interference_link, (a_ap, sta), 2)):
            line.set_data(*zip(APS[src], STATION_POS[dst]))
            line.set_linewidth(width)
        for ring, name in ((self.anchor_ring, a_ap), (self.partner_ring, ap)):
            ring.set_center(APS[name])
            ring.set_radius(self.span * (0.04 + 0.30 * self.progress))
            ring.set_alpha(1 - self.progress)
        self.ax_net.set_title(f"{method.label}   ·   cycle {self.cycle:,}\n"
                              f"{a_ap} → {a_sta}  +  {ap} → {sta}  ({rate:.0f} Mb/s)",
                              fontsize=12, color=INK, loc="left")

    def _draw_plot(self, key):
        """Top-right chart: one of three views, chosen with the 'Chart' buttons."""
        mode = key.split("|")[1]
        shown = methods_of(mode)
        if self.view == "time":
            metric, ylabel = PLOT_METRIC[mode]
            title, xlabel = f"{MODES[mode]}  ({SMOOTHING}-cycle moving average)", "Cycle"
            curves = [self.lines[k] for k in shown]
            for line in curves:
                series = {m: np.array(v[-GRAPH_WINDOW:]) for m, v in self.history[line.key].items()}
                if len(series["rate"]) >= SMOOTHING:
                    y = smoothed(series, metric, SMOOTHING)
                    line.set_data(self.cycle - len(y) + 1 + np.arange(len(y)), y)
        elif self.view == "cdf_alg":
            ylabel, xlabel = "Delivered within delay [%]", "Packet delay [ms]"
            title = f"{MODES[mode]}: packet-delay CDF since reset"
            curves = [self.lines[k] for k in shown]
            for line in curves:
                line.set_data(*self.methods[line.key].traffic.delay_cdf(ACCESS_CATEGORIES))
        else:
            ylabel, xlabel = "Delivered within delay [%]", "Packet delay [ms]"
            title = f"{self.methods[key].label}: delay CDF per access category"
            curves = list(self.ac_lines.values())
            for ac, line in self.ac_lines.items():
                line.set_data(*self.methods[key].traffic.delay_cdf([ac]))

        for line in list(self.lines.values()) + list(self.ac_lines.values()):
            line.set_visible(line in curves)
        state = (mode, self.view, title)
        if state != self.plot_state:
            self.plot_state = state
            self.ax_plot.set(xlabel=xlabel, ylabel=ylabel)
            self.ax_plot.set_title(title, loc="left", fontweight="bold", fontsize=12)
            self.ax_plot.legend(handles=curves, loc="lower left", ncol=4, fontsize=9, frameon=False,
                                bbox_to_anchor=(0, 1.08))
        if self.view == "time":
            self.ax_plot.set_xlim(max(0, self.cycle - GRAPH_WINDOW), max(self.cycle, 300))
            ys = [line.get_ydata() for line in curves if len(line.get_ydata())]
            if ys:
                low, high = min(y.min() for y in ys), max(y.max() for y in ys)
                pad = max(2.0, 0.1 * (high - low))
                self.ax_plot.set_ylim(low - pad, high + pad)
        else:
            self.ax_plot.set_xlim(0, (MAX_DEADLINE + 1) * TXOP_MS)
            self.ax_plot.set_ylim(0, 100)

    def _draw_cards(self, key, method):
        h = {m: np.array(v[-KPI_WINDOW:]) for m, v in self.history[key].items()}
        traffic = method.traffic
        voice = 100 * traffic.ac_dropped["AC_VO"] / max(traffic.ac_arrived["AC_VO"], 1)
        values = (f"{h['rate'].mean():.1f} Mb/s", f"{100 * h['dropped'].sum() / max(h['arrived'].sum(), 1):.1f} %",
                  f"{voice:.1f} %", f"{h['throughput'].mean():.0f} Mb/s")
        for text, value in zip(self.card_values, values):
            text.set_text(value)

    def _draw_bars(self, key, info):
        share = [100 * self.partner_count[key][s] / self.cycle for s in STATION_POS]
        for bar, s, value in zip(self.bars, STATION_POS, share):
            bar.set_width(value)
            bar.set_edgecolor(PARTNER if s == info["sta"] else "none")
        self.ax_bars.set_xlim(0, max(20, 1.15 * max(share)))

    # ---------------------------------------------------------------- layout
    def _build_network(self):
        xs = [p[0] for p in list(APS.values()) + list(STATION_POS.values())]
        ys = [p[1] for p in list(APS.values()) + list(STATION_POS.values())]
        self.span = max(max(xs) - min(xs), max(ys) - min(ys), 10)
        pad = 0.24 * self.span
        ax = self.ax_net = self.fig.add_axes([0.0, 0.36, 0.52, 0.52])
        ax.set(xlim=(min(xs) - pad, max(xs) + pad), ylim=(min(ys) - pad, max(ys) + pad), aspect="equal")
        ax.axis("off")
        self.anchor_ring = ax.add_patch(Circle((0, 0), 1, fill=False, lw=2.5, ec=ANCHOR))
        self.partner_ring = ax.add_patch(Circle((0, 0), 1, fill=False, lw=2.5, ec=PARTNER))
        self.interference_link, = ax.plot([], [], color=INTERFERENCE, ls="--", zorder=1)
        self.anchor_link, = ax.plot([], [], color=ANCHOR, zorder=2)
        self.partner_link, = ax.plot([], [], color=PARTNER, zorder=2)
        self.ap_dots = ax.scatter(*zip(*APS.values()), marker="^", s=420, edgecolor=INK, zorder=4)
        self.sta_dots = ax.scatter(*zip(*STATION_POS.values()), s=140, zorder=4,
                                   color=[AC_COLORS[STATION_APP[s]] for s in STATION_POS])
        center_x, center_y = np.mean(xs), np.mean(ys)
        self.ap_text = {}
        for a, (x, y) in APS.items():
            below = y >= center_y          # label on the side facing the middle of the layout
            self.ap_text[a] = ax.annotate("", (x, y), xytext=(0, -18 if below else 18), textcoords="offset points",
                                          ha="center", va="top" if below else "bottom", fontweight="bold",
                                          fontsize=10.5, color=INK, zorder=6, path_effects=HALO)
        self.sta_text = {}
        for ap, stations in STATIONS.items():
            ax_, ay_ = APS[ap]
            for i, (s, (x, y)) in enumerate(stations.items()):
                right = x >= ax_                     # horizontally: always on the side away from the AP
                up = y >= ay_ if i else y < center_y  # vertically: away from the AP, but the near station faces the middle
                self.sta_text[s] = ax.annotate("", (x, y), xytext=(8 if right else -8, 8 if up else -8),
                                               textcoords="offset points", ha="left" if right else "right",
                                               va="bottom" if up else "top", fontsize=8.5, color=INK_SOFT,
                                               linespacing=1.15, zorder=6, path_effects=HALO)
        handles = [Line2D([], [], color=ANCHOR, lw=3, label="sharing AP → station (won the TXOP)"),
                   Line2D([], [], color=PARTNER, lw=3, label="partner AP → station (bandit's choice)"),
                   Line2D([], [], color=INTERFERENCE, lw=2, ls="--", label="interference at the partner station")]
        handles += [Line2D([], [], marker="o", ls="", color=AC_COLORS[ac], ms=8, label=f"{ac[3:]}: {info['name']}")
                    for ac, info in ACCESS_CATEGORIES.items()]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=8.5, frameon=False,
                  title="dot colour = access category, dot size = urgency", title_fontsize=9)

    def _build_plot(self):
        ax = self.ax_plot = self.fig.add_axes([0.56, 0.62, 0.42, 0.25])
        self.lines = {}
        for k in METHOD_KEYS:
            self.lines[k], = ax.plot([], [], label=method_label(k, short=True), lw=2.2,
                                     color=ALG_COLORS[k.split("|")[0]], ls="--" if k.startswith("random") else "-")
            self.lines[k].key = k
        self.ac_lines = {ac: ax.plot([], [], lw=2.2, color=AC_COLORS[ac], label=f"{ac[3:]} ({info['name']})")[0]
                         for ac, info in ACCESS_CATEGORIES.items()}
        ax.grid(alpha=0.3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        self.view, self.plot_state = "time", None

    def _build_cards(self):
        labels = ("Data rate", "Packets dropped", "Voice dropped", "Delivered")
        hints = (f"last {KPI_WINDOW} cycles", f"last {KPI_WINDOW} cycles", "since reset", f"last {KPI_WINDOW} cycles")
        self.card_values = []
        for i, (label, hint) in enumerate(zip(labels, hints)):
            x = 0.56 + i * 0.107
            self.fig.patches.append(FancyBboxPatch((x, 0.465), 0.097, 0.085, boxstyle="round,pad=0.004,rounding_size=0.01",
                                                   transform=self.fig.transFigure, fc=PANEL, ec="none"))
            self.fig.text(x + 0.008, 0.53, label, fontsize=9.5, color=INK_SOFT)
            self.card_values.append(self.fig.text(x + 0.008, 0.49, "", fontsize=15, fontweight="bold", color=INK))
            self.fig.text(x + 0.008, 0.472, hint, fontsize=7.5, color=INK_SOFT)

    def _build_bars(self):
        ax = self.ax_bars = self.fig.add_axes([0.63, 0.2, 0.35, 0.22])
        self.bars = ax.barh(range(len(STATION_POS)), np.zeros(len(STATION_POS)), linewidth=2,
                            color=[AC_COLORS[STATION_APP[s]] for s in STATION_POS])
        ax.set_yticks(range(len(STATION_POS)), [f"{s} ({STATION_APP[s][3:]})" for s in STATION_POS], fontsize=8.5)
        ax.invert_yaxis()
        ax.set_title("How often each station is the partner [% of cycles]", loc="left", fontsize=10.5,
                     fontweight="bold")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    def _build_controls(self):
        def radio(rect, title, options, active, attr):
            ax = self.fig.add_axes(rect, frameon=False)
            ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
            buttons = RadioButtons(ax, list(options.values()), active=active)
            buttons.on_clicked(lambda label: setattr(self, attr, next(k for k, v in options.items() if v == label)))
            return buttons

        algorithms = {k: name for k, (name, _) in ALGORITHMS.items()}
        self.alg_radio = radio([0.015, 0.02, 0.1, 0.19], "Algorithm", algorithms, 2, "algorithm")
        self.mode_radio = radio([0.12, 0.02, 0.15, 0.19], "Urgency", MODES, 0, "mode")

        def slider(row, label, low, high, value, **kw):
            return Slider(self.fig.add_axes([0.385, 0.215 - row * 0.033, 0.17, 0.02]), label, low, high,
                          valinit=value, color=ALG_COLORS["egreedy"], **kw)

        sliders = {
            "epsilon": slider(0, "ε (ε-greedy)", 0, 1, EPSILON),
            "ucb_c": slider(1, "c (UCB) [Mb/s]", 0, 60, UCB_C),
            "ts_sigma": slider(2, "σ (Thompson) [Mb/s]", 0, 80, TS_SIGMA),
            "urgency_weight": slider(3, "urgency weight w", 0, 240, URGENCY_WEIGHT),
            "drop_step": slider(4, "drop step", 0, 2, DROP_STEP),
            "target_drop": slider(5, "target drop", 0, 1, TARGET_DROP),
        }
        for name, s in sliders.items():
            s.on_changed(lambda value, name=name: PARAMS.update({name: value}))
        self.speed = slider(6, "speed [cycles/s]", 1, 500, START_SPEED, valstep=1)
        self.sliders = (*sliders.values(), self.speed)

        self.view_radio = radio([0.62, 0.02, 0.17, 0.12], "Chart", VIEWS, 0, "view")
        self.pause_button = Button(self.fig.add_axes([0.81, 0.07, 0.08, 0.05]), "Pause", color=PANEL)
        self.reset_button = Button(self.fig.add_axes([0.9, 0.07, 0.08, 0.05]), "Reset", color=PANEL)
        self.pause_button.on_clicked(self.toggle_pause)
        self.reset_button.on_clicked(self.reset)

    def toggle_pause(self, _):
        self.paused = not self.paused
        self.pause_button.label.set_text("Play" if self.paused else "Pause")


def run_live():
    app = LiveView()
    plt.show()
    return app
