# IEEE 802.11bn Multi-AP Coordinated Spatial Reuse Simulator

In every TXOP one AP wins the channel (the sharing AP) and serves one of its stations. A hierarchical
multi-armed bandit (H-MAB) then picks a second AP (level 1) and one of its stations (level 2) to transmit
at the same time. The bandit learns from the data rate of that second link.

## Run

```
pip install -r requirements.txt
python main.py            # live window (chart: over time, or delay CDF per algorithm / access category)
python main.py --graphs   # run all experiments, graphs go to results/ (about 80 s)
```

## Project layout

```
main.py                 entry point
simulator/
  config.py             every constant, with the reason or assumption behind it
  topology.py           AP and station positions and each station's application
  channel.py            path loss, SINR and data rate
  traffic.py            packet queues, retries, deadline drops, per-AP drop urgency
  urgency.py            the four urgency types and the four access-category levels
  bandits.py            Random, epsilon-greedy, UCB and Thompson sampling agents
  hmab.py               two-level hierarchical bandit
  methods.py            a method = algorithm + urgency type; runs it cycle by cycle
  graphs.py             offline experiments and graphs
  live_view.py          live window
  style.py              shared colours
results/                generated graphs
paper/                  LaTeX source, bibliography and figures of the paper
references/             the two reference papers
```

## Methods

Every learning algorithm (epsilon-greedy, UCB, Thompson sampling) runs with every urgency type; Random is the
baseline.

| Urgency type | Urgency of a station |
|---|---|
| None | not used |
| Application (per station) | level of its access category: voice 1.0, video 0.75, best effort 0.5, background 0.25 |
| Packet drop (per AP) | level of its AP, raised or lowered by how much more or less the AP drops than the network |
| Combined | average of the two, rounded to one of the four levels |

With urgency, an arm's score is `Q + w * u` (plus the exploration term of UCB or Thompson), the sharing AP
picks its station with probability proportional to urgency, and a station with an empty queue has urgency 0.
The bandits always learn from the data rate only.

## Graphs

| File | Content |
|---|---|
| `graph_no_urgency.png` | data rate, drops and data-rate CDF without urgency |
| `graph_app_urgency.png` | application urgency |
| `graph_drop_urgency.png` | packet-drop urgency |
| `graph_combined_urgency.png` | combined urgency |
| `graph_cdf.png` | packet-delay CDFs per algorithm and per access category |
