"""Entry point.

    python main.py            open the live window
    python main.py --graphs   run all experiments and save the graphs in results/
"""
import argparse

from simulator.topology import check_topology


def main():
    parser = argparse.ArgumentParser(description="IEEE 802.11bn multi-AP coordinated spatial reuse simulator")
    parser.add_argument("--graphs", action="store_true", help="run the experiments and save the graphs")
    args = parser.parse_args()
    check_topology()
    if args.graphs:
        from simulator.graphs import make_graphs
        make_graphs()
    else:
        from simulator.live_view import run_live
        run_live()


if __name__ == "__main__":
    main()
