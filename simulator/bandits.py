"""Bandit agents. Each agent owns a set of arms and keeps, per arm, a value estimate Q and a play count n."""
import math
import random

from .config import ALPHA, PARAMS


class Bandit:
    """Base class: stores Q and n and implements the shared learning rule."""

    def __init__(self, arms):
        self.q = {arm: 0.0 for arm in arms}
        self.n = {arm: 0 for arm in arms}

    def choose(self, bonus):
        """Return an arm. bonus[arm] is the urgency term w * u added to the arm's score."""
        raise NotImplementedError

    # Formula (exponential moving average): Q <- (1 - ALPHA) * Q + ALPHA * r,  n <- n + 1
    def update(self, arm, reward):
        self.q[arm] = (1 - ALPHA) * self.q[arm] + ALPHA * reward
        self.n[arm] += 1


class RandomBandit(Bandit):
    """Baseline: picks uniformly at random and ignores what it learned."""

    def choose(self, bonus):
        return random.choice(list(self.q))


class EpsilonGreedy(Bandit):
    # Formula: with probability epsilon a random arm, otherwise argmax (Q + bonus), ties broken at random.
    def choose(self, bonus):
        if random.random() < PARAMS["epsilon"]:
            return random.choice(list(self.q))
        return max(self.q, key=lambda a: (self.q[a] + bonus[a], random.random()))


class UCB(Bandit):
    # Formula: try every arm once, then argmax (Q + bonus + c * sqrt(ln N / n)),  N = total plays.
    def choose(self, bonus):
        untried = [a for a in self.q if self.n[a] == 0]
        if untried:
            return random.choice(untried)
        total = sum(self.n.values())
        return max(self.q, key=lambda a: self.q[a] + bonus[a]
                   + PARAMS["ucb_c"] * math.sqrt(math.log(total) / self.n[a]))


class Thompson(Bandit):
    # Formula (Gaussian Thompson sampling): sample Q + sigma * z / sqrt(n + 1), z ~ N(0, 1),
    # and pick argmax (sample + bonus). Rarely played arms get wide samples, so they are explored.
    def choose(self, bonus):
        return max(self.q, key=lambda a: self.q[a] + bonus[a]
                   + PARAMS["ts_sigma"] * random.gauss(0, 1) / math.sqrt(self.n[a] + 1))


ALGORITHMS = {"random": ("Random", RandomBandit), "egreedy": ("ε-greedy", EpsilonGreedy),
              "ucb": ("UCB", UCB), "thompson": ("Thompson", Thompson)}
