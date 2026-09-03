"""Baseline selectors, all under identical cardinality and weighting rules.

Every method returns ``(x, EvalCounter)`` so that solution quality, objective
evaluation count and best-so-far trajectory are recorded on a common footing.
The metaheuristics operate directly on fixed-cardinality binary selections
(GA, SA) or decode a continuous vector by taking its B largest components
(PSO, DE), which is the standard treatment for these operators on a
choose-B-of-N problem.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .analyst import asset_sharpe_ratios
from .risk import EvalCounter

__all__ = ["classical_mvo", "strong_greedy", "genetic_algorithm",
           "particle_swarm", "differential_evolution", "simulated_annealing",
           "random_sampling", "BASELINE_REGISTRY", "DEFAULT_SETTINGS"]

DEFAULT_SETTINGS = {
    "ga": {"population": 50, "generations": 80, "tournament": 3,
           "mutation_rate": 0.20, "elitism": 2},
    "pso": {"particles": 40, "iterations": 80, "w": 0.7, "c1": 1.5,
            "c2": 1.5},
    "de": {"population": 40, "generations": 80, "F": 0.8, "CR": 0.9},
    "sa": {"iterations": 1200, "t0": 1.0, "cooling": 0.995},
}


def _random_selection(n: int, b: int, rng: np.random.Generator) -> np.ndarray:
    x = np.zeros(n, dtype=int)
    x[rng.choice(n, size=b, replace=False)] = 1
    return x


def _decode_topb(vector: np.ndarray, b: int) -> np.ndarray:
    x = np.zeros(vector.size, dtype=int)
    x[np.argsort(-vector)[:b]] = 1
    return x


def classical_mvo(mu, Sigma, B, rng, rf=0.0, **_):
    """Continuous maximum-Sharpe solution truncated to B assets.

    This is the classical reference retained because of its prevalence in
    practice.  The truncation step is precisely where solution quality is lost
    when the optimum lies on the non-convex boundary of the feasible set.
    """
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    n = mu.size
    counter = EvalCounter(mu, Sigma, rf)

    def negative_sharpe(w):
        vol = np.sqrt(max(float(w @ Sigma @ w), 1e-24))
        return -(float(mu @ w) - rf) / vol

    result = minimize(
        negative_sharpe, np.full(n, 1.0 / n), method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 300, "ftol": 1e-10})
    weights = np.clip(result.x, 0.0, None) if result.success else np.full(
        n, 1.0 / n)
    x = _decode_topb(weights, B)
    counter.sharpe(x)
    return x, counter


def strong_greedy(mu, Sigma, B, rng, rf=0.0, **_):
    """Top-B assets by individual Sharpe ratio; ignores pairwise correlation."""
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    counter = EvalCounter(mu, Sigma, rf)
    ranking = asset_sharpe_ratios(mu, Sigma, rf)
    x = _decode_topb(ranking, B)
    counter.sharpe(x)
    return x, counter


def random_sampling(mu, Sigma, B, rng, rf=0.0, n_samples=1024, **_):
    """Uniform sampling of the feasible set; the null model for Tier 2."""
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    counter = EvalCounter(mu, Sigma, rf)
    n = mu.size
    for _ in range(int(n_samples)):
        counter.sharpe(_random_selection(n, B, rng))
    return counter.best_x, counter


def genetic_algorithm(mu, Sigma, B, rng, rf=0.0, settings=None, **_):
    cfg = {**DEFAULT_SETTINGS["ga"], **(settings or {})}
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    n = mu.size
    counter = EvalCounter(mu, Sigma, rf)

    population = [_random_selection(n, B, rng)
                  for _ in range(cfg["population"])]
    fitness = np.array([counter.sharpe(ind) for ind in population])

    def tournament():
        contenders = rng.choice(len(population), size=cfg["tournament"],
                                replace=False)
        return population[contenders[np.argmax(fitness[contenders])]].copy()

    def repair(child):
        ones = np.flatnonzero(child == 1)
        if ones.size > B:
            child[rng.choice(ones, size=ones.size - B, replace=False)] = 0
        elif ones.size < B:
            zeros = np.flatnonzero(child == 0)
            child[rng.choice(zeros, size=B - ones.size, replace=False)] = 1
        return child

    for _ in range(cfg["generations"]):
        order = np.argsort(-fitness)
        new_population = [population[i].copy()
                          for i in order[:cfg["elitism"]]]
        while len(new_population) < cfg["population"]:
            p1, p2 = tournament(), tournament()
            mask = rng.random(n) < 0.5
            child = np.where(mask, p1, p2)
            if rng.random() < cfg["mutation_rate"]:
                ones = np.flatnonzero(child == 1)
                zeros = np.flatnonzero(child == 0)
                if ones.size and zeros.size:
                    child[rng.choice(ones)] = 0
                    child[rng.choice(zeros)] = 1
            new_population.append(repair(child))
        population = new_population
        fitness = np.array([counter.sharpe(ind) for ind in population])

    return counter.best_x, counter


def particle_swarm(mu, Sigma, B, rng, rf=0.0, settings=None, **_):
    cfg = {**DEFAULT_SETTINGS["pso"], **(settings or {})}
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    n = mu.size
    counter = EvalCounter(mu, Sigma, rf)

    position = rng.random((cfg["particles"], n))
    velocity = rng.normal(0.0, 0.1, size=(cfg["particles"], n))
    personal_best = position.copy()
    personal_value = np.array([counter.sharpe(_decode_topb(p, B))
                               for p in position])
    g = int(np.argmax(personal_value))
    global_best = personal_best[g].copy()

    for _ in range(cfg["iterations"]):
        r1 = rng.random((cfg["particles"], n))
        r2 = rng.random((cfg["particles"], n))
        velocity = (cfg["w"] * velocity
                    + cfg["c1"] * r1 * (personal_best - position)
                    + cfg["c2"] * r2 * (global_best - position))
        position = np.clip(position + velocity, 0.0, 1.0)
        values = np.array([counter.sharpe(_decode_topb(p, B))
                           for p in position])
        improved = values > personal_value
        personal_best[improved] = position[improved]
        personal_value[improved] = values[improved]
        g = int(np.argmax(personal_value))
        global_best = personal_best[g].copy()

    return counter.best_x, counter


def differential_evolution(mu, Sigma, B, rng, rf=0.0, settings=None, **_):
    cfg = {**DEFAULT_SETTINGS["de"], **(settings or {})}
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    n = mu.size
    counter = EvalCounter(mu, Sigma, rf)

    population = rng.random((cfg["population"], n))
    values = np.array([counter.sharpe(_decode_topb(p, B))
                       for p in population])

    for _ in range(cfg["generations"]):
        for i in range(cfg["population"]):
            pool = [j for j in range(cfg["population"]) if j != i]
            a, b, c = rng.choice(pool, size=3, replace=False)
            mutant = population[a] + cfg["F"] * (population[b] - population[c])
            crossover = rng.random(n) < cfg["CR"]
            if not crossover.any():
                crossover[rng.integers(n)] = True
            trial = np.where(crossover, mutant, population[i])
            trial = np.clip(trial, 0.0, 1.0)
            value = counter.sharpe(_decode_topb(trial, B))
            if value > values[i]:
                population[i], values[i] = trial, value

    return counter.best_x, counter


def simulated_annealing(mu, Sigma, B, rng, rf=0.0, settings=None,
                        x0=None, **_):
    cfg = {**DEFAULT_SETTINGS["sa"], **(settings or {})}
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    n = mu.size
    counter = EvalCounter(mu, Sigma, rf)

    current = (_random_selection(n, B, rng) if x0 is None
               else np.asarray(x0, dtype=int).copy())
    current_value = counter.sharpe(current)
    temperature = cfg["t0"]

    for _ in range(cfg["iterations"]):
        candidate = current.copy()
        ones = np.flatnonzero(candidate == 1)
        zeros = np.flatnonzero(candidate == 0)
        candidate[rng.choice(ones)] = 0
        candidate[rng.choice(zeros)] = 1
        value = counter.sharpe(candidate)
        delta = value - current_value
        if delta > 0 or rng.random() < np.exp(delta / max(temperature, 1e-12)):
            current, current_value = candidate, value
        temperature *= cfg["cooling"]

    return counter.best_x, counter


BASELINE_REGISTRY = {
    "Classical MVO": classical_mvo,
    "Strong Greedy": strong_greedy,
    "GA": genetic_algorithm,
    "PSO": particle_swarm,
    "DE": differential_evolution,
    "SA": simulated_annealing,
}
