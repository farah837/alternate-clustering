"""
synthetic_data.py
-----------------
Generates the synthetic time series dataset used for ACeDeC clustering experiments.

4 clusters x 100 samples x 100 time steps:
  - Cluster 0: High-frequency oscillation (fast vibration / sensor)
  - Cluster 1: Low-frequency sine + upward trend (seasonal with drift)
  - Cluster 2: Cumulative random walk (Brownian motion / stock price)
  - Cluster 3: Damped oscillation (mechanical resonance decaying)

Noise sigma = 0.3 (realistic — clusters overlap visually but are structurally distinct).

Usage:
    from synthetic_data import load_synthetic_timeseries
    X, y_true = load_synthetic_timeseries()
"""

import numpy as np


def load_synthetic_timeseries(
    n_per_class: int = 100,
    T: int = 100,
    noise: float = 0.3,
    random_state: int = 42,
) -> tuple:
    """
    Generate a synthetic time series dataset with 4 structural archetypes.

    Parameters
    ----------
    n_per_class : int
        Number of samples per cluster (default: 100)
    T : int
        Number of time steps per series (default: 100)
    noise : float
        Standard deviation of Gaussian noise added to each series (default: 0.3)
    random_state : int
        Random seed for reproducibility (default: 42)

    Returns
    -------
    X : np.ndarray, shape (n_per_class * 4, T), dtype float32
        The raw (unscaled) time series matrix.
    y_true : np.ndarray, shape (n_per_class * 4,)
        Integer cluster labels (0–3).
    """
    np.random.seed(random_state)
    t = np.linspace(0, 2 * np.pi, T)

    def high_freq(n):
        """4 full sine cycles — fast oscillation."""
        return np.sin(4 * t) + np.random.normal(0, noise, (n, T))

    def trend_sine(n):
        """1 sine cycle with a positive linear drift."""
        base = np.sin(t) + np.linspace(0, 1.5, T)
        return base + np.random.normal(0, noise, (n, T))

    def random_walk(n):
        """Cumulative sum of Gaussian steps — Brownian motion."""
        steps = np.random.normal(0, 0.15, (n, T))
        return np.cumsum(steps, axis=1) + np.random.normal(0, noise * 0.5, (n, T))

    def damped(n):
        """Sine wave with exponential decay envelope."""
        decay = np.exp(-t / (2 * np.pi))
        return np.sin(3 * t) * decay + np.random.normal(0, noise * 0.7, (n, T))

    X = np.vstack([
        high_freq(n_per_class),
        trend_sine(n_per_class),
        random_walk(n_per_class),
        damped(n_per_class),
    ]).astype(np.float32)

    y_true = np.repeat(np.arange(4), n_per_class)

    return X, y_true


if __name__ == "__main__":
    X, y_true = load_synthetic_timeseries()
    print(f"X shape  : {X.shape}")
    print(f"y shape  : {y_true.shape}")
    print(f"Classes  : {np.unique(y_true)}")
    print(f"X mean   : {X.mean():.4f}, std: {X.std():.4f}")
