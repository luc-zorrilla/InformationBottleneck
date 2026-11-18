"""
Generic Euler–Maruyama simulator for overdamped Langevin dynamics:

    dX = F(X, t; params) dt + noise_increment(X, t, dt; params)

The state X may be a scalar or a NumPy array. The first axis represents
independent trajectories, so initial_positions can have shape (n_trajectories, ...)
or just the shape of one state.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import plotly.graph_objects as go


Force = Callable[[np.ndarray, float, dict[str, Any]], np.ndarray]
NoiseIncrement = Callable[
    [np.ndarray, float, float, dict[str, Any], np.random.Generator],
    np.ndarray,
]


def simulate_langevin(
    initial_positions: np.ndarray | float,
    dt: float,
    n_steps: int,
    force: Force,
    noise_increment: NoiseIncrement,
    params: dict[str, Any] | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate trajectories with the Euler–Maruyama method.

    Args:
        initial_positions: Initial state, or a batch of initial states. For
            example, shape (n_trajectories, dimension).
        dt: Positive time step.
        n_steps: Number of integration steps.
        force: Callable force(X, t, params), returning an array with the
            same shape as X.
        noise_increment: Callable noise_increment(X, t, dt, params, rng),
            returning the random increment to add during one time step.
        params: Optional model parameters passed to force and noise_increment.
        seed: Optional random seed.

    Returns:
        times: Array of shape (n_steps + 1,).
        trajectories: Array whose first axis is time. Its remaining axes
            match the batched initial state.
    """
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative.")

    params = {} if params is None else params
    rng = np.random.default_rng(seed)

    initial = np.asarray(initial_positions, dtype=float)
    # Treat a single state as a batch containing one trajectory.
    if initial.ndim == 0:
        initial = initial.reshape(1)

    times = np.arange(n_steps + 1) * dt
    trajectories = np.empty((n_steps + 1,) + initial.shape, dtype=float)
    trajectories[0] = initial

    for i in range(n_steps):
        t = times[i]
        x = trajectories[i]
        drift = np.asarray(force(x, t, params), dtype=float)
        random_step = np.asarray(
            noise_increment(x, t, dt, params, rng), dtype=float
        )

        if drift.shape != x.shape or random_step.shape != x.shape:
            raise ValueError(
                "force and noise_increment must return arrays with "
                "the same shape as the state."
            )

        trajectories[i + 1] = x + drift * dt + random_step

    return times, trajectories


def brownian_noise(
    x: np.ndarray,
    t: float,
    dt: float,
    params: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    """Additive noise for dX = F dt + sigma dW.

    sigma can be a scalar or an array broadcastable to the state shape.
    """
    sigma = np.asarray(params.get("sigma", 1.0), dtype=float)
    return sigma * np.sqrt(dt) * rng.standard_normal(size=x.shape)

def plot_trajectories(
    times: np.ndarray,
    trajectories: np.ndarray,
    title: str = "Langevin trajectories",
    line_opacity: float = 0.2,
) -> go.Figure:
    """Plot trajectories as overlapping, semi-transparent grey lines."""
    if trajectories.shape[0] != len(times):
        raise ValueError("The first trajectory axis must match len(times).")
    if not 0.0 <= line_opacity <= 1.0:
        raise ValueError("line_opacity must be between 0 and 1.")

    values = trajectories.reshape(len(times), -1)
    fig = go.Figure()

    for j in range(values.shape[1]):
        fig.add_trace(
            go.Scatter(
                x=times,
                y=values[:, j],
                mode="lines",
                line=dict(color=f"rgba(90, 90, 90, {line_opacity})"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Position",
        template="plotly_white",
    )
    return fig


def plot_trajectory_histogram(
    trajectories: np.ndarray,
    bins: int = 50,
    title: str = "Position distribution",
) -> go.Figure:
    """Plot a density histogram of all positions across all times and trajectories."""
    values = np.asarray(trajectories, dtype=float).ravel()
    values = values[np.isfinite(values)]

    if values.size == 0:
        raise ValueError("trajectories must contain at least one finite value.")
    if bins < 1:
        raise ValueError("bins must be at least 1.")

    fig = go.Figure(
        go.Histogram(
            x=values,
            nbinsx=bins,
            histnorm="probability density",
            marker_color="grey",
            opacity=0.75,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Position",
        yaxis_title="Probability density",
        template="plotly_white",
    )
    return fig

def plot_potential_from_force(
    force: Force,
    x_min: float,
    x_max: float,
    params: dict[str, Any] | None = None,
    n_points: int = 1_000,
    t: float = 0.0,
    title: str = "Potential reconstructed from force",
) -> go.Figure:
    """Plot a 1D potential reconstructed from its force by numerical integration.

    Assumes force(x) = -dV/dx and that `force` accepts an array of x values.
    """
    if x_max <= x_min:
        raise ValueError("x_max must be greater than x_min.")
    if n_points < 2:
        raise ValueError("n_points must be at least 2.")

    params = {} if params is None else params
    x = np.linspace(x_min, x_max, n_points)
    force_values = np.asarray(force(x, t, params), dtype=float)

    if force_values.shape != x.shape:
        raise ValueError("force must return an array with the same shape as x.")

    # V(x) = -integral F(x) dx. Use cumulative trapezoidal integration.
    dx = np.diff(x)
    potential = np.empty_like(x)
    potential[0] = 0.0
    potential[1:] = np.cumsum(
        -0.5 * (force_values[:-1] + force_values[1:]) * dx
    )
    potential -= np.min(potential)

    fig = go.Figure(
        go.Scatter(
            x=x,
            y=potential,
            mode="lines",
            line=dict(color="grey"),
            showlegend=False,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Position",
        yaxis_title="Potential V(x)",
        template="plotly_white",
    )
    return fig


# Example: Ornstein–Uhlenbeck process
#     dX = -theta (X - mu) dt + sigma dW
def ou_force(
    x: np.ndarray, t: float, params: dict[str, Any]
) -> np.ndarray:
    theta = params["theta"]
    mu = params.get("mu", 0.0)
    return -theta * (x - mu)

def double_well_force(
    x: np.ndarray, t: float, params: dict[str, Any]
) -> np.ndarray:
    """Force for U(x) = a * (x**2 - b**2)**2.

    The potential has minima at x = ±b.
    """
    a = params.get("a", 1.0)
    b = params.get("b", 1.0)
    return -4.0 * a * x * (x**2 - b**2)
    
def four_well_force(
    x: np.ndarray, t: float, params: dict[str, Any]
) -> np.ndarray:
    """1D force from V(x) = A*(x**2 - a**2)**2*(x**2 - b**2)**2."""
    A = params.get("A", 1.0)
    a = params.get("a", 0.5)
    b = params.get("b", 1.0)

    u = x**2
    return -8.0 * A * x * (u - a**2) * (u - b**2) * (
        u - 0.5 * (a**2 + b**2)
    )


if __name__ == "__main__":
    ou_params = {
        "theta": 1.0,  # Mean-reversion rate
        "mu": 0.0,     # Long-run mean
        "sigma": 0.7,  # Noise amplitude
    }

    double_well_params = {
        "a": 1.0,
        "b": 1.0,
        "sigma": 0.7,  # Noise amplitude
    }

    four_well_params = {
        "A": 0.5,
        "a": 0.5,
        "b": 1.0,
        "sigma": 0.1,  # Noise amplitude
    }

    times, trajectories = simulate_langevin(
        initial_positions=np.linspace(-1.0, 1.0, 100),
        dt=0.01,
        n_steps=10000,
        force=four_well_force,# ou_force,
        noise_increment=brownian_noise,
        params=four_well_params,
        seed=42,
    )

    plot_potential_from_force(
        four_well_force,
        x_min=-1.1,
        x_max=1.1,
        params=four_well_params,
    ).show()
    plot_trajectories(times, trajectories, line_opacity=0.15).show()
    plot_trajectory_histogram(trajectories, bins=1000).show()