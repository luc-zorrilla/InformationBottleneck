"""
Discrete predictive Information Bottleneck with an Ulam approximation.

Requires:
    numpy

Run tests:
    python predictive_ib.py --test

Run example:
    python predictive_ib.py
"""

from __future__ import annotations

import argparse
import unittest
from typing import NamedTuple

import numpy as np
from scipy.linalg import eig

import plotly.graph_objects as go
from plotly.subplots import make_subplots

class UlamResult(NamedTuple):
    """Discrete transition model and estimated stationary distribution."""

    transition: np.ndarray  # P[i, j] = p(x_next=j | x=i)
    stationary: np.ndarray  # p(x=i)

class UlamEigenResult(NamedTuple):
    eigenvalues: np.ndarray
    right_eigenvectors: np.ndarray  # columns v: P @ v = lambda * v
    left_eigenvalues: np.ndarray
    left_eigenvectors: np.ndarray   # columns u: P.T @ u = lambda * u

def empirical_stationary_distribution(
    trajectory: np.ndarray,
    bins=100,
    *,
    burn_in: int = 0,
    value_range=None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Estimate stationary bin probabilities from post-burn-in samples.

    The first axis is time. Any remaining axes before the final coordinate
    axis are treated as independent samples (for example, repeats). For a
    scalar system, pass data shaped (time,) or (time, repeats).

    Returns
    -------
    probabilities:
        Probability mass in each bin; sums to 1.
    bin_edges:
        One array of bin edges per coordinate.
    """
    data = np.asarray(trajectory, dtype=float)

    if data.ndim == 0 or data.shape[0] == 0:
        raise ValueError("trajectory must have a non-empty time axis.")
    if not np.all(np.isfinite(data)):
        raise ValueError("trajectory must contain only finite values.")
    if not isinstance(burn_in, (int, np.integer)) or burn_in < 0:
        raise ValueError("burn_in must be a nonnegative integer.")
    if burn_in >= data.shape[0]:
        raise ValueError("burn_in must be smaller than the number of time steps.")

    samples = data[burn_in:]

    # A scalar trajectory has one coordinate per sample.
    if samples.ndim == 1:
        points = samples.reshape(-1, 1)
    # A 1D system with repeats may be shaped (time, repeats).
    elif samples.ndim == 2:
        points = samples.reshape(-1, 1)
    else:
        # Last axis stores coordinates; preceding axes store samples/repeats.
        points = samples.reshape(-1, samples.shape[-1])

    if points.shape[0] == 0:
        raise ValueError("No samples remain after burn-in.")

    counts, bin_edges = np.histogramdd(
        points,
        bins=bins,
        range=value_range,
    )
    total = counts.sum()
    if total == 0:
        raise ValueError("No samples fell within the requested histogram range.")

    return counts / total, bin_edges

def stationary_distribution(
    transition: np.ndarray,
    *,
    tolerance: float = 1e-12,
    max_iterations: int = 100_000,
) -> np.ndarray:
    """Estimate a stationary distribution pi satisfying pi @ P = pi."""
    transition = np.asarray(transition, dtype=float)

    if transition.ndim != 2 or transition.shape[0] != transition.shape[1]:
        raise ValueError("transition must be a square matrix.")
    if np.any(transition < 0) or not np.all(np.isfinite(transition)):
        raise ValueError("transition entries must be finite and nonnegative.")
    if not np.allclose(transition.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("Each transition-matrix row must sum to 1.")

    n_states = transition.shape[0]
    distribution = np.full(n_states, 1.0 / n_states)

    for _ in range(max_iterations):
        updated = distribution @ transition
        residual = np.linalg.norm(updated - distribution, ord=1)

        if residual < tolerance:
            return updated / updated.sum()

        distribution = updated

    raise RuntimeError(
        f"Stationary-distribution iteration did not converge; "
        f"last residual = {residual:.3e} < {tolerance} = tolerance"
    )

def ulam_approximation(
    trajectories: np.ndarray,
    bin_edges: np.ndarray,
    *,
    burn_in: int = 0,
) -> UlamResult:
    """Estimate a Ulam transition matrix and empirical stationary occupancy.

    `stationary` is the normalized histogram of post-burn-in samples.
    """
    trajectories = np.asarray(trajectories, dtype=float)
    bin_edges = np.asarray(bin_edges, dtype=float)

    if trajectories.ndim == 1:
        trajectories = trajectories[None, :]
    if trajectories.ndim != 2:
        raise ValueError("trajectories must be 1D or 2D.")
    if trajectories.shape[1] == 0:
        raise ValueError("trajectories must contain at least one time step.")
    if not np.all(np.isfinite(trajectories)):
        raise ValueError("trajectories must contain only finite values.")
    if bin_edges.ndim != 1 or len(bin_edges) < 2:
        raise ValueError("bin_edges must be a 1D array with at least two edges.")
    if not np.all(np.isfinite(bin_edges)) or not np.all(np.diff(bin_edges) > 0):
        raise ValueError("bin_edges must be finite and strictly increasing.")
    if not isinstance(burn_in, (int, np.integer)) or burn_in < 0:
        raise ValueError("burn_in must be a nonnegative integer.")
    if burn_in >= trajectories.shape[1]:
        raise ValueError("burn_in must be smaller than the number of time steps.")

    n_bins = len(bin_edges) - 1
    counts = np.zeros((n_bins, n_bins), dtype=float)

    # Estimate transitions using only pairs whose source and target are
    # both in the post-burn-in portion of each trajectory.
    for trajectory in trajectories:
        post_burn = trajectory[burn_in:]
        if len(post_burn) < 2:
            continue

        source = np.searchsorted(bin_edges, post_burn[:-1], side="right") - 1
        target = np.searchsorted(bin_edges, post_burn[1:], side="right") - 1

        # Include values exactly equal to the final edge in the last bin.
        source[post_burn[:-1] == bin_edges[-1]] = n_bins - 1
        target[post_burn[1:] == bin_edges[-1]] = n_bins - 1

        valid = (
            (source >= 0) & (source < n_bins)
            & (target >= 0) & (target < n_bins)
        )
        np.add.at(counts, (source[valid], target[valid]), 1.0)

    row_sums = counts.sum(axis=1)
    transition = np.zeros_like(counts)
    observed_rows = row_sums > 0
    transition[observed_rows] = (
        counts[observed_rows] / row_sums[observed_rows, None]
    )

    # Keep unobserved rows stochastic without inventing outgoing transitions.
    unobserved = ~observed_rows
    transition[unobserved, unobserved] = 1.0

    # Empirical occupancy from all post-burn-in samples, pooled across runs.
    samples = trajectories[:, burn_in:].ravel()
    in_range = (samples >= bin_edges[0]) & (samples <= bin_edges[-1])
    samples = samples[in_range]

    if samples.size == 0:
        raise ValueError("No post-burn-in samples fall within the bin range.")

    occupancy, _ = np.histogram(samples, bins=bin_edges)
    stationary = occupancy.astype(float) / occupancy.sum()

    return UlamResult(transition=transition, stationary=stationary)


def ulam_eigendelements(result: UlamResult) -> UlamEigenResult:
    """Compute right and left eigensystems of an Ulam transition matrix."""
    matrix = np.asarray(result.transition, dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Ulam transition matrix must be square.")

    eigenvalues, right_vectors = np.linalg.eig(matrix)
    left_eigenvalues, left_vectors = np.linalg.eig(matrix.T)

    # Sort each eigensystem by decreasing eigenvalue magnitude.
    order = np.argsort(-np.abs(eigenvalues))
    eigenvalues = eigenvalues[order]
    right_vectors = right_vectors[:, order]

    left_order = np.argsort(-np.abs(left_eigenvalues))
    left_eigenvalues = left_eigenvalues[left_order]
    left_vectors = left_vectors[:, left_order]

    return UlamEigenResult(
        eigenvalues=eigenvalues,
        right_eigenvectors=right_vectors,
        left_eigenvalues=left_eigenvalues,
        left_eigenvectors=left_vectors,
    )

def _normalize_probabilities(values: np.ndarray, axis: int) -> np.ndarray:
    """Normalize nonnegative values along an axis."""
    values = np.asarray(values, dtype=float)

    if not np.all(np.isfinite(values)):
        raise ValueError("Probabilities must be finite.")
    if np.any(values < 0):
        raise ValueError("Probabilities must be nonnegative.")

    # Scaling preserves the normalized result and reduces overflow risk.
    scaled = values / np.max(values, axis=axis, keepdims=True)
    totals = scaled.sum(axis=axis, keepdims=True)

    if np.any(totals <= 0):
        raise ValueError("Cannot normalize a zero-mass distribution.")

    return scaled / totals

def mutual_information(joint: np.ndarray) -> float:
    """MI for a joint distribution indexed as joint[a, b]."""
    # I(A;B) = sum_a,b p(a,b) log[p(a,b)/(p(a)p(b))]

    p_a = joint.sum(axis=1, keepdims=True)
    p_b = joint.sum(axis=0, keepdims=True)
    denominator = p_a * p_b
    mask = (joint > 0) & (denominator > 0)
    return float(
        np.sum(joint[mask] * np.log(joint[mask] / denominator[mask]))
    )/ np.log(2)

def information_quantities(
    transition: np.ndarray,
    stationary: np.ndarray,
    encoder: np.ndarray,
) -> dict[str, float]:
    """Compute I(H;X) and I(H;X_next) in nats or bits.

    Use base=np.e for nats and base=2 for bits.
    """
    transition = np.asarray(transition, dtype=float)
    stationary = np.asarray(stationary, dtype=float)
    encoder = np.asarray(encoder, dtype=float)

    n_x = stationary.size
    if transition.shape != (n_x, n_x):
        raise ValueError("transition must have shape (n_states, n_states)")
    if encoder.ndim != 2 or encoder.shape[0] != n_x:
        raise ValueError("encoder must have shape (n_states, n_reduced_states)")
    if np.any(stationary < 0) or np.any(transition < 0) or np.any(encoder < 0):
        raise ValueError("Probabilities must be non-negative")
    if not np.isclose(stationary.sum(), 1.0):
        raise ValueError("stationary must sum to 1")
    if not np.allclose(transition.sum(axis=1), 1.0):
        raise ValueError("Each transition-matrix row must sum to 1")
    if not np.allclose(encoder.sum(axis=1), 1.0):
        raise ValueError("Each encoder row must sum to 1")

    # p(h, x) = p(x) p(h|x)
    p_hx = stationary[:, None] * encoder
    i_hx = mutual_information(p_hx.T)  # rows=h, columns=x

    # p(h, y) = sum_x p(x) p(h|x) P(x_next|x)
    p_hy = (stationary[:, None] * encoder).T @ transition
    i_hy = mutual_information(p_hy)

    return {"I_HX": i_hx, "I_HY": i_hy}

def blahut_arimoto(
    transition: np.ndarray,
    stationary: np.ndarray,
    n_reduced_states: int,
    beta: float,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 10_000,
    seed: int | None = 0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Optimize the encoder q(h|x) for the predictive Information Bottleneck, 
    using the Blahut-Arimoto algorithm.

    Minimizes:
        I(H; X) - beta * I(H; Y)

    where :
        - X is the current binned state
        - Y is the next binned state.
    The supplied transition matrix defines p(y|x), and `stationary` defines p(x).

    Returns
    -------
    encoder:
        Array shaped (n_full_states, n_reduced_states), with each row
        representing q(h|x).
    information:
        Dictionary containing I(H;X), I(H;Y), and the objective in bits.
    """
    transition = np.asarray(transition, dtype=float)
    stationary = np.asarray(stationary, dtype=float)

    if transition.ndim != 2 or transition.shape[0] != transition.shape[1]:
        raise ValueError("transition must be a square matrix.")
    n_states = transition.shape[0]
    if stationary.shape != (n_states,):
        raise ValueError("stationary must have one entry per full state.")
    if n_reduced_states < 1:
        raise ValueError("n_reduced_states must be at least 1.")
    if beta < 0:
        raise ValueError("beta must be nonnegative.")
    if not np.allclose(transition.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("Each transition-matrix row must sum to 1.")
    if np.any(transition < 0) or np.any(stationary < 0):
        raise ValueError("Probabilities must be nonnegative.")
    if not np.isclose(stationary.sum(), 1.0):
        raise ValueError("stationary must sum to 1.")

    # Remove zero-occupancy states; they do not affect either mutual information.
    # TODO: check if necessary
    active = stationary > 0
    p_x = stationary[active]
    p_x = p_x / p_x.sum()

    p_y_given_x = transition[active].copy()
    row_sums = p_y_given_x.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Every active transition row must have positive mass.")
    p_y_given_x /= row_sums

    # Initialize encoder randomly
    rng = np.random.default_rng(seed)
    encoder = rng.random((len(p_x), n_reduced_states)) + 1e-3
    encoder = _normalize_probabilities(encoder, axis=1)

    # Initialize marginal and conditional distributions as well
    eps = np.finfo(float).tiny
    p_h = p_x @ encoder # p(h) = sum_x p(x) p(h|x)
    p_y_given_h = ( # p(y|h) = sum_x p(y|x) p(x|h) = sum_x p(x) p(h|x) p(y|x) / p(h) (Bayes formula)
        (p_x[:, None] * encoder).T @ p_y_given_x
    ) / np.maximum(p_h[:, None], eps) 

    for _ in range(max_iterations):
        
        # 1. New encoder

        # KL[p(y|x) || p(y|h)] = sum_x p(y|x) log [p(y|x) / p(y|h)], for every x and h.
        log_p_y_given_x = np.log(np.maximum(p_y_given_x, eps))
        log_p_y_given_h = np.log(np.maximum(p_y_given_h, eps))
        kl = np.sum(
            p_y_given_x[:, None, :]
            * (log_p_y_given_x[:, None, :] - log_p_y_given_h[None, :, :]), 
            axis=2, # TODO: check if axis 2 is correct
        )

        # BA update in log space:
        # p(h|x) ∝ p(h) * exp(-beta * KL[p(y|x) || p(y|h)])
        log_weights = np.log(np.maximum(p_h, eps))[None, :] - beta * kl
        log_weights -= np.max(log_weights, axis=1, keepdims=True) # Stabilize each row before exponentiating.

        updated = _normalize_probabilities(np.exp(log_weights), axis=1)

        # Check convergence before replacing the current encoder.
        change = np.max(np.abs(updated - encoder))
        encoder = updated    

        # 2. Update marginal distribution

        # p(h) = sum_x p(x) p(h|x)
        p_h = p_x @ encoder 

        # 3. Update conditional distribution

        # p(y|h) = sum_x p(y|x) p(x|h) = sum_x p(x) p(h|x) p(y|x) / p(h) (Bayes formula)
        p_y_given_h = (
            (p_x[:, None] * encoder).T @ p_y_given_x
        ) / np.maximum(p_h[:, None], eps)

        if change < tolerance:
            break

    # Reinsert any zero-occupancy states as arbitrary normalized rows. # TODO: check if it makes sense
    full_encoder = np.full((n_states, n_reduced_states), 1.0 / n_reduced_states)
    full_encoder[active] = encoder

    information = information_quantities(
        transition, stationary, full_encoder
    )
    information["objective"] = (
        information["I_HX"] - beta * information["I_HY"]
    )
    return full_encoder, information

# Examples

def doubling_map(x: np.ndarray) -> np.ndarray:
    """One step of x -> 2x mod 1 on [0, 1)."""
    return np.mod(2.0 * x, 1.0)


def compare_ulam_with_exact(
    n_samples: int = 20_000,
    seed: int = 0,
        ) -> tuple[UlamResult, go.Figure]:
    """Estimate the two-bin Ulam matrix for the doubling map and plot results."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive.")

    rng = np.random.default_rng(seed)

    # Draw independent states from the uniform invariant distribution.
    x = rng.uniform(0.0, 1.0, size=n_samples)
    x_next = doubling_map(x)

    # Each row is one observed one-step transition: [x_t, x_(t+dt)].
    transition_pairs = np.column_stack((x, x_next))

    bin_edges = np.array([0.0, 0.5, 1.0])
    estimate = ulam_approximation(transition_pairs, bin_edges)

    exact = np.array([
        [0.5, 0.5],
        [0.5, 0.5],
    ])
    error = estimate.transition - exact

    labels = ["[0, 0.5)", "[0.5, 1)"]
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Exact matrix", "Ulam estimate", "Estimate − exact"),
    )

    for col, matrix, colorscale, zmin, zmax in [
        (1, exact, "Blues", 0.0, 1.0),
        (2, estimate.transition, "Blues", 0.0, 1.0),
        (3, error, "RdBu", -0.5, 0.5),
    ]:
        fig.add_trace(
            go.Heatmap(
                z=matrix,
                x=labels,
                y=labels,
                colorscale=colorscale,
                zmin=zmin,
                zmax=zmax,
                text=np.round(matrix, 4),
                texttemplate="%{text}",
                showscale=(col == 3),
                colorbar=dict(title="Error") if col == 3 else None,
            ),
            row=1,
            col=col,
        )

    fig.update_layout(
        title=f"Ulam approximation for the doubling map ({n_samples:,} samples)",
        template="plotly_white",
        height=420,
    )
    fig.update_xaxes(title_text="Next-state bin")
    fig.update_yaxes(title_text="Current-state bin", autorange="reversed")

    return estimate, fig

def test_ib_on_four_bin_doubling_map(
    beta: float = 100,
    n_seeds: int = 20,
) -> tuple[np.ndarray, dict[str, float], go.Figure]:
    """Test predictive IB on the exact four-bin doubling-map transition matrix."""
    transition = np.array([
        [0.5, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.5],
        [0.5, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.5],
    ])
    stationary = np.full(4, 0.25)

    # Try multiple initializations because BA can converge to different
    # stationary points. Keep the encoder with the lowest objective.
    best_encoder = None
    best_info = None

    for seed in range(n_seeds):
        encoder, info = blahut_arimoto(
            transition=transition,
            stationary=stationary,
            n_reduced_states=2,
            beta=beta,
            seed=seed,
        )
        if best_info is None or info["objective"] < best_info["objective"]:
            best_encoder = encoder
            best_info = info

    assert best_encoder is not None and best_info is not None

    # The doubling map itself, plotted over [0, 1).
    x = np.linspace(0.0, 1.0, 1000, endpoint=False)
    y = np.mod(2.0 * x, 1.0)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Doubling map: xₜ₊₁ = 2xₜ mod 1",
            "Exact transition matrix P",
            "Learned encoder q(h|x)",
            "Information retained",
        ),
        specs=[[{}, {}], [{}, {}]],
    )

    fig.add_trace(
        go.Scatter(x=x, y=y, mode="lines", name="T(x)", line=dict(color="grey")),
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text="xₜ", row=1, col=1)
    fig.update_yaxes(title_text="xₜ₊₁", row=1, col=1)

    bin_labels = ["bin 0", "bin 1", "bin 2", "bin 3"]
    fig.add_trace(
        go.Heatmap(
            z=transition,
            x=bin_labels,
            y=bin_labels,
            zmin=0,
            zmax=1,
            colorscale="Blues",
            text=transition,
            texttemplate="%{text:.1f}",
            showscale=False,
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="Next-state bin", row=1, col=2)
    fig.update_yaxes(title_text="Current-state bin", autorange="reversed", row=1, col=2)

    fig.add_trace(
        go.Heatmap(
            z=best_encoder,
            x=["h = 0", "h = 1"],
            y=bin_labels,
            zmin=0,
            zmax=1,
            colorscale="Viridis",
            text=np.round(best_encoder, 2),
            texttemplate="%{text}",
            showscale=False,
        ),
        row=2,
        col=1,
    )
    fig.update_xaxes(title_text="Reduced state h", row=2, col=1)
    fig.update_yaxes(title_text="Full-state bin x", autorange="reversed", row=2, col=1)

    fig.add_trace(
        go.Bar(
            x=["I(H;X)", "I(H;Xₜ₊Δt)"],
            y=[best_info["I_HX"], best_info["I_HY"]],
            marker_color=["grey", "darkslateblue"],
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    fig.update_yaxes(title_text="Mutual information (bits)", row=2, col=2)

    fig.update_layout(
        title=f"Predictive Information Bottleneck on the doubling map (β={beta})",
        template="plotly_white",
        height=750,
    )

    return best_encoder, best_info, fig


if __name__ == "__main__":

    # --- TEST ULAM --- #
    result, figure = compare_ulam_with_exact(n_samples=2000, seed=1)

    print("Exact matrix:")
    print(np.array([[0.5, 0.5], [0.5, 0.5]]))
    print("\nUlam estimate:")
    print(result.transition)
    print("\nEstimated stationary distribution:")
    print(result.stationary)

    figure.show()

    eig = ulam_eigendelements(result)

    print("Eigenvalues:", eig.eigenvalues)
    # print("Right eigenvectors (columns):\n", eig.right_eigenvectors)
    # print("Left eigenvectors (columns):\n", eig.left_eigenvectors)


    # --- TEST IB --- # 
    encoder, info, figure = test_ib_on_four_bin_doubling_map()

    print("Learned encoder p(h|x):\n", np.round(encoder, 3))
    print("Information values:", info)
    figure.show()