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

def ulam_approximation(
    trajectories: np.ndarray,
    bin_edges: np.ndarray,
    *,
    pseudocount: float = 0.0,
) -> UlamResult:
    """Estimate a discrete transfer operator from trajectory samples.

    Parameters
    ----------
    trajectories:
        Array shaped (n_trajectories, n_times), or a 1D time series.
        Each consecutive pair contributes one transition.
    bin_edges:
        Strictly increasing bin boundaries. Values outside the boundaries
        are ignored. Bins are [edge_i, edge_{i+1}), except the final bin,
        which includes its right edge.
    pseudocount:
        Nonnegative count added to every transition cell. Use a small
        positive value to avoid zero-probability transitions.

    Returns
    -------
    UlamResult:
        `transition[i, j]` is the estimated probability of transitioning
        from source bin i to target bin j. `stationary` is estimated by
        power iteration on that matrix.
    """
    trajectories = np.asarray(trajectories, dtype=float)
    bin_edges = np.asarray(bin_edges, dtype=float)

    if trajectories.ndim == 1:
        trajectories = trajectories[None, :]
    if trajectories.ndim != 2:
        raise ValueError("trajectories must be 1D or 2D.")
    if bin_edges.ndim != 1 or len(bin_edges) < 2:
        raise ValueError("bin_edges must be a 1D array with at least two edges.")
    if not np.all(np.diff(bin_edges) > 0):
        raise ValueError("bin_edges must be strictly increasing.")
    if pseudocount < 0:
        raise ValueError("pseudocount must be nonnegative.")

    n_bins = len(bin_edges) - 1
    counts = np.full((n_bins, n_bins), pseudocount, dtype=float)

    for trajectory in trajectories:
        if len(trajectory) < 2:
            continue

        source = np.searchsorted(bin_edges, trajectory[:-1], side="right") - 1
        target = np.searchsorted(bin_edges, trajectory[1:], side="right") - 1

        # Include values exactly equal to the final edge in the last bin.
        source[trajectory[:-1] == bin_edges[-1]] = n_bins - 1
        target[trajectory[1:] == bin_edges[-1]] = n_bins - 1

        valid = (
            (source >= 0) & (source < n_bins)
            & (target >= 0) & (target < n_bins)
        )
        np.add.at(counts, (source[valid], target[valid]), 1.0)

    row_sums = counts.sum(axis=1)
    transition = np.zeros_like(counts)
    observed_rows = row_sums > 0
    transition[observed_rows] = counts[observed_rows] / row_sums[observed_rows, None]

    # For unobserved bins, use a self-transition. This keeps P row-stochastic
    # without inventing transitions to other bins.
    unobserved = ~observed_rows
    transition[unobserved, unobserved] = 1.0

    stationary = stationary_distribution(transition)
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
        if np.linalg.norm(updated - distribution, ord=1) < tolerance:
            return updated / updated.sum()
        distribution = updated

    raise RuntimeError("Stationary-distribution iteration did not converge.")


def _normalize_probabilities(values: np.ndarray, axis: int) -> np.ndarray:
    """Normalize nonnegative values along an axis."""
    totals = values.sum(axis=axis, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("Cannot normalize a zero-mass distribution.")
    return values / totals


def predictive_information_bottleneck(
    transition: np.ndarray,
    stationary: np.ndarray,
    n_reduced_states: int,
    beta: float,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 10_000,
    seed: int | None = 0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Optimize the encoder q(h|x) for the predictive Information Bottleneck.

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
        Dictionary containing I(H;X), I(H;Y), and the objective in nats.
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
    active = stationary > 0
    px = stationary[active]
    p_y_given_x = transition[active]
    px = px / px.sum()

    rng = np.random.default_rng(seed)
    encoder = rng.random((len(px), n_reduced_states)) + 1e-3
    encoder = _normalize_probabilities(encoder, axis=1)

    eps = np.finfo(float).tiny

    for _ in range(max_iterations):
        # q(h) = sum_x p(x) q(h|x)
        p_h = px @ encoder

        # p(y|h) = sum_x p(x) q(h|x) p(y|x) / p(h)
        p_y_given_h = (
            (px[:, None] * encoder).T @ p_y_given_x
        ) / np.maximum(p_h[:, None], eps)

        # KL[p(y|x) || p(y|h)], for every x and h.
        log_p_y_given_x = np.log(np.maximum(p_y_given_x, eps))
        log_p_y_given_h = np.log(np.maximum(p_y_given_h, eps))
        kl = np.sum(
            p_y_given_x[:, None, :]
            * (log_p_y_given_x[:, None, :] - log_p_y_given_h[None, :, :]),
            axis=2,
        )

        # BA update: q(h|x) proportional to q(h) exp(-beta * KL).
        log_weights = np.log(np.maximum(p_h, eps))[None, :] - beta * kl
        log_weights -= np.max(log_weights, axis=1, keepdims=True)
        updated = np.exp(log_weights)
        updated = _normalize_probabilities(updated, axis=1)

        if np.max(np.abs(updated - encoder)) < tolerance:
            encoder = updated
            break
        encoder = updated

    # Reinsert any zero-occupancy states as arbitrary normalized rows.
    full_encoder = np.full((n_states, n_reduced_states), 1.0 / n_reduced_states)
    full_encoder[active] = encoder

    information = information_quantities(
        transition, stationary, full_encoder
    )
    information["objective"] = (
        information["I_HX"] - beta * information["I_HY"]
    )
    return full_encoder, information


def information_quantities(
    transition: np.ndarray,
    stationary: np.ndarray,
    encoder: np.ndarray,
) -> dict[str, float]:
    """Compute I(H;X) and I(H;Y) in nats for an encoder q(h|x)."""
    transition = np.asarray(transition, dtype=float)
    stationary = np.asarray(stationary, dtype=float)
    encoder = np.asarray(encoder, dtype=float)

    px_h = stationary[:, None] * encoder
    p_h = px_h.sum(axis=0)
    p_y = stationary @ transition
    p_hy = px_h.T @ transition

    eps = np.finfo(float).tiny

    # I(H;X) = sum_x,h p(x,h) log[p(h|x)/p(h)]
    ix_terms = np.zeros_like(px_h)
    valid_xh = px_h > 0
    ix_terms[valid_xh] = px_h[valid_xh] * np.log(
        (encoder / np.maximum(p_h[None, :], eps))[valid_xh]
    )
    i_hx = float(ix_terms.sum())

    # I(H;Y) = sum_h,y p(h,y) log[p(h,y)/(p(h)p(y))]
    denominator = p_h[:, None] * p_y[None, :]
    valid_hy = p_hy > 0
    iy_terms = np.zeros_like(p_hy)
    iy_terms[valid_hy] = p_hy[valid_hy] * np.log(
        (p_hy / np.maximum(denominator, eps))[valid_hy]
    )
    i_hy = float(iy_terms.sum())

    return {"I_HX": i_hx, "I_HY": i_hy}

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
    beta: float = 1.000001,
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
        encoder, info = predictive_information_bottleneck(
            transition=transition,
            stationary=stationary,
            n_reduced_states=2,
            beta=beta,
            seed=seed,
        )
        if best_info is None or info["objective"] < best_info["objective"]:
            best_encoder = encoder
            best_info = info
            best_info["I_HX"] /= np.log(2)
            best_info["I_HY"] /= np.log(2)

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

    print("Learned encoder q(h|x):\n", np.round(encoder, 3))
    print("Information values:", info)
    figure.show()