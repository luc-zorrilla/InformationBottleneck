"""
Simulate Dynamical Systems (DS) and use Information Bottleneck (IB) to reduce dimensionality
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import sample_colorscale

from StochasticDynamics import (
    simulate_langevin,
    four_well_force,
    four_well_potential,
    brownian_noise,
    plot_trajectories,
    plot_trajectory_histogram,
)
from InformationBottleneck import (
    ulam_approximation,
    ulam_eigendelements,
    blahut_arimoto,
    information_quantities,
)

def plot_results(
    times: np.ndarray,
    trajectory: np.ndarray,
    bin_edges: np.ndarray,
    transition: np.ndarray,
    encoder: np.ndarray,
    potential_params: dict,
) -> go.Figure:
    """Plot potential, trajectory, Ulam transition matrix, and encoder."""
    x_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    potential = four_well_potential(x_centers, potential_params)
    bin_labels = [f"{i}" for i in range(len(x_centers))]

    # Plot a subset of the trajectory so the overview remains readable.
    stride = max(1, len(times) // 10_000)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Four-well potential",
            "Sample Langevin trajectory",
            "Ulam transition matrix",
            "Learned encoder q(h|x)",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=x_centers,
            y=potential,
            mode="lines",
            line=dict(color="grey"),
            name="V(x)",
        ),
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text="Position x", row=1, col=1)
    fig.update_yaxes(title_text="V(x)", row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=times[::stride],
            y=trajectory[::stride],
            mode="lines",
            line=dict(color="rgba(70, 70, 70, 0.45)", width=1),
            name="x(t)",
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="Time", row=1, col=2)
    fig.update_yaxes(title_text="Position x", row=1, col=2)

    fig.add_trace(
        go.Heatmap(
            z=transition,
            x=bin_labels,
            y=bin_labels,
            colorscale="Blues",
            zmin=0,
            zmax=1,
            colorbar=dict(title="P(i → j)", x=0.47),
        ),
        row=2,
        col=1,
    )
    fig.update_xaxes(title_text="Next-state bin", row=2, col=1)
    fig.update_yaxes(
        title_text="Current-state bin",
        autorange="reversed",
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            z=encoder,
            x=[f"h={j}" for j in range(encoder.shape[1])],
            y=bin_labels,
            colorscale="Viridis",
            zmin=0,
            zmax=1,
            colorbar=dict(title="q(h|x)", x=1.0),
        ),
        row=2,
        col=2,
    )
    fig.update_xaxes(title_text="Reduced state h", row=2, col=2)
    fig.update_yaxes(
        title_text="Full-state bin x",
        autorange="reversed",
        row=2,
        col=2,
    )

    fig.update_layout(
        title="Predictive Information Bottleneck for four-well Langevin dynamics",
        template="plotly_white",
        height=850,
    )
    return fig



def main() -> None:
    base_params = {
        "A": 10,
        "a": 0.5,
        "b": 1.0,
    }

    n_traj = int(1e1)
    n_repeats = int(1e1)
    starts = np.linspace(-1, 1, n_traj)
    initial_batch = np.repeat(starts, n_repeats)[:, None]
    periodic_bounds=(-1.0, 1.0)
    
    dt = 1e0
    n_steps = 200_000
    burn_in_steps = n_steps//10
    beta = 1e6
    n_states_to_test = range(1,10)
    sigma_values = np.linspace(0.05, 0.4, 8)

    bin_edges = np.linspace(-1, 1, 1000)
    n_h_values = list(n_states_to_test)

    # Store all results, indexed by sigma and then number of reduced states.
    results_by_sigma = {}

    for sigma in sigma_values:
        model_params = {**base_params, "sigma": float(sigma)}

        times, trajectories = simulate_langevin(
            initial_positions=initial_batch,
            dt=dt,
            n_steps=n_steps,
            force=four_well_force,
            noise_increment=brownian_noise,
            params=model_params,
            seed=10,
            periodic_bounds=(-1.0, 1.0),
        )

        # plot_trajectories(times, trajectories).show()
        # plot_trajectory_histogram(trajectories).show()

        # Shape: (n_trajectories, n_steps + 1)
        x_all = trajectories[:, :, 0].T

        ulam = ulam_approximation(
            x_all,
            bin_edges,
            burn_in=burn_in_steps,
        )
        
        results_for_sigma = {}

        print(f"Exact IB computation...")
        for n_h in n_h_values:

            print(f"n_h = {n_h}")

            encoder, info = blahut_arimoto(
                transition=ulam.transition,
                stationary=ulam.stationary,
                n_reduced_states=n_h,
                beta=beta,
                seed=4,
            )

            p_h = ulam.stationary @ encoder
            positive = p_h > 0
            h_entropy = -np.sum(
                p_h[positive] * np.log(p_h[positive])
            )

            # --- DISTILLATION ---
            # One entry per full-state bin: the most likely reduced state for that bin.
            h_of_x = np.argmax(encoder, axis=1)

            # # hard_encoder[x, h] = 1 if h is the most likely state for x, else 0
            # hard_encoder = np.eye(encoder.shape[1])[h_of_x]

            # hard_info = information_quantities(
            #     transition=ulam.transition,
            #     stationary=ulam.stationary,
            #     encoder=hard_encoder,
            # )

            results_for_sigma[n_h] = {
                "encoder": encoder,
                "h_of_x": h_of_x,
                "p_h": p_h,
                "I_HX": info["I_HX"],
                "I_HY": info["I_HY"],
                "H_H": h_entropy,
                "objective": info["objective"],
                "transition": ulam.transition,
                "stationary": ulam.stationary,
            }

            # results_for_sigma[n_h] = { # TODO: check whether it works and come back to continuous encoder otherwise
            #     "encoder": hard_encoder,
            #     "h_of_x": h_of_x,
            #     "p_h": p_h,
            #     "I_HX": hard_info["I_HX"],
            #     "I_HY": hard_info["I_HY"],
            #     "H_H": h_entropy,
            #     "objective": info["objective"],
            #     "transition": ulam.transition,
            #     "stationary": ulam.stationary,
            # }


        print(f"Done.")

        results_by_sigma[float(sigma)] = results_for_sigma

    # Potential does not depend on sigma.
    x_plot = bin_edges
    v_plot = four_well_potential(x_plot, base_params)

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "Four-well potential",
            "Predictive information",
            "Compression information",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=x_plot,
            y=v_plot,
            mode="lines",
            line=dict(color="grey"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text="Position x", row=1, col=1)
    fig.update_yaxes(title_text="V(x)", row=1, col=1)

    colorscale = "Viridis"
    sigma_min = float(sigma_values.min())
    sigma_max = float(sigma_values.max())

    for sigma in sigma_values:
        results = results_by_sigma[float(sigma)]
        color_position = (sigma - sigma_min) / (sigma_max - sigma_min)
        color = sample_colorscale(colorscale, [color_position])[0]

        i_hy = [
            results[n_h]["I_HY"]
            for n_h in n_h_values
        ]
        i_hx = [
            results[n_h]["I_HX"]
            for n_h in n_h_values
        ]

        # Show a colorbar on the predictive-information subplot.
        # Only one trace needs to display it.
        show_scale = np.isclose(sigma, sigma_values[-1])

        fig.add_trace(
            go.Scatter(
                x=n_h_values,
                y=i_hy,
                mode="lines",
                line=dict(color=color, width=1.5),
                showlegend=False,
                hovertemplate=(
                    f"sigma={sigma:.3f}<br>"
                    "n_H=%{x}<br>I(H;X_next)=%{y:.4f} bits"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=n_h_values,
                y=i_hx,
                mode="lines",
                line=dict(color=color, width=1.5),
                showlegend=False,
                hovertemplate=(
                    f"sigma={sigma:.3f}<br>"
                    "n_H=%{x}<br>I(H;X)=%{y:.4f} bits"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=3,
        )

        if show_scale:
            # Invisible points provide a continuous sigma colorbar.
            fig.add_trace(
                go.Scatter(
                    x=[None, None],
                    y=[None, None],
                    mode="markers",
                    marker=dict(
                        color=[sigma_min, sigma_max],
                        cmin=sigma_min,
                        cmax=sigma_max,
                        colorscale=colorscale,
                        showscale=True,
                        colorbar=dict(title="σ"),
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=2,
            )

    fig.update_xaxes(
        title_text="Number of reduced states",
        dtick=1,
        row=1,
        col=2,
    )
    fig.update_yaxes(
        title_text="I(H; X_next) (bits)",
        row=1,
        col=2,
    )
    fig.update_xaxes(
        title_text="Number of reduced states",
        dtick=1,
        row=1,
        col=3,
    )
    fig.update_yaxes(
        title_text="I(H; X) (bits)",
        row=1,
        col=3,
    )

    fig.update_layout(
        title=f"Information Bottleneck results (β={beta:g})",
        template="plotly_white",
        showlegend=False,
        height=450,
    )
    fig.show()


if __name__ == "__main__":
    main()
