import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors
from scipy.special import ndtr


# Langevin equation: dx = (mu*x - x**3)dt + sigma*dW
SIGMA = 0.1
DT = 2e-3
XMIN, XMAX = -1.0, 1.0
N_BINS = 100

MU_VALUES = [-1, 1, 10]
MU_GRID = np.linspace(-1.0, 1.0, 100)
DELAYS = np.logspace(-1, 1, 100)


def potential(x, mu):
    return (1/4) * (mu - x**2) ** 2

def make_transition(mu, sigma=SIGMA, dt=DT, n_bins=N_BINS):
    """Euler–Maruyama transition matrix on a finite state grid."""
    edges = np.linspace(XMIN, XMAX, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    sd = sigma * np.sqrt(dt)

    if sd == 0:
        # Deterministic transition, assigned to the destination bin.
        destinations = centers + drift(centers, mu) * dt
        indices = np.clip(
            np.digitize(destinations, edges[1:-1]), 0, n_bins - 1
        )
        P = np.zeros((n_bins, n_bins))
        P[np.arange(n_bins), indices] = 1.0
        return centers, P

    means = centers + drift(centers, mu) * dt
    z_hi = (edges[1:][None, :] - means[:, None]) / sd
    z_lo = (edges[:-1][None, :] - means[:, None]) / sd
    P = ndtr(z_hi) - ndtr(z_lo)

    # Assign probability falling beyond the grid to the edge bins.
    P[:, 0] += ndtr((XMIN - means) / sd)
    P[:, -1] += 1.0 - ndtr((XMAX - means) / sd)
    P /= P.sum(axis=1, keepdims=True)
    return centers, P


def stationary_distribution(P):
    """Return the stationary row distribution p satisfying p @ P = p."""
    values, vectors = np.linalg.eig(P.T)
    index = np.argmin(np.abs(values - 1.0))
    p = np.real(vectors[:, index])
    if p.sum() < 0:
        p = -p
    p = np.maximum(p, 0.0)
    return p / p.sum()


def mutual_information_from_joint(joint):
    """Mutual information in nats for a joint probability matrix."""
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    denom = px @ py
    mask = joint > 0
    return np.sum(joint[mask] * np.log(joint[mask] / denom[mask]))


def lagged_mutual_information(p, P, delay):
    """I(X_t; X_{t+delay}) for the discretized Markov process."""
    lag_steps = max(1, int(round(delay / DT)))
    P_lag = np.linalg.matrix_power(P, lag_steps)
    joint = p[:, None] * P_lag
    return mutual_information_from_joint(joint)


def stochastic_trajectories(mu, sigma=SIGMA, dt=DT, duration=10.0,
                            n_trajectories=18, seed=0):
    """Simulate independent Langevin paths from uniformly spaced starts."""
    rng = np.random.default_rng(seed)
    n_steps = int(duration / dt)
    times = np.arange(n_steps + 1) * dt
    x = np.empty((n_trajectories, n_steps + 1))

    x[:, 0] = np.linspace(-1.5, 1.5, n_trajectories)
    for k in range(n_steps):
        x[:, k + 1] = (
            x[:, k]
            + sigma * np.sqrt(dt) * rng.normal(size=n_trajectories)
        )

    return times, x


def make_figure_2a():
    """Figure a: three rows (i, ii, iii), one column per mu."""
    fig = make_subplots(
        rows=3,
        cols=3,
        subplot_titles=[
            f"(i) Potential, μ={mu:g}" for mu in MU_VALUES
        ] + [
            f"(ii) Trajectories, μ={mu:g}" for mu in MU_VALUES
        ] + [
            f"(iii) I(X;Y) vs Δt, μ={mu:g}" for mu in MU_VALUES
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
        shared_yaxes=True,
    )

    for col, mu in enumerate(MU_VALUES, start=1):
        # a-i: potential
        xplot = np.linspace(XMIN, XMAX, 100) * np.max((1, 1.5*np.abs(mu)))
        Uplot = potential(xplot, mu)
        Uplot -= np.min(Uplot)
        Uplot /= np.max(Uplot)
        # xplot /= np.max(xplot)
        fig.add_trace(
            go.Scatter(
                x=xplot,
                y=Uplot,
                mode="lines",
                showlegend=False,
                line=dict(color="darkorange"),
            ),
            row=1, col=col,
        )

        # a-ii: noisy Langevin trajectories
        times, paths = stochastic_trajectories(mu, seed=100 + col, n_trajectories=500)
        for i in range(paths.shape[0]):
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=paths[i]/np.max(np.abs(paths[i])),
                    mode="lines",
                    line=dict(color="black", width=1),
                    opacity=0.04,
                    showlegend=False,
                ),
                row=2, col=col,
            )

        # a-iii: stationary present–future information versus delay
        _, P = make_transition(mu)
        p = stationary_distribution(P)
        infos = [lagged_mutual_information(p, P, delay) for delay in DELAYS]
        fig.add_trace(
            go.Scatter(
                x=DELAYS,
                y=infos,
                mode="lines+markers",
                showlegend=False,
                line=dict(color="royalblue"),
            ),
            row=3, col=col,
        )

    for col in range(1, 4):
        fig.update_xaxes(title_text="x", row=1, col=col)
        fig.update_yaxes(title_text="V(x)", row=1, col=col)
        fig.update_xaxes(title_text="time", row=2, col=col)
        fig.update_yaxes(title_text="x(t)", row=2, col=col)
        fig.update_xaxes(title_text="Δt", row=3, col=col)
        fig.update_yaxes(title_text="I(X;Y), nats", row=3, col=col)

    fig.update_layout(
        title=f"Figure a — σ={SIGMA:g}, dt={DT:g}",
        height=900,
        width=1200,
    )
    fig.show()

def make_figure_2b():
    """Plot I(X;Y) versus mu, with line color encoding log-scaled delta_t."""

    if np.any(np.asarray(DELAYS) <= 0):
        raise ValueError("DELAYS must all be positive for a log-scaled colorbar.")

    log_delays = np.log10(DELAYS)
    cmin, cmax = log_delays.min(), log_delays.max()
    colorscale = "RdBu" # "Viridis"

    fig = go.Figure()

    # One line per delay, colored according to log10(delta_t).
    for delay, log_delay in zip(DELAYS, log_delays):
        z = []

        for mu in MU_GRID:
            _, P = make_transition(mu)
            p = stationary_distribution(P)
            z.append(lagged_mutual_information(p, P, delay))

        color_fraction = (
            0.5 if cmax == cmin else (log_delay - cmin) / (cmax - cmin)
        )
        line_color = plotly.colors.sample_colorscale(
            colorscale, [color_fraction]
        )[0]

        fig.add_trace(
            go.Scatter(
                x=MU_GRID,
                y=z,
                mode="lines",
                line=dict(color=line_color, width=2),
                name=f"Δt={delay:g}",
                showlegend=False,
                hovertemplate=(
                    "μ=%{x:.3f}<br>"
                    "I(X;Y)=%{y:.4f} nats"
                    f"<br>Δt={delay:g}<extra></extra>"
                ),
            )
        )

    # Invisible trace supplies a continuous colorbar in log10(delta_t) space.
    tick_delays = np.geomspace(DELAYS.min(), DELAYS.max(), 5)
    tick_log_delays = np.log10(tick_delays)

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            showlegend=False,
            marker=dict(
                color=[cmin, cmax],
                cmin=cmin,
                cmax=cmax,
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(
                    title="Δt (log scale)",
                    tickmode="array",
                    tickvals=tick_log_delays,
                    ticktext=[f"{d:.3g}" for d in tick_delays],
                ),
            ),
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title="Figure b — Present–future information",
        xaxis_title="μ",
        yaxis_title="I(X;Y), nats",
        width=800,
        height=550,
    )
    fig.show()


def make_figure_2c(n_eigenvalues=10):
    """Figure c: real parts of leading generator eigenvalues versus mu."""
    fig = go.Figure()

    for mu in MU_GRID:
        _, P = make_transition(mu)
        eigenvalues = np.linalg.eigvals(P)

        # P approximates the one-step transfer operator.
        # Convert its eigenvalues to continuous-time generator rates.
        eigenvalues = eigenvalues[np.argsort(-np.abs(eigenvalues))]
        eigenvalues = eigenvalues[:n_eigenvalues]
        # rates = np.log(eigenvalues.astype(complex)) / DT

        fig.add_trace(
            go.Scatter(
                x=np.full(len(eigenvalues), mu),
                y=np.real(eigenvalues),
                mode="markers",
                line= dict(color = "black"),
                showlegend=False,
            )
        )

    fig.update_layout(
        title="Figure c — Transfer-operator spectrum",
        xaxis_title="μ",
        yaxis_title="Re(λᵢ)",
        width=800,
        height=550,
    )
    fig.show()

# Four symmetric minima at ±a and ±b.
CENTRAL_BARRIER = 1.5
A_INNER = 0.85
B_OUTER = 1.20
A_SCALE = CENTRAL_BARRIER / (A_INNER**4 * B_OUTER**4)

# Euler–Maruyama and spatial discretization.
DT = 0.01
LAG_TIME = 1.0
XMIN, XMAX = -1.6, 1.6
N_BINS = 80

# Illustrative low/high noise levels; adjust to match the visual you want.
NOISE_LEVELS = [0.05, 0.15]
SIGMA_GRID = np.linspace(0.05, 0.2, 40)
ALPHABET_SIZES = range(1, 7)
BETA = 30.0


# def potential(x):
#     """Symmetric four-well potential."""
#     return A_SCALE * ((x**2 - A_INNER**2) * (x**2 - B_OUTER**2)) ** 2


def force(x):
    """Force -dV/dx for the four-well potential."""
    return (
        -4.0 * A_SCALE * x
        * (x**2 - A_INNER**2)
        * (x**2 - B_OUTER**2)
        * (2.0 * x**2 - A_INNER**2 - B_OUTER**2)
    )


def one_step_transition(sigma):
    """Euler–Maruyama transition matrix on a discretized state space."""
    edges = np.linspace(XMIN, XMAX, N_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = centers + force(centers) * DT
    sd = sigma * np.sqrt(DT)

    z_hi = (edges[1:][None, :] - means[:, None]) / sd
    z_lo = (edges[:-1][None, :] - means[:, None]) / sd
    P = ndtr(z_hi) - ndtr(z_lo)

    # Put probability outside the finite grid into the edge bins.
    P[:, 0] += ndtr((XMIN - means) / sd)
    P[:, -1] += 1.0 - ndtr((XMAX - means) / sd)
    P /= P.sum(axis=1, keepdims=True)

    return centers, P


def stationary_distribution(P):
    """Stationary row distribution p satisfying p @ P = p."""
    values, vectors = np.linalg.eig(P.T)
    index = np.argmin(np.abs(values - 1.0))
    p = np.real(vectors[:, index])
    if p.sum() < 0:
        p = -p
    p = np.maximum(p, 0.0)
    return p / p.sum()


def mutual_information(joint):
    """Mutual information in nats for a joint probability matrix."""
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    denom = px @ py
    mask = joint > 0
    return np.sum(joint[mask] * np.log(joint[mask] / denom[mask]))


def ba_encoder(p_x, p_y_given_x, n_codes, beta=BETA,
            seed=0, max_iter=5000, tol=1e-9):
    """Blahut–Arimoto IB encoder q(h|x)."""
    rng = np.random.default_rng(seed)
    n_x = len(p_x)
    tiny = np.finfo(float).tiny

    q_h_given_x = rng.random((n_x, n_codes))
    q_h_given_x /= q_h_given_x.sum(axis=1, keepdims=True)

    for _ in range(max_iter):
        p_h = p_x @ q_h_given_x
        p_y_given_h = (
            (p_x[:, None] * q_h_given_x).T @ p_y_given_x
        ) / np.maximum(p_h[:, None], tiny)

        log_ratio = (
            np.log(np.maximum(p_y_given_x[:, None, :], tiny))
            - np.log(np.maximum(p_y_given_h[None, :, :], tiny))
        )
        kl = np.sum(p_y_given_x[:, None, :] * log_ratio, axis=2)

        logits = np.log(np.maximum(p_h[None, :], tiny)) - beta * kl
        logits -= logits.max(axis=1, keepdims=True)
        updated = np.exp(logits)
        updated /= updated.sum(axis=1, keepdims=True)

        if np.max(np.abs(updated - q_h_given_x)) < tol:
            q_h_given_x = updated
            break
        q_h_given_x = updated

    return q_h_given_x


def best_encoder(p_x, p_y_given_x, n_codes, n_starts=4):
    """Try several initializations and retain the best IB objective."""
    best = None

    for seed in range(n_starts):
        q = ba_encoder(
            p_x, p_y_given_x, n_codes=n_codes,
            beta=BETA, seed=seed
        )
        p_h = p_x @ q
        p_y_given_h = (
            (p_x[:, None] * q).T @ p_y_given_x
        ) / np.maximum(p_h[:, None], np.finfo(float).tiny)

        i_xh = mutual_information(p_x[:, None] * q)
        i_hy = mutual_information(p_h[:, None] * p_y_given_h)
        objective = i_xh - BETA * i_hy

        if best is None or objective < best[0]:
            best = (objective, i_hy)

    # Figure panel b is in bits.
    return best[1] / np.log(2.0)


def calculate_for_noise(sigma):
    """Return encoder information and transfer spectrum at one noise level."""
    centers, P = one_step_transition(sigma)
    lag_steps = max(1, round(LAG_TIME / DT))
    P_lag = np.linalg.matrix_power(P, lag_steps)
    p_x = stationary_distribution(P_lag)

    info_bits = []
    for n_codes in ALPHABET_SIZES:
        info_bits.append(
            best_encoder(p_x, P_lag, n_codes=n_codes)
        )

    eigenvalues = np.linalg.eigvals(P_lag)
    eigenvalues = np.real(eigenvalues)
    eigenvalues.sort()
    eigenvalues = eigenvalues[::-1]

    return centers, info_bits, eigenvalues

def spectrum_for_sigma(sigma, n_show=8):
    _, P = one_step_transition(sigma)
    lag_steps = max(1, round(LAG_TIME / DT))
    P_lag = np.linalg.matrix_power(P, lag_steps)

    eigenvalues = np.linalg.eigvals(P_lag)
    eigenvalues = np.real(eigenvalues)
    return np.sort(eigenvalues)[::-1][:n_show]

def conditional_mutual_information_from_encoder(p_x, P, q):
    """
    I(H_prev; H_next | H_now), in nats.

    p_x: stationary distribution over discretized states
    P:   one-lag transition matrix over discretized states
    q:   encoder probabilities q(h | x), shape (n_x, n_codes)
    """
    n_codes = q.shape[1]
    p_hhh = np.zeros((n_codes, n_codes, n_codes))

    # Sum over the current physical state x_now. Given x_now,
    # the previous and next physical states are conditionally independent.
    for j in range(len(p_x)):
        p_hprev_and_xnow = (p_x * P[:, j]) @ q
        p_hnext_given_xnow = P[j] @ q

        p_hhh += np.einsum(
            "a,b,c->abc",
            p_hprev_and_xnow,
            q[j],
            p_hnext_given_xnow,
        )

    p_hhh /= p_hhh.sum()

    # I(A; C | B) = sum_{a,b,c} p(a,b,c) log[p(a,c|b)/(p(a|b)p(c|b))]
    p_ab = p_hhh.sum(axis=2)
    p_bc = p_hhh.sum(axis=0)
    p_b = p_hhh.sum(axis=(0, 2))

    denom = np.zeros_like(p_hhh)
    for b in range(n_codes):
        if p_b[b] > 0:
            denom[:, b, :] = np.outer(p_ab[:, b], p_bc[b, :]) / p_b[b]

    mask = (p_hhh > 0) & (denom > 0)
    return np.sum(p_hhh[mask] * np.log(p_hhh[mask] / denom[mask]))


def non_markovianity_for_noise(sigma):
    _, P = one_step_transition(sigma)
    lag_steps = max(1, round(LAG_TIME / DT))
    P_lag = np.linalg.matrix_power(P, lag_steps)
    p_x = stationary_distribution(P_lag)

    values = []
    for n_codes in ALPHABET_SIZES:
        q = ba_encoder(p_x, P_lag, n_codes=n_codes, beta=BETA)
        cmi_nats = conditional_mutual_information_from_encoder(p_x, P_lag, q)
        values.append(cmi_nats / np.log(2.0))  # bits

    return values

def make_figure_3():
    centers, info_low, eig_low = calculate_for_noise(NOISE_LEVELS[0])
    _, info_high, eig_high = calculate_for_noise(NOISE_LEVELS[1])

    sizes = list(ALPHABET_SIZES)
    gains_low = np.diff(info_low, prepend=0.0)
    gains_high = np.diff(info_high, prepend=0.0)

    fig = make_subplots(
        rows=1,
        cols=4,
        subplot_titles=[
            "(a) Four-well potential",
            "(b) Information vs. encoding size",
            "(c) Information gain",
            "(d) Transfer-operator eigenvalues",
        ],
        horizontal_spacing=0.10,
    )

    # (a) Potential.
    x = np.linspace(XMIN, XMAX, 700)
    fig.add_trace(
        go.Scatter(x=x, y=potential(x), mode="lines",
                line=dict(color="black"), showlegend=False),
        row=1, col=1,
    )
    fig.update_xaxes(title_text="x", row=1, col=1)
    fig.update_yaxes(title_text="V(x)", row=1, col=1)

    # (b) I(H; Y) in bits as alphabet size changes.
    for label, values, color in [
        (f"σ={NOISE_LEVELS[0]:g}", info_low, "firebrick"),
        (f"σ={NOISE_LEVELS[1]:g}", info_high, "lightsalmon"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=sizes, y=values, mode="lines+markers",
                name=label, line=dict(color=color),
            ),
            row=1, col=2,
        )
    fig.update_xaxes(title_text="N_H", row=1, col=2)
    fig.update_yaxes(title_text="I(H; X_future), bits", row=1, col=2)

    # (c) Incremental information gained by adding one code.
    for label, values, color in [
        (f"σ={NOISE_LEVELS[0]:g}", gains_low, "firebrick"),
        (f"σ={NOISE_LEVELS[1]:g}", gains_high, "lightsalmon"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=sizes, y=values, mode="lines+markers",
                name=label, line=dict(color=color),
                showlegend=False,
            ),
            row=1, col=3,
        )
    fig.update_xaxes(title_text="N_H", row=1, col=3)
    fig.update_yaxes(title_text="Information gain (bits)", row=1, col=3)

    # (d) Leading transfer-operator eigenvalues across a continuous sigma grid.
    spectra = np.array([
        spectrum_for_sigma(sigma, n_show=8)
        for sigma in SIGMA_GRID
    ])

    for k in range(spectra.shape[1]):
        fig.add_trace(
            go.Scatter(
                x=SIGMA_GRID,
                y=spectra[:, k],
                mode="lines+markers",
                name=f"Λ{k + 1}",
                line=dict(width=1.5),
                marker=dict(size=4),
            ),
            row=1, col=4,
        )

    fig.update_xaxes(title_text="Noise amplitude σ", row=1, col=4)
    fig.update_yaxes(title_text="Re(Λᵢ)", row=1, col=4)

    fig.update_layout(
        title=(
            f"Four-well information bottleneck "
            f"(Δt={LAG_TIME:g}, β={BETA:g})"
        ),
        width=1500,
        height=480,
        legend_title_text="Noise",
    )
    fig.show()

# Figure 4a

SIGMA = 0.09
ALPHABET_SIZES = np.arange(1, 40)

def make_figure_4a():
    """Plot I(H_{t-Δt}; H_{t+Δt} | H_t) versus alphabet size, in bits."""
    cmi_bits = non_markovianity_for_noise(SIGMA)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ALPHABET_SIZES,
            y=cmi_bits,
            mode="lines",
            line=dict(color="#303034", width=3),
            hovertemplate="Nₕ=%{x}<br>CMI=%{y:.4f} bits<extra></extra>",
        )
    )

    fig.update_layout(
        template="plotly_white",
        xaxis_title=r"$N_H$",
        yaxis_title=r"$I(H_{t-\Delta t}; H_{t+\Delta t}\mid H_t)$ (bits)",
        showlegend=False,
        width=700,
        height=400,
        margin=dict(l=70, r=25, t=25, b=60),
    )
    fig.update_xaxes(range=[0, 40], showline=True, linewidth=1, linecolor="black")
    fig.update_yaxes(rangemode="tozero", showline=True, linewidth=1, linecolor="black")

    fig.show()

import numpy as np
import pysindy as ps
import plotly.graph_objects as go


def fit_sindy_by_dimension(H_train, H_test, dt, dimensions=range(1, 7)):
    """Fit SINDy models to the first d latent coordinates.

    H_train and H_test are arrays shaped (n_trajectories, n_times, n_latent).
    Returns fitted models and normalized test prediction errors.
    """
    errors = []
    models = {}

    for d in dimensions:
        model = ps.SINDy(
            feature_library=ps.PolynomialLibrary(
                degree=3,
                include_bias=True,
            ),
            optimizer=ps.STLSQ(threshold=0.05),
            differentiation_method=ps.FiniteDifference(),
        )

        train = [trajectory[:, :d] for trajectory in H_train]
        test = [trajectory[:, :d] for trajectory in H_test]

        model.fit(train, t=dt, multiple_trajectories=True)

        trajectory_errors = []
        for trajectory in test:
            initial_state = trajectory[0]
            t = np.arange(len(trajectory)) * dt
            prediction = model.simulate(initial_state, t)

            scale = np.mean((trajectory - trajectory.mean(axis=0)) ** 2)
            mse = np.mean((prediction - trajectory) ** 2)
            trajectory_errors.append(mse / max(scale, 1e-12))

        models[d] = model
        errors.append(np.mean(trajectory_errors))

    return models, np.asarray(errors)


def make_figure_5d(dimensions, errors):
    """Plot normalized held-out SINDy prediction error versus latent dimension."""
    fig = go.Figure(
        go.Scatter(
            x=list(dimensions),
            y=errors,
            mode="lines+markers",
            line=dict(color="#303034", width=2.5),
            marker=dict(size=7),
            hovertemplate="dim(H)=%{x}<br>normalized error=%{y:.4g}<extra></extra>",
        )
    )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Latent dimension",
        yaxis_title="Normalized SINDy prediction error",
        showlegend=False,
        width=650,
        height=380,
    )
    fig.show()


if __name__ == "__main__":
    make_figure_2a()
    make_figure_2b()
    make_figure_2c()
    # make_figure_3()
    # make_figure_4a()