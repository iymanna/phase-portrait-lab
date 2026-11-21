"""Draw phase portraits for planar linear systems z' = A z."""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

PRESETS = {
    "node": ([[-1.0, 0.0], [0.0, -0.4]], "#115a91"),
    "saddle": ([[0.8, 0.0], [0.0, -0.6]], "#9c4027"),
    "spiral": ([[-0.35, -1.15], [1.15, -0.35]], "#633ba4"),
    "center": ([[0.0, -1.0], [1.0, 0.0]], "#157064"),
}

PAPER = "#f7f5ef"
INK = "#162c40"
MUTED = "#475a69"
WINDOW = 3
SPECTRAL_TOLERANCE = 1e-9


def format_eigenvalues(values):
    """Display a conjugate pair with the usual mathematical notation."""
    first = values[0]
    if first.imag:
        magnitude = abs(first.imag)
        coefficient = "" if magnitude == 1 else f"{magnitude:.5g}"
        return f"{first.real:.5g} ± {coefficient}i" if first.real else f"±{coefficient}i"
    return ", ".join(f"{value.real:.5g}" for value in values)


def spectrum(matrix):
    """Return eigenvalues and classify the origin on the matrix's scale."""
    scale = np.max(np.abs(matrix))
    scaled = matrix / scale if scale else matrix
    values = np.linalg.eigvals(scaled)
    real = values.real
    tol = SPECTRAL_TOLERANCE

    if np.all(np.abs(real) <= tol) and np.all(np.abs(values.imag) > tol):
        label = "Center"
    elif np.any(np.abs(real) <= tol):
        label = "Degenerate or near zero"
    elif real.min() < 0 < real.max():
        label = "Saddle"
    else:
        stability = "Stable" if real.max() < 0 else "Unstable"
        shape = "spiral" if np.any(np.abs(values.imag) > tol) else "node"
        label = f"{stability} {shape}"
    return values * scale, label


def integrate(matrix, initial_states, duration):
    """Advance each initial state using fixed-step fourth-order Runge–Kutta."""
    scale = float(np.max(np.abs(matrix)))
    extent = scale * duration
    if extent > 40:
        raise ValueError("matrix entries are large for this duration. Try a shorter --duration.")

    steps = max(240, math.ceil(100 * extent))
    step_matrix = (duration / steps) * matrix
    states = np.empty((steps + 1, *initial_states.shape))
    states[0] = initial_states

    for index in range(steps):
        state = states[index]
        k1 = state @ step_matrix.T
        k2 = (state + k1 / 2) @ step_matrix.T
        k3 = (state + k2 / 2) @ step_matrix.T
        k4 = (state + k3) @ step_matrix.T
        states[index + 1] = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return states


def draw_portrait(axes, matrix, color, duration):
    """Add a unit-length direction field and forward-time trajectories."""
    eigenvalues, label = spectrum(matrix)
    if label == "Center":
        seeds = np.array([[0.65, 0.0], [0.0, 1.35], [-2.2, 0.0], [0.0, -2.65]])
    else:
        angles = np.arange(8) * (np.pi / 4)
        seeds = 2.2 * np.column_stack((np.cos(angles), np.sin(angles)))
        seeds[np.abs(seeds) < 1e-12] = 0

    coordinates = np.linspace(-2.75, 2.75, 15)
    x, y = np.meshgrid(coordinates, coordinates)
    scale = np.max(np.abs(matrix))
    a, b, c, d = (matrix / scale if scale else matrix).ravel()
    u, v = a * x + b * y, c * x + d * y
    speed = np.hypot(u, v)
    safe_speed = np.where(speed == 0, 1, speed)
    axes.quiver(
        x, y, np.ma.masked_where(speed == 0, u / safe_speed),
        np.ma.masked_where(speed == 0, v / safe_speed),
        color="#9caeb9", angles="xy", scale_units="xy", scale=5.6,
        pivot="middle", width=0.0032, zorder=1,
    )

    states = integrate(matrix, seeds, duration)
    for curve in states.transpose(1, 0, 2):
        visible = np.max(np.abs(curve), axis=1) <= 2 * WINDOW
        points = np.where(visible[:, None], curve, np.nan)
        axes.plot(*points.T, color=color, linewidth=1.75, zorder=2)
    axes.scatter(
        *seeds.T, facecolor="white", edgecolor=color,
        s=23, linewidth=1.35, zorder=3,
    )
    axes.scatter(0, 0, s=20, color=INK, zorder=4)

    axes.set(
        xlim=(-WINDOW, WINDOW), ylim=(-WINDOW, WINDOW), aspect="equal",
        xticks=[-3, -2, -1, 0, 1, 2, 3], yticks=[-3, -2, -1, 0, 1, 2, 3],
        facecolor="white",
    )
    axes.set_axisbelow(True)
    axes.grid(color="#e9edf0", linewidth=0.8)
    axes.axhline(0, color="#c5ced4", linewidth=0.9, zorder=0)
    axes.axvline(0, color="#c5ced4", linewidth=0.9, zorder=0)
    for spine in axes.spines.values():
        spine.set_color("#d4dce0")
    axes.tick_params(length=0, pad=5, labelsize=8, colors=MUTED)
    axes.text(1.025, -0.005, "x", transform=axes.transAxes, color=MUTED, fontsize=10)
    axes.text(-0.033, 1.027, "y", transform=axes.transAxes, color=MUTED, fontsize=10, ha="right")
    axes.text(
        0, 1.085, label, transform=axes.transAxes, color=color,
        fontsize=16, fontweight="bold",
    )
    axes.text(
        0, 1.025, f"Eigenvalues  {format_eigenvalues(eigenvalues)}",
        transform=axes.transAxes, color=MUTED, fontsize=10,
    )


def render(systems, duration, output):
    """Save a four-panel gallery or a portrait of one system as a PNG."""
    if len(systems) == 4:
        fig = plt.figure(figsize=(9, 9), facecolor=PAPER)
        panels = [
            (0.105, 0.49, 0.335, 0.335), (0.565, 0.49, 0.335, 0.335),
            (0.105, 0.08, 0.335, 0.335), (0.565, 0.08, 0.335, 0.335),
        ]
    else:
        fig = plt.figure(figsize=(6.5, 6.5), facecolor=PAPER)
        panels = [(0.15, 0.125, 0.70, 0.70)]

    try:
        fig.text(0.08, 0.965, "Phase Portrait Lab", va="top", fontsize=26,
                 fontweight="bold", color=INK)
        fig.text(0.08, 0.912, r"Planar linear systems  ·  $\mathbf{z}' = A\mathbf{z}$",
                 va="top", fontsize=12, color=MUTED)
        for (matrix, color), panel in zip(systems, panels):
            draw_portrait(fig.add_axes(panel), np.asarray(matrix), color, duration)
        fig.text(0.08, 0.027, "Open circles mark initial states. Arrows point forward in time.",
                 fontsize=9, color=MUTED)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, format="png")
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    system = parser.add_mutually_exclusive_group()
    system.add_argument("--preset", choices=PRESETS, help="draw one preset (default: all four)")
    system.add_argument(
        "--matrix", type=float, nargs=4, metavar=("A", "B", "C", "D"),
        help="use the matrix [[A, B], [C, D]]",
    )
    parser.add_argument("--duration", type=float, default=8.0, help="positive final time (default: 8)")
    parser.add_argument("--output", type=Path, default=Path("outputs/phase-portraits.png"))
    args = parser.parse_args()

    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration must be finite and positive")
    if args.output.suffix.lower() != ".png":
        parser.error("--output must end in .png")

    if args.matrix is not None:
        matrix = np.array(args.matrix).reshape(2, 2)
        if not np.isfinite(matrix).all():
            parser.error("--matrix must contain only finite numbers")
        systems = [(matrix, PRESETS["node"][1])]
    elif args.preset:
        systems = [PRESETS[args.preset]]
    else:
        systems = list(PRESETS.values())

    try:
        render(systems, args.duration, args.output)
    except (ValueError, OSError) as error:
        parser.error(str(error))

    for matrix, _ in systems:
        values, label = spectrum(np.asarray(matrix))
        print(f"{label}: eigenvalues {format_eigenvalues(values)}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
