#!/usr/bin/env python3
"""Reconstruct GEO-array Georgia coverage from GAmap.mat and array_design.npz.

Run:
    python recreate_georgia_coverage_window.py --mat GAmap.mat --npz array_design.npz

This version opens interactive matplotlib windows instead of saving a PNG,
and it also shows the active element positions colored by phase.
"""
import argparse
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

RE_A = 6378137.0
E2 = 6.69437999014e-3
R_GEO = 42164170.0


def ecef(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    n = RE_A / np.sqrt(1 - E2 * np.sin(la) ** 2)
    return np.stack((
        n * np.cos(la) * np.cos(lo),
        n * np.cos(la) * np.sin(lo),
        n * (1 - E2) * np.sin(la)
    ), axis=-1)


def geometry(lat, lon, mask, sat_lon):
    points = ecef(lat, lon).reshape(-1, 3)
    sat = np.array([
        R_GEO * np.cos(np.radians(sat_lon)),
        R_GEO * np.sin(np.radians(sat_lon)),
        0.0
    ])
    d = points - sat
    R = np.linalg.norm(d, axis=1)
    dh = d / R[:, None]

    # Match direction_cosines() in the original optimization.
    w = dh[mask.ravel()].mean(axis=0)
    w /= np.linalg.norm(w)
    ey = np.array([0.0, 0.0, 1.0]) - w[2] * w
    ey /= np.linalg.norm(ey)
    ex = np.cross(ey, w)

    u = dh @ ex
    v = dh @ ey
    g = R[mask.ravel()].mean() / R
    return u, v, g


def compute_scores(u, v, g, x, y, phases, chunk=256):
    weights = np.exp(1j * phases).astype(np.complex64)
    scores = np.empty(len(u), dtype=np.float64)
    for start in range(0, len(u), chunk):
        stop = min(start + chunk, len(u))
        exponent = 2j * np.pi * (
            np.outer(u[start:stop], x) + np.outer(v[start:stop], y)
        )
        steering = np.exp(exponent).astype(np.complex64)
        field = steering @ weights
        scores[start:stop] = (np.abs(field).astype(np.float64) * g[start:stop]) ** 2
    return scores


def best_cut(scores, inm):
    n_in = int(inm.sum())
    order = np.argsort(-scores)
    ci = np.cumsum(inm[order])
    k = np.arange(1, len(scores) + 1)
    E = n_in + k - 2 * ci
    kb = int(np.argmin(E))
    if E[kb] >= n_in:
        raise ValueError("Best threshold is infinite: the all-off baseline wins.")
    return float(scores[order[kb]])


def load_inputs(mat_path, npz_path):
    ga = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)["GA"]
    lon = np.asarray(ga.lon, dtype=float)
    lat = np.asarray(ga.lat, dtype=float)
    if np.nanmax(lat) < 0:
        lat = -lat
    mask = np.asarray(ga.data) >= 128

    with np.load(npz_path) as design:
        x = np.asarray(design["x_lambda"], dtype=float)
        y = np.asarray(design["y_lambda"], dtype=float)
        phases = np.asarray(design["phase_rad"], dtype=float)
        sat_lon = float(design["sat_lon"])
        saved_T = float(design["T"]) if "T" in design.files else None
        saved_E = int(design["E_full"]) if "E_full" in design.files else None

    if not (len(x) == len(y) == len(phases)):
        raise ValueError("x_lambda, y_lambda, phase_rad must have equal lengths.")
    if len(x) == 0:
        raise ValueError("NPZ contains no active antenna elements.")

    return lon, lat, mask, x, y, phases, sat_lon, saved_T, saved_E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", default="GAmap.mat")
    ap.add_argument("--npz", default="array_design.npz")
    ap.add_argument("--chunk", type=int, default=256)
    args = ap.parse_args()

    lon, lat, mask, x, y, phases, sat_lon, saved_T, saved_E = load_inputs(args.mat, args.npz)

    u, v, g = geometry(lat, lon, mask, sat_lon)
    scores = compute_scores(u, v, g, x, y, phases, args.chunk)
    threshold = saved_T if saved_T is not None else best_cut(scores, mask.ravel())
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Finite positive threshold T is required for this plot.")

    power_dbm = -80.0 + 10.0 * np.log10(np.maximum(scores, 1e-300) / threshold)
    power_dbm = power_dbm.reshape(mask.shape)
    lit = scores.reshape(mask.shape) >= threshold

    miss = int(np.count_nonzero(mask & ~lit))
    spill = int(np.count_nonzero(~mask & lit))
    E = miss + spill

    # ---------------- Figure of merit ----------------
    # FOM = 1/sqrt(N_on + 100) * 1/(missed Georgia points + spilled outside points)
    N_on = len(x)
    elem_term = 1.0 / np.sqrt(N_on + 100)
    err_term = 1.0 / E if E > 0 else float("inf")   # E = miss + spill
    fom = elem_term * err_term

    print("\n=== Figure of merit ===")
    print(f"On elements (N):          {N_on}")
    print(f"Missed Georgia points:    {miss:,}")
    print(f"Spill outside Georgia:    {spill:,}")
    print(f"1/sqrt(N+100):            {elem_term:.6f}")
    print(f"1/(miss+spill):           {err_term:.6e}")
    print(f"FOM:                      {fom:.6e}")

    if saved_E is not None:
        fom_saved = elem_term / saved_E if saved_E > 0 else float("inf")
        print(f"FOM using saved E_full:   {fom_saved:.6e}")

    extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]
    origin = "upper" if lat[0, 0] > lat[-1, 0] else "lower"

    # All three visualizations in ONE interactive window.
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(20, 6.5), constrained_layout=True
    )
    fig.suptitle(f"GEO Array Design — {len(x)} Active Elements", fontsize=15)

    # --- Panel 1: received power ---
    im1 = ax1.imshow(
        power_dbm, extent=extent, origin=origin,
        vmin=-120, vmax=-40, cmap="turbo", interpolation="nearest", aspect="auto"
    )
    ax1.contour(lon, lat, mask.astype(float), levels=[0.5], colors="black", linewidths=0.8)
    ax1.set_title("Received Power (dBm)")
    ax1.set_xlabel("Longitude (°)")
    ax1.set_ylabel("Latitude (°N)")
    ax1.grid(alpha=0.15)
    fig.colorbar(im1, ax=ax1, shrink=0.8, label="Received power (dBm)")

    # --- Panel 2: coverage classification ---
    cls = np.where(mask, np.where(lit, 0, 1), np.where(lit, 2, 3))
    cmap = ListedColormap(["#2ca02c", "#d62728", "#ff7f0e", "#dddddd"])
    legend_labels = ["GA covered", "GA missed", "Outside illuminated", "Outside dark"]
    ax2.imshow(
        cls, cmap=cmap, vmin=-0.5, vmax=3.5, extent=extent,
        origin=origin, interpolation="nearest", aspect="auto"
    )
    ax2.contour(lon, lat, mask.astype(float), levels=[0.5], colors="black", linewidths=0.7)
    ax2.set_title(f"Coverage at -80 dBm\n{E:,} errors ({miss:,} missed, {spill:,} spill)")
    ax2.set_xlabel("Longitude (°)")
    ax2.set_ylabel("Latitude (°N)")
    ax2.grid(alpha=0.15)
    ax2.legend(
        handles=[Patch(facecolor=c, label=l) for c, l in zip(cmap.colors, legend_labels)],
        loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8,
    )

    # --- Panel 3: active element positions colored by phase ---
    phase_deg = np.mod(np.degrees(phases), 360.0)
    sc = ax3.scatter(x, y, c=phase_deg, cmap="hsv", vmin=0, vmax=360, s=18)
    ax3.set_aspect("equal", adjustable="box")
    ax3.set_xlabel("x position (wavelengths)")
    ax3.set_ylabel("y position (wavelengths)")
    ax3.set_title(f"Element Positions / Phase\nN = {len(x)}")
    ax3.grid(alpha=0.25)
    fig.colorbar(sc, ax=ax3, shrink=0.8, label="Phase (degrees)")

    plt.show()


if __name__ == "__main__":
    main()
