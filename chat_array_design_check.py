#!/usr/bin/env python3
"""Reconstruct GEO-array Georgia coverage from GAmap.mat and array_design.npz.

Run: python recreate_georgia_coverage.py --mat GAmap.mat --npz array_design.npz
Requires numpy, scipy, matplotlib.
"""
import argparse
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, TwoSlopeNorm
from matplotlib.patches import Patch

RE_A = 6378137.0
E2 = 6.69437999014e-3
R_GEO = 42164170.0


def ecef(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    n = RE_A / np.sqrt(1 - E2 * np.sin(la)**2)
    return np.stack((n * np.cos(la) * np.cos(lo),
                     n * np.cos(la) * np.sin(lo),
                     n * (1-E2) * np.sin(la)), axis=-1)


def geometry(lat, lon, mask, sat_lon):
    points = ecef(lat, lon).reshape(-1, 3)
    sat = np.array([R_GEO*np.cos(np.radians(sat_lon)),
                    R_GEO*np.sin(np.radians(sat_lon)), 0.0])
    d = points - sat
    R = np.linalg.norm(d, axis=1)
    dh = d / R[:, None]
    # Match direction_cosines() in the original optimization exactly.
    w = dh[mask.ravel()].mean(axis=0)
    w /= np.linalg.norm(w)
    ey = np.array([0., 0., 1.]) - w[2] * w
    ey /= np.linalg.norm(ey)
    ex = np.cross(ey, w)
    u = dh @ ex
    v = dh @ ey
    R_ref = R[mask.ravel()].mean()
    return u, v, R_ref / R


def compute_scores(u, v, g, x, y, phases, chunk=256):
    weights = np.exp(1j * phases).astype(np.complex64)
    scores = np.empty(len(u), dtype=np.float64)
    for start in range(0, len(u), chunk):
        stop = min(start+chunk, len(u))
        exponent = 2j*np.pi*(np.outer(u[start:stop], x) +
                              np.outer(v[start:stop], y))
        steering = np.exp(exponent).astype(np.complex64)
        field = steering @ weights
        scores[start:stop] = (np.abs(field).astype(np.float64)*g[start:stop])**2
    return scores


def best_cut(scores, inm):
    # Copy of the optimizer's threshold search, for NPZs missing saved T.
    n_in = int(inm.sum())
    order = np.argsort(-scores)
    ci = np.cumsum(inm[order])
    k = np.arange(1, len(scores)+1)
    E = n_in + k - 2*ci
    kb = int(np.argmin(E))
    if E[kb] >= n_in:
        raise ValueError('Best threshold is infinite: the all-off baseline wins.')
    return float(scores[order[kb]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mat', default='GAmap.mat')
    ap.add_argument('--npz', default='array_design.npz')
    ap.add_argument('--out', default='georgia_coverage.png')
    ap.add_argument('--chunk', type=int, default=256)
    args = ap.parse_args()

    ga = sio.loadmat(args.mat, squeeze_me=True, struct_as_record=False)['GA']
    lon = np.asarray(ga.lon, dtype=float)
    lat = np.asarray(ga.lat, dtype=float)
    if np.nanmax(lat) < 0:
        lat = -lat
    mask = np.asarray(ga.data) >= 128

    with np.load(args.npz) as design:
        x = np.asarray(design['x_lambda'], dtype=float)
        y = np.asarray(design['y_lambda'], dtype=float)
        phases = np.asarray(design['phase_rad'], dtype=float)
        sat_lon = float(design['sat_lon'])
        saved_T = float(design['T']) if 'T' in design.files else None
        saved_E = int(design['E_full']) if 'E_full' in design.files else None

    if not (len(x) == len(y) == len(phases)):
        raise ValueError('x_lambda, y_lambda, phase_rad must have equal lengths.')
    if not len(x):
        raise ValueError('NPZ contains no active antenna elements.')

    u, v, g = geometry(lat, lon, mask, sat_lon)
    scores = compute_scores(u, v, g, x, y, phases, args.chunk)
    threshold = saved_T if saved_T is not None else best_cut(scores, mask.ravel())
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('Finite positive threshold T is required for this plot.')

    # At score = T, the original link-budget calculation assigns -80 dBm.
    power_dbm = -80.0 + 10.0*np.log10(np.maximum(scores, 1e-300)/threshold)
    power_dbm = power_dbm.reshape(mask.shape)
    lit = scores.reshape(mask.shape) >= threshold
    miss = int(np.count_nonzero(mask & ~lit))
    spill = int(np.count_nonzero(~mask & lit))
    E = miss + spill
    print(f'Active elements: {len(x)}')
    print(f'Georgia points below -80 dBm: {miss:,}')
    print(f'Outside-Georgia points >= -80 dBm: {spill:,}')
    print(f'Total full-grid errors: {E:,}')
    if saved_E is not None:
        print(f'Saved E_full: {saved_E:,}  (difference {E-saved_E:+,})')

    # Original class conventions: 0 green, 1 red, 2 orange, 3 gray.
    cls = np.where(mask, np.where(lit, 0, 1), np.where(lit, 2, 3))
    extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]
    # GAmap's first row is northernmost after the latitude sign correction.
    origin = 'upper' if lat[0, 0] > lat[-1, 0] else 'lower'

    fig, ax = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    im = ax[0].imshow(power_dbm, extent=extent, origin=origin,
                      vmin=-120, vmax=-40, cmap='turbo', interpolation='nearest', aspect='auto')
    ax[0].contour(lon, lat, mask.astype(float), levels=[0.5], colors='black', linewidths=0.8)
    ax[0].set_title('Received power (dBm), Georgia outline in black')
    cb = fig.colorbar(im, ax=ax[0], shrink=0.83)
    cb.set_label('Received power (dBm)')

    cmap = ListedColormap(['#2ca02c', '#d62728', '#ff7f0e', '#dddddd'])
    ax[1].imshow(cls, cmap=cmap, vmin=-0.5, vmax=3.5, extent=extent,
                 origin=origin, interpolation='nearest', aspect='auto')
    ax[1].contour(lon, lat, mask.astype(float), levels=[0.5], colors='black', linewidths=0.65)
    ax[1].set_title(f'Coverage at -80 dBm: {E:,} errors ({miss:,} missed, {spill:,} spill)')
    legend_labels = ['GA covered', 'GA missed', 'Outside illuminated', 'Outside dark']
    ax[1].legend(handles=[Patch(facecolor=c, label=l) for c, l in zip(cmap.colors, legend_labels)],
                 loc='lower center', bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=9)
    for a in ax:
        a.set_xlabel('Longitude (°)')
        a.set_ylabel('Latitude (°N)')
        a.grid(alpha=0.15)
    fig.savefig(args.out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {args.out}')

if __name__ == '__main__':
    main()
