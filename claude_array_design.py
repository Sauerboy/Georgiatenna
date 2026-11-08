#!/usr/bin/env python3
"""
Uniform-amplitude, phase-only, thinnable lambda/2 array in GEO that covers Georgia.

Figure of merit:   FoM = 1 / ( sqrt(N_on + 100) * E ),
                   E   = (# Georgia points below -80 dBm) + (# non-Georgia points at/above -80 dBm)

Key facts the code exploits
  * Tx power is a free scalar.  For any (mask, phases) the received-power pattern is known up to one
    constant, so the best power is found EXACTLY by sorting the scores and scanning every possible
    threshold crossing (best_cut).  Power is never an optimisation variable.
  * Pattern shaping is done in three stages per candidate geometry:
        1. Gerchberg-Saxton phase retrieval   (good starting phases)
        2. L-BFGS on a sigmoid "soft error count" with annealed sharpness (phases + threshold)
        3. Greedy thinning with the TRUE figure of merit (batched rank-1 updates), then re-polish phases
  * Outer loop sweeps aperture shape / size / rotation and keeps the best FoM.
  * A "do nothing" baseline (all elements off) is always reported, because with this FoM it is
    a serious competitor (see the printout).

Usage:  python design_array.py --mat GAmap.mat --stride 16 --sizes 6,9,12 --shapes hexagon,ellipse
"""
import argparse, os, time
import numpy as np
import scipy.io as sio
from scipy.optimize import minimize
try:
    from numba import njit, prange
except ImportError:                                   # pure-python fallback: works but is MUCH slower
    print("WARNING: numba not found (pip install numba); running ~10x slower")
    prange = range
    def njit(*a, **k):
        return (lambda f: f) if not (len(a) == 1 and callable(a[0])) else a[0]

RE_A, E2, R_GEO = 6378137.0, 6.69437999014e-3, 42164170.0
C0 = 299792458.0
P_TH_W = 1e-11                       # -80 dBm


# ----------------------------------------------------------------------------- geometry
def load_map(path, thresh=128):
    g = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["GA"]
    lon = np.asarray(g.lon, float)
    lat = np.asarray(g.lat, float)
    if np.nanmax(lat) < 0:           # file stores latitude with flipped sign (row 0 = 36N)
        lat = -lat
    return lon, lat, np.asarray(g.data) >= thresh


def ecef(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    n = RE_A / np.sqrt(1 - E2 * np.sin(la) ** 2)
    return np.stack([n * np.cos(la) * np.cos(lo), n * np.cos(la) * np.sin(lo),
                     n * (1 - E2) * np.sin(la)], -1)


def direction_cosines(lat, lon, inmask, sat_lon):
    """(u, v, range) of every map point as seen from a GEO sat; boresight = centroid of Georgia."""
    P = ecef(lat, lon).reshape(-1, 3)
    sat = np.array([R_GEO * np.cos(np.radians(sat_lon)), R_GEO * np.sin(np.radians(sat_lon)), 0.0])
    d = P - sat
    R = np.linalg.norm(d, axis=1)
    dh = d / R[:, None]
    w = dh[inmask.ravel()].mean(0)
    w /= np.linalg.norm(w)
    ey = np.array([0, 0, 1.0]) - w[2] * w
    ey /= np.linalg.norm(ey)
    ex = np.cross(ey, w)
    return dh @ ex, dh @ ey, R


# ----------------------------------------------------------------------------- aperture
def make_aperture(lattice, shape, size, aspect=1.3, rot_deg=0.0):
    """Element coordinates (in wavelengths). Nearest-neighbour spacing = 0.5 lambda.
    size = radius (circle / hexagon circumradius / equal-area ellipse) in wavelengths."""
    a = 0.5
    n = int(2.5 * size * (aspect if shape == "ellipse" else 1) / a) + 3
    i, j = np.meshgrid(np.arange(-n, n + 1), np.arange(-n, n + 1))
    if lattice == "hex":
        x, y = (i + 0.5 * (j % 2)) * a, j * a * np.sqrt(3) / 2
    else:
        x, y = i * a, j * a
    x, y = x.ravel(), y.ravel()
    x, y = x - x.mean(), y - y.mean()
    t = np.radians(rot_deg)
    xr, yr = x * np.cos(t) + y * np.sin(t), -x * np.sin(t) + y * np.cos(t)
    if shape == "circle":
        keep = xr ** 2 + yr ** 2 <= size ** 2
    elif shape == "ellipse":
        A_, B_ = size * np.sqrt(aspect), size / np.sqrt(aspect)
        keep = (xr / A_) ** 2 + (yr / B_) ** 2 <= 1
    elif shape == "hexagon":
        keep = np.ones_like(x, bool)
        for k in range(3):
            th = np.radians(30 + 60 * k)
            keep &= np.abs(xr * np.cos(th) + yr * np.sin(th)) <= size * np.sqrt(3) / 2
    else:
        raise ValueError(shape)
    return x[keep], y[keep]


def steering(u, v, g, x, y, chunk=4000):
    """A[m,n] = g_m * exp(j 2 pi (u_m x_n + v_m y_n)),  complex64.  g = Rref/R_m (spreading loss)."""
    out = np.empty((len(u), len(x)), np.complex64)
    for s in range(0, len(u), chunk):
        sl = slice(s, s + chunk)
        out[sl] = (g[sl, None] * np.exp(2j * np.pi * (np.outer(u[sl], x) + np.outer(v[sl], y)))
                   ).astype(np.complex64)
    return out


# ----------------------------------------------------------------------------- figure of merit
def fom(n_on, E):
    return 1.0 / (np.sqrt(n_on + 100.0) * max(E, 1))       # E=0 -> floored at 1 (FoM would be inf)


@njit(cache=True)
def _cut(s, inm, n_in):
    """Scan every threshold: lit = top-k scores. Returns (min errors, threshold); threshold=inf -> dark."""
    order = np.argsort(s)
    M = len(s)
    ci = 0
    bestE, bestT = n_in, np.inf
    for k in range(M):
        idx = order[M - 1 - k]
        if inm[idx]:
            ci += 1
        E = n_in + (k + 1) - 2 * ci
        if E < bestE:
            bestE, bestT = E, s[idx]
    return bestE, bestT


@njit(cache=True)
def _soft(s, inm, Tdb, tau):
    """Smooth error count (sigmoid in dB) around threshold Tdb."""
    tot = 0.0
    for m in range(len(s)):
        d = 10.0 * np.log10(s[m] + 1e-30) - Tdb
        z = -d / tau if inm[m] else d / tau
        z = min(max(z, -50.0), 50.0)
        tot += 1.0 / (1.0 + np.exp(-z))
    return tot


@njit(cache=True)
def _field(AT, idx, phi):
    y = np.zeros(AT.shape[1], np.complex128)
    for i in range(len(idx)):
        k = idx[i]
        e = np.exp(1j * phi[k])
        for m in range(AT.shape[1]):
            y[m] += AT[k, m] * e
    return y


@njit(cache=True)
def _score(y, inm, n_in, n):
    s = y.real ** 2 + y.imag ** 2
    E, T = _cut(s, inm, n_in)
    soft = _soft(s, inm, 10.0 * np.log10(T), 0.5) if np.isfinite(T) else float(E)
    return 0.5 * np.log(n + 100.0) + np.log(max(E + 0.3 * soft, 1.0)), E


@njit(parallel=True, cache=True)
def _thin_eval(y, AT, c, idx, inm, n_in, Tdb):
    """For every active element k: exact E and soft loss of the array with k removed (parallel over k)."""
    K, M = len(idx), len(y)
    Es, Ls = np.empty(K), np.empty(K)
    for i in prange(K):
        k = idx[i]
        s = np.empty(M)
        for m in range(M):
            d = y[m] - AT[k, m] * c[k]
            s[m] = d.real ** 2 + d.imag ** 2
        E, T = _cut(s, inm, n_in)
        Es[i] = E
        Ls[i] = _soft(s, inm, Tdb, 0.35)
    return Es, Ls


def best_cut(s, inm):
    """Exact optimal power: lit = {s >= T}.  Returns (E, T). T=inf means array dark."""
    E, T = _cut(np.ascontiguousarray(s, dtype=np.float64), inm, int(inm.sum()))
    return int(E), float(T)


# ----------------------------------------------------------------------------- phase optimisation
def gs_init(Aa, inm, rng, iters=60):
    M = Aa.shape[0]
    ext = 1.0
    psi = rng.uniform(0, 2 * np.pi) + 0 * np.arange(M)
    # random smooth quadratic phase on the target field spreads the beam over the footprint
    gamma = rng.uniform(0, 3)
    idx = np.arange(M)
    yt = np.where(inm, np.exp(1j * (psi + gamma * np.pi * np.sin(idx / 37.0) ** 2)), 0).astype(np.complex64)
    AaH = np.ascontiguousarray(Aa.conj().T)
    for _ in range(iters):
        c = np.exp(1j * np.angle(AaH @ yt)).astype(np.complex64)
        y = Aa @ c
        a = np.abs(y[inm]).mean()
        yt = np.where(inm, a * np.exp(1j * np.angle(y)), 0).astype(np.complex64)
    return np.angle(c)


def soft_loss_grad(x, Aa, inm, tau):
    phi, T = x[:-1], x[-1]
    c = np.exp(1j * phi).astype(np.complex64)
    y = Aa @ c
    s = np.abs(y).astype(np.float64) ** 2 + 1e-30
    sdb = 10 * np.log10(s)
    z = np.clip(np.where(inm, T - sdb, sdb - T) / tau, -60, 60)
    sig = 1 / (1 + np.exp(-z))
    dz = sig * (1 - sig) / tau
    dL_dsdb = np.where(inm, -dz, dz)
    dL_dT = np.where(inm, dz, -dz).sum()
    h = (dL_dsdb * (10 / np.log(10)) / s).astype(np.float32)
    g = -2 * np.imag(c * (Aa.T @ (h * np.conj(y))))
    return sig.sum(), np.r_[g.astype(np.float64), dL_dT]


def optimise_phases(Aa, inm, phi0, T0_db, taus=(3.0, 1.5, 0.7, 0.35), iters=60):
    x = np.r_[phi0, T0_db]
    for tau in taus:
        x = minimize(soft_loss_grad, x, args=(Aa, inm, tau), jac=True, method="L-BFGS-B",
                     options=dict(maxiter=iters)).x
    return x[:-1], x[-1]


def evaluate(Aa, phi, inm):
    s = np.abs(Aa @ np.exp(1j * phi).astype(np.complex64)).astype(np.float64) ** 2
    E, T = best_cut(s, inm)
    return E, T, s


# ----------------------------------------------------------------------------- thinning
def thin(AT, inm, active, phi, T_db, verbose=False, batch_frac=0.10):
    """Greedy removal of elements using the TRUE FoM; candidates ranked by exact E then soft loss."""
    active = active.copy()
    n_in = int(inm.sum())
    c_all = np.exp(1j * phi)
    y = _field(AT, np.flatnonzero(active), phi)
    E, _ = best_cut(np.abs(y) ** 2, inm)
    f_cur = fom(active.sum(), E)
    B = max(1, int(batch_frac * active.sum()))
    while B >= 1 and active.sum() > 1:
        idx = np.flatnonzero(active)
        Es, Ls = _thin_eval(y, AT, c_all, idx, inm, n_in, T_db)
        f_new = 1.0 / (np.sqrt(active.sum() - 1 + 100.0) * np.maximum(Es, 1))
        good = np.flatnonzero(f_new > f_cur)
        if len(good) == 0:
            break
        pick = idx[good[np.lexsort((Ls[good], Es[good]))[:B]]]
        trial = active.copy()
        trial[pick] = False
        y_t = _field(AT, np.flatnonzero(trial), phi)
        E_t, _ = best_cut(np.abs(y_t) ** 2, inm)
        f_t = fom(trial.sum(), E_t)
        if f_t > f_cur:
            active, y, E, f_cur = trial, y_t, E_t, f_t
            B = min(2 * B, max(1, int(batch_frac * active.sum())))      # grow again after a success
            if verbose:
                print(f"      N={active.sum():6d}  E={E:6d}  FoM={f_cur:.3e}")
        else:
            B //= 2
            if B < max(1, int(0.005 * active.sum())) and active.sum() > 200:
                break                                                   # batches too small to be worth it; SA takes over
    return active


# ----------------------------------------------------------------------------- simulated annealing (true FoM)
@njit(cache=True)
def _anneal(AT, inm, act0, phi0, iters, T_hi, T_lo, seed):
    np.random.seed(seed)
    Ns, M = AT.shape
    n_in = 0
    for m in range(M):
        n_in += inm[m]
    active = np.zeros(Ns, np.bool_)
    phi = phi0.copy()
    lst = np.empty(Ns, np.int64)
    pos = np.full(Ns, -1, np.int64)
    n = 0
    for i in range(len(act0)):
        k = act0[i]
        active[k] = True; pos[k] = n; lst[n] = k; n += 1
    y = _field(AT, lst[:n], phi)
    en, E = _score(y, inm, n_in, n)
    best_f = 1.0 / (np.sqrt(n + 100.0) * max(E, 1))
    best_act, best_phi, best_E = active.copy(), phi.copy(), E
    for it in range(iters):
        Tmp = T_hi * (T_lo / T_hi) ** (it / max(iters - 1, 1))
        r = np.random.random()
        mv = -1
        if r < 0.35 and n > 0:
            mv = 0                                    # phase tweak
        elif r < 0.65 and n > 0 and n < Ns:
            mv = 1                                    # relocate
        elif r < 0.82 and n < Ns:
            mv = 2                                    # add
        elif n > 1:
            mv = 3                                    # remove
        if mv < 0:
            continue
        k, m, p = -1, -1, 0.0
        n_new = n
        if mv == 0:
            k = lst[np.random.randint(0, n)]
            if np.random.random() < 0.6:
                p = phi[k] + np.random.normal(0.0, 0.3)
            else:
                p = phi[k] + np.random.uniform(-np.pi, np.pi)
            y_new = y + AT[k] * (np.exp(1j * p) - np.exp(1j * phi[k]))
        elif mv == 3:
            k = lst[np.random.randint(0, n)]
            y_new = y - AT[k] * np.exp(1j * phi[k])
            n_new = n - 1
        else:
            if mv == 1:
                k = lst[np.random.randint(0, n)]
                y0 = y - AT[k] * np.exp(1j * phi[k])
            else:
                y0 = y
                n_new = n + 1
            while True:
                m = np.random.randint(0, Ns)
                if not active[m]:
                    break
            sc_best, E_c = np.inf, 0
            y_new = y0
            for _ in range(3):                        # best of 3 random phases for the new element
                pp = np.random.uniform(0.0, 2.0 * np.pi)
                yy = y0 + AT[m] * np.exp(1j * pp)
                sc, e_ = _score(yy, inm, n_in, n_new)
                if sc < sc_best:
                    sc_best, E_c, y_new, p = sc, e_, yy, pp
        if mv == 1 or mv == 2:
            en_new, E_new = sc_best, E_c
        else:
            en_new, E_new = _score(y_new, inm, n_in, n_new)
        if en_new <= en or np.random.random() < np.exp(-(en_new - en) / Tmp):
            if mv == 0:
                phi[k] = p
            if mv == 1 or mv == 3:                    # remove k (swap-with-last)
                pk = pos[k]; last = lst[n - 1]
                lst[pk] = last; pos[last] = pk; pos[k] = -1
                active[k] = False
                if mv == 1:
                    n -= 1
            if mv == 3:
                n -= 1
            if mv == 1 or mv == 2:                    # add m
                active[m] = True; pos[m] = n; lst[n] = m; phi[m] = p; n += 1
            y, en, E = y_new, en_new, E_new
            f = 1.0 / (np.sqrt(n + 100.0) * max(E, 1))
            if f > best_f:
                best_f, best_act, best_phi, best_E = f, active.copy(), phi.copy(), E
        if it % 500 == 0:                             # kill accumulated round-off
            y = _field(AT, lst[:n], phi)
    return best_f, best_act, best_phi, best_E


@njit(parallel=True, cache=True)
def _anneal_multi(AT, inm, act0s, phi0s, iters, T_hi, T_lo, seeds):
    C, Ns = act0s.shape[0], AT.shape[0]
    fs, Es = np.empty(C), np.empty(C, np.int64)
    acts, phis = np.zeros((C, Ns), np.bool_), np.zeros((C, Ns))
    for c in prange(C):                               # independent chains, one per core
        f, a, p, E = _anneal(AT, inm, act0s[c], phi0s[c], iters, T_hi, T_lo, seeds[c])
        fs[c], acts[c], phis[c], Es[c] = f, a, p, E
    return fs, acts, phis, Es


def anneal(AT, inm, act0s, phi0s, iters, rng, T_hi=0.03, T_lo=0.002, verbose=False):
    """Run several independent annealing chains in parallel (numba prange) and return the best one:
    (FoM, active mask, phases, E).  act0s: (chains, n0) int site indices; phi0s: (chains, Nsites)."""
    seeds = rng.integers(2 ** 31 - 1, size=len(act0s)).astype(np.int64)
    fs, acts, phis, Es = _anneal_multi(AT, inm, np.ascontiguousarray(act0s, dtype=np.int64),
                                       np.ascontiguousarray(phi0s, dtype=np.float64),
                                       int(iters), T_hi, T_lo, seeds)
    b = int(np.argmax(fs))
    if verbose:
        print(f"      SA ({len(fs)} chains): best N={int(acts[b].sum())} E={Es[b]} FoM={fs[b]:.3e}")
    return fs[b], acts[b], phis[b], int(Es[b])


# ----------------------------------------------------------------------------- full pipeline for one geometry
def design_one(A_sub, inm, n_starts, thin_rounds, sa_iters, rng, verbose=False, sparse_n0=8, chains=1):
    N = A_sub.shape[1]
    AT = np.ascontiguousarray(A_sub.T)               # element-major copy for the compiled kernels
    best = None
    for st in range(n_starts):
        phi = gs_init(A_sub, inm, rng)
        E0, T0, _ = evaluate(A_sub, phi, inm)
        s = np.abs(A_sub @ np.exp(1j * phi).astype(np.complex64)) ** 2
        T_db = 10 * np.log10(T0 if np.isfinite(T0) else np.median(s))
        phi, T_db = optimise_phases(A_sub, inm, phi, T_db)
        E, T, _ = evaluate(A_sub, phi, inm)
        active = np.ones(N, bool)
        if verbose:
            print(f"    start {st}: filled  N={N}  E={E}  FoM={fom(N, E):.3e}")
        cand = (fom(N, E), active.copy(), phi.copy(), E, T)
        for r in range(thin_rounds):                              # greedy thinning + phase re-polish
            active = thin(AT, inm, active, phi, T_db, verbose)
            ia = np.flatnonzero(active)
            phi_a, T_db = optimise_phases(A_sub[:, ia], inm, phi[ia], T_db, taus=(0.7, 0.35), iters=60)
            phi[ia] = phi_a
            E, T, _ = evaluate(A_sub[:, ia], phi_a, inm)
            f = fom(len(ia), E)
            if verbose:
                print(f"    start {st} thin round {r}: N={len(ia)}  E={E}  FoM={f:.3e}")
            if f > cand[0]:
                cand = (f, active.copy(), phi.copy(), E, T)
        if sa_iters > 0:                                          # annealing with moves the greedy pass cannot make
            f, act, ph, E = anneal(AT, inm, np.tile(np.flatnonzero(cand[1]), (chains, 1)),
                                   np.tile(cand[2], (chains, 1)), sa_iters, rng, verbose=verbose)
            if f > cand[0]:
                s = np.abs(A_sub[:, act] @ np.exp(1j * ph[act]).astype(np.complex64)) ** 2
                cand = (f, act, ph, E, best_cut(s.astype(np.float64), inm)[1])
        if sa_iters > 0 and sparse_n0 > 0:                        # independent sparse seed (essential for big apertures)
            act0s = np.array([rng.choice(N, min(sparse_n0, N), replace=False) for _ in range(chains)])
            ph0s = rng.uniform(0, 2 * np.pi, (chains, N))
            f, act, ph, E = anneal(AT, inm, act0s, ph0s, sa_iters, rng, T_hi=0.06, verbose=verbose)
            if f > cand[0]:
                s = np.abs(A_sub[:, act] @ np.exp(1j * ph[act]).astype(np.complex64)) ** 2
                cand = (f, act, ph, E, best_cut(s.astype(np.float64), inm)[1])
        if best is None or cand[0] > best[0]:
            best = cand
    return best


# ----------------------------------------------------------------------------- full-grid evaluation / reporting
def full_scores(u, v, g, x, y, phi, chunk=3000):
    c = np.exp(1j * phi).astype(np.complex64)
    out = np.empty(len(u))
    for s in range(0, len(u), chunk):
        sl = slice(s, s + chunk)
        out[sl] = np.abs(steering(u[sl], v[sl], g[sl], x, y) @ c).astype(np.float64) ** 2
    return out


def power_report(T, R_ref, n_on, freq_ghz, rx_gain_dbi=0.0):
    lam = C0 / (freq_ghz * 1e9)
    p_elem = P_TH_W * (4 * np.pi * R_ref / lam) ** 2 / T / 10 ** (rx_gain_dbi / 10)   # isotropic elements, no atmos. loss
    return p_elem, p_elem * n_on


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", default="GAmap.mat")
    ap.add_argument("--sat_lon", type=float, default=-83.0, help="GEO slot longitude, deg (east +)")
    ap.add_argument("--freq_ghz", type=float, default=30.0, help="only used to report Tx power")
    ap.add_argument("--rx_gain_dbi", type=float, default=10.0)
    ap.add_argument("--stride", type=int, default=16, help="map sub-sampling used while optimising")
    ap.add_argument("--lattice", default="hex", choices=["hex", "square"])
    ap.add_argument("--shapes", default="hexagon,ellipse")
    ap.add_argument("--sizes", default="6,9,12", help="aperture radius in wavelengths")
    ap.add_argument("--rots", default="0", help="aperture rotation(s), deg")
    ap.add_argument("--aspect", type=float, default=1.3)
    ap.add_argument("--starts", type=int, default=2)
    ap.add_argument("--thin_rounds", type=int, default=2)
    ap.add_argument("--chains", type=int, default=min(8, os.cpu_count() or 1),
                    help="parallel annealing chains per geometry (one per core is ideal)")
    ap.add_argument("--sparse_n0", type=int, default=8, help="elements in the random sparse seed")
    ap.add_argument("--sa_iters", type=int, default=20000, help="annealing moves per geometry")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_full_eval", action="store_true")
    ap.add_argument("--out", default="array_design")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    lon, lat, mask = load_map(a.mat)
    u, v, R = direction_cosines(lat, lon, mask, a.sat_lon)
    R_ref = R[mask.ravel()].mean()
    g = R_ref / R
    H, W = mask.shape
    sub = np.zeros((H, W), bool)
    sub[::a.stride, ::a.stride] = True
    sub = sub.ravel()
    us, vs, gs, ms = u[sub], v[sub], g[sub], mask.ravel()[sub]
    n_in = int(ms.sum())
    print(f"Full grid: {mask.sum()} Georgia / {mask.size - mask.sum()} outside points.   "
          f"Optimisation grid (stride {a.stride}): {n_in} in / {len(ms) - n_in} out")
    print(f"Georgia footprint: u in [{u[mask.ravel()].min():.4f},{u[mask.ravel()].max():.4f}], "
          f"v in [{v[mask.ravel()].min():.4f},{v[mask.ravel()].max():.4f}]  (direction cosines)")
    base = fom(0, n_in)
    print(f"BASELINE (array off, N=0): E={n_in}  FoM={base:.4e}\n")

    results = []
    for shape in a.shapes.split(","):
        for size in [float(s) for s in a.sizes.split(",")]:
            for rot in ([0.0] if shape == "circle" else [float(r) for r in a.rots.split(",")]):
                t0 = time.time()
                x, y = make_aperture(a.lattice, shape, size, a.aspect, rot)
                A = steering(us, vs, gs, x, y)
                f, active, phi, E, T = design_one(A, ms, a.starts, a.thin_rounds, a.sa_iters, rng, a.verbose, a.sparse_n0, a.chains)
                print(f"{shape:8s} r={size:5.1f}λ rot={rot:4.0f}  N_full={len(x):6d} -> N_on={active.sum():6d}  "
                      f"E={E:6d}  FoM={f:.3e}  ({f / base:.2f}x baseline, {time.time() - t0:.0f}s)")
                results.append((f, shape, size, rot, x, y, active, phi, T))

    f, shape, size, rot, x, y, active, phi, T = max(results, key=lambda r: r[0])
    ia = np.flatnonzero(active)
    print(f"\nBEST: {shape} r={size}λ rot={rot}: N_on={len(ia)}  FoM(sub-grid)={f:.3e}  = {f / base:.2f}x baseline")
    if f <= base:
        print("  >> No array beat the all-off baseline on this grid. See notes in the README/answer.")

    xa, ya, pa = x[ia], y[ia], phi[ia]
    res = dict(x_lambda=xa, y_lambda=ya, phase_rad=pa % (2 * np.pi), sat_lon=a.sat_lon,
               lattice=a.lattice, shape=shape, size=size)
    if not a.no_full_eval:
        s_full = full_scores(u, v, g, xa, ya, pa)
        E_full, T_full = best_cut(s_full, mask.ravel())
        E_des = int((s_full[mask.ravel()] < T).sum() + (s_full[~mask.ravel()] >= T).sum())
        n_full = int(mask.sum())
        print(f"FULL {H}x{W} GRID:  best-power E={E_full} (FoM={fom(len(ia), E_full):.3e})   "
              f"all-off baseline E={n_full} (FoM={fom(0, n_full):.3e})")
        p_el, p_tot = power_report(T_full, R_ref, len(ia), a.freq_ghz, a.rx_gain_dbi)
        print(f"Tx power @ {a.freq_ghz} GHz (Rx gain {a.rx_gain_dbi} dBi, no atmos. loss): {p_el:.3g} W/element, "
              f"{p_tot:.3g} W total ({10 * np.log10(p_tot):.1f} dBW)")
        res.update(T=T_full, p_elem_W=p_el, p_total_W=p_tot, E_full=E_full)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            lit = s_full >= T_full
            cls = np.where(mask.ravel(), np.where(lit, 0, 1), np.where(lit, 2, 3)).reshape(H, W)
            fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
            ax[0].scatter(xa, ya, c=pa % (2 * np.pi), s=4, cmap="hsv")
            ax[0].set_aspect("equal"); ax[0].set_title(f"{len(ia)} elements on (colour = phase), axes in λ")
            cmap = matplotlib.colors.ListedColormap(["#2ca02c", "#d62728", "#ff7f0e", "#dddddd"])
            ax[1].imshow(cls, cmap=cmap, vmin=0, vmax=3, extent=[lon.min(), lon.max(), lat.min(), lat.max()])
            ax[1].set_title(f"green: GA lit | red: GA missed | orange: outside lit   (E={E_full})")
            fig.tight_layout(); fig.savefig(a.out + ".png", dpi=130)
        except Exception as e:                                       # plotting is optional
            print("plot skipped:", e)
    np.savez(a.out + ".npz", **res)
    print(f"saved {a.out}.npz")


if __name__ == "__main__":
    main()