"""
ensemble_polarfield_calibration.py

Ensemble calibration of the two free transport parameters -- the Yeates (2020)
meridional-flow amplitude ``du`` (entered as the peak speed ``v0``) and the
supergranular diffusivity ``eta`` -- against the observed HMI polar field, with
the model driven by **direct HMI SHARP patch insertion** (``SHARPPatchSource``,
the real per-region ``B_r`` maps) rather than idealized bipoles.

    python examples/ensemble_polarfield_calibration.py                # full 5x5 ensemble
    python examples/ensemble_polarfield_calibration.py --quick        # 2x2 smoke test
    python examples/ensemble_polarfield_calibration.py --v0 12 14 16 --eta 3.5e8 4.5e8

Every ensemble member is a full 2010-2026 forward run on a 181x360 grid.  Each
member's snapshots (27-day cadence ``B_r`` maps) and diagnostic time series are
written to their own ``.npz`` under ``--outdir``; the members are scored against
HMI and ranked in a final table (CSV + Markdown + figure).

Why these two parameters
------------------------
``du`` sets how fast trailing-polarity flux is carried to the pole -- it controls
the *timing* of the polar reversal and, with the decay term, the plateau level.
``eta`` sets how much leading-polarity flux cancels across the equator before it
can be transported -- it controls the *amplitude* of the polar field.  They trade
off against each other, so they must be scanned jointly rather than one at a time.
``tau`` (the flux-decay time) is held fixed at the calibrated 10 yr; override with
``--tau``.

Scoring (this is not just a global correlation)
-----------------------------------------------
Each member is compared to the HMI ``mean_north``/``mean_south`` cap field
(poleward of 60 deg, ``hmi.meanpf_720s`` CAPN2/CAPS2) on five separate axes, per
hemisphere where it makes sense:

``corrN``/``corrS``
    Pearson correlation of the whole 2010-2026 series -- overall shape.
``dtN24``/``dtS24``/``dtN25``/``dtS25``
    Signed error (model - observed, in years) of the **polar-reversal epoch** for
    cycle 24 and cycle 25, found as the last sign change that "sticks" for a year.
    Observed: N 2014.33, S 2014.18, N 2024.33, S 2024.15.
``ampN``/``ampS``
    Signed cycle-24 **plateau amplitude** (the extremum over 2016-2023) per
    hemisphere, and its ratio to the observed +3.86 G (N) / -4.65 G (S).
``fluxN``/``fluxS``
    Signed **polar flux** [Mx] poleward of 60 deg at the same plateau epoch, per
    hemisphere, against the HMI-equivalent flux (observed cap field x cap area).
``peakD``
    Peak axial dipole moment [G] -- the amplitude observable that is *not*
    sensitive to the HMI cap-field metric offset.

The composite objective ``J`` (lower is better) sums three normalised terms, each
~1.0 when that axis is exactly at its stated tolerance::

    J = mean|dt| / 0.5 yr  +  mean|amp_ratio - 1| / 0.25  +  (1 - mean corr) / 0.10

so half a year of reversal-timing error, a 25% amplitude error, and a 0.10 drop
in correlation are all penalised equally.  The per-axis columns are kept in the
output table so a member can be judged on any single criterion too.

.. note::
   The HMI polar-cap field carries a known line-of-sight/metric offset of roughly
   1.6-2x relative to the true radial cap field, whereas the axial dipole is
   robust (see ``docs/sft2d-theory.md``).  ``ampN``/``ampS`` are therefore
   reported *and* scored, but a member that matches ``peakD`` ~ 3-4 G while
   sitting low on ``amp_ratio`` is not necessarily worse -- read the amplitude
   columns together with ``peakD``.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

# One BLAS thread per worker: the stepper is elementwise-bound, so letting each
# of N worker processes spawn N threads only oversubscribes the machine.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

YEAR = 365.25 * 86400.0

DEFAULT_DB = os.environ.get(
    "SFT2D_SHARPS_DB",
    "/Users/sdash/NSO/Work/GIT-Projects/sharps-bmrs-db/sharps-bmrs-db",
)

# --- ensemble ranges (override on the command line) -------------------------
V0S = [10.0, 12.0, 14.0, 16.0, 18.0]                      # m/s   (Yeates du)
ETAS = [2.5e8, 3.5e8, 4.5e8, 5.5e8, 6.5e8]                # m^2/s (250..650 km^2/s)

START, END = "2010-05-01", "2026-06-30"
T0 = 2010 + 0.33                       # decimal year of START, matches the runs
CAP = 30.0                             # cap half-angle [deg] -> poleward of 60
TAU_YEARS = 10.0
NTHETA, NPHI = 181, 360
REC_EVERY = 10                         # diagnostic cadence [days]
SNAP_EVERY = 27                        # snapshot cadence [days] (~1 rotation)

# Observed reversal epochs and plateau amplitudes, measured from the bundled HMI
# series by the same detectors used on the model (see _observed()).
REV_WIN_24 = (2011.5, 2016.5)
REV_WIN_25 = (2022.5, 2026.6)
PLATEAU_WIN = (2016.0, 2023.0)

# scoring tolerances (a term contributes 1.0 to J at its tolerance)
TOL_DT, TOL_AMP, TOL_CORR = 0.5, 0.25, 0.10


# ---------------------------------------------------------------- metrics ---
def reversal_time(t, b, window, stick_years=1.0):
    """Epoch of the polar reversal inside ``window`` [decimal yr], or NaN.

    Takes the *last* zero crossing whose new sign then holds for ``stick_years``,
    so brief excursions around the reversal do not count as the reversal itself.
    """
    t = np.asarray(t, float); b = np.asarray(b, float)
    m = (t >= window[0]) & (t <= window[1])
    tt, bb = t[m], b[m]
    if tt.size < 3:
        return np.nan
    s = np.sign(bb)
    best = np.nan
    for i in np.where(np.diff(s) != 0)[0]:
        if s[i] == 0 or bb[i + 1] == bb[i]:
            continue
        tc = tt[i] + (tt[i + 1] - tt[i]) * (-bb[i]) / (bb[i + 1] - bb[i])
        post = (t > tc) & (t <= tc + stick_years)
        if post.any() and np.all(np.sign(b[post]) == s[i + 1]):
            best = tc
    return best


def plateau_amplitude(t, b, window=PLATEAU_WIN):
    """Signed extremum of ``b`` in ``window`` and the epoch it occurs."""
    t = np.asarray(t, float); b = np.asarray(b, float)
    m = (t >= window[0]) & (t <= window[1])
    if not m.any():
        return np.nan, np.nan
    i = int(np.argmax(np.abs(b[m])))
    return float(b[m][i]), float(t[m][i])


def _observed():
    """HMI reference: series, reversal epochs, plateau amplitudes, cap flux."""
    from sft2d.data import load_hmi_polar_field

    h = load_hmi_polar_field()
    idx = h["mean_north"].index
    t = np.asarray(idx.year + (idx.dayofyear - 1) / 365.25, float)
    n = np.asarray(h["mean_north"].values, float)
    s = np.asarray(h["mean_south"].values, float)
    ok = np.isfinite(n) & np.isfinite(s)
    t, n, s = t[ok], n[ok], s[ok]
    return dict(
        t=t, n=n, s=s,
        revN24=reversal_time(t, n, REV_WIN_24), revS24=reversal_time(t, s, REV_WIN_24),
        revN25=reversal_time(t, n, REV_WIN_25), revS25=reversal_time(t, s, REV_WIN_25),
        ampN=plateau_amplitude(t, n)[0], ampS=plateau_amplitude(t, s)[0],
    )


def score(res, obs, cap_area_cm2):
    """All metrics for one member, plus the composite objective ``J``."""
    t, pn, ps = res["yr"], res["pn"], res["ps"]
    out = {}

    # shape: correlation on the model's own time base
    for lab, model, o in (("N", pn, obs["n"]), ("S", ps, obs["s"])):
        oi = np.interp(t, obs["t"], o)
        out[f"corr{lab}"] = float(np.corrcoef(model, oi)[0, 1])
        out[f"rmse{lab}"] = float(np.sqrt(np.mean((model - oi) ** 2)))

    # reversal timing, both cycles, both hemispheres (signed model - observed)
    for lab, model, win, key in (("N24", pn, REV_WIN_24, "revN24"),
                                 ("S24", ps, REV_WIN_24, "revS24"),
                                 ("N25", pn, REV_WIN_25, "revN25"),
                                 ("S25", ps, REV_WIN_25, "revS25")):
        tm = reversal_time(t, model, win)
        out[f"rev{lab}"] = tm
        out[f"dt{lab}"] = tm - obs[key] if np.isfinite(tm) else np.nan

    # plateau amplitude and polar flux, per hemisphere
    for lab, model, oamp in (("N", pn, obs["ampN"]), ("S", ps, obs["ampS"])):
        a, tp = plateau_amplitude(t, model)
        out[f"amp{lab}"] = a
        out[f"tamp{lab}"] = tp
        out[f"ratio{lab}"] = a / oamp if (np.isfinite(a) and oamp) else np.nan
        out[f"flux{lab}"] = a * cap_area_cm2          # model cap flux [Mx]
        out[f"obsflux{lab}"] = oamp * cap_area_cm2    # HMI-equivalent cap flux
    out["peakD"] = float(np.nanmax(np.abs(res["dip"])))

    # composite: timing + amplitude + shape, each normalised by its tolerance
    dts = [out[f"dt{k}"] for k in ("N24", "S24", "N25", "S25")]
    dts = [abs(d) for d in dts if np.isfinite(d)]
    rats = [out["ratioN"], out["ratioS"]]
    rats = [abs(r - 1.0) for r in rats if np.isfinite(r)]
    cors = [out["corrN"], out["corrS"]]

    # a member that never reverses is not "missing data" -- penalise it
    n_missing = sum(1 for k in ("N24", "S24", "N25", "S25")
                    if not np.isfinite(out[f"dt{k}"]))
    out["n_missing_rev"] = n_missing
    j_dt = (np.mean(dts) if dts else 0.0) / TOL_DT + 2.0 * n_missing
    j_amp = (np.mean(rats) if rats else 1.0) / TOL_AMP
    j_corr = (1.0 - np.mean(cors)) / TOL_CORR
    out["J_dt"], out["J_amp"], out["J_corr"] = j_dt, j_amp, j_corr
    out["J"] = j_dt + j_amp + j_corr
    return out


# ------------------------------------------------------------ one member ----
def run_member(cfg):
    """Run one (v0, eta) member, save its snapshots + series, return metrics."""
    import warnings
    warnings.filterwarnings("ignore")

    from sft2d.analysis.analysis import (calculate_dm, calculate_polar_field,
                                         calculate_polar_flux, calculate_usflx)
    from sft2d.data import HMI_SYNOPTIC_FITS
    from sft2d.src.grid import create_grid
    from sft2d.src.initial_conditions import initialize_field
    from sft2d.src.sharp_patch_driver import SHARPPatchSource
    from sft2d.src.stepper import evolve
    from sft2d.src.transport_profiles import differential_rotation, meridional_flow

    v0, eta = cfg["v0"], cfg["eta"]
    t_start = time.time()

    grid = create_grid(cfg["ntheta"], cfg["nphi"])
    mf = meridional_flow(grid, peak_speed=v0, profile="yeates2020")
    dr = differential_rotation(grid)
    field = initialize_field(grid, "read", path=str(HMI_SYNOPTIC_FITS))
    src = SHARPPatchSource(cfg["catalogue"], nc_dir=cfg["nc_dir"],
                           start_date=cfg["start"], end_date=cfg["end"],
                           flux_scale=1.0)

    yr, pn, ps, dip, usf, fn, fs, bfly = [], [], [], [], [], [], [], []
    snap_br, snap_day = [], []
    rec_every, snap_every, cap = cfg["rec_every"], cfg["snap_every"], cfg["cap"]

    class Rec:
        def record(self, day, B):
            if day % rec_every == 0:
                yr.append(T0 + day / 365.25)
                n, s = calculate_polar_field(B, grid, pol_cap_extent_deg=cap)
                pn.append(n); ps.append(s)
                a, b = calculate_polar_flux(B, grid, pol_cap_extent_deg=cap)
                fn.append(a); fs.append(b)
                dip.append(calculate_dm(B, grid))
                usf.append(calculate_usflx(B, grid))
                bfly.append(B.mean(axis=1).copy())
            if day % snap_every == 0:
                snap_br.append(B.astype(np.float32).copy()); snap_day.append(day)

    tau_s = cfg["tau_years"] * YEAR if cfg["tau_years"] else None
    evolve(field, grid, mf, dr, eta, src.num_days, source=src,
           tau_decay_s=tau_s, recorder=Rec())

    res = dict(yr=np.array(yr), pn=np.array(pn), ps=np.array(ps),
               dip=np.array(dip), usf=np.array(usf),
               fluxN=np.array(fn), fluxS=np.array(fs),
               bfly=np.array(bfly).T)

    # ---- save this member's snapshots + series ----
    tag = f"v{v0:04.1f}_eta{eta/1e6:03.0f}".replace(".", "p")
    out = Path(cfg["outdir"]) / f"member_{tag}.npz"
    lat = np.rad2deg(np.pi / 2 - grid["colatitude"])
    lon = np.rad2deg(grid["longitude"])
    snap_year = T0 + np.asarray(snap_day) / 365.25
    np.savez_compressed(
        out,
        br=np.array(snap_br), snap_day=np.asarray(snap_day), snap_year=snap_year,
        lat=lat, lon=lon,
        yr=res["yr"], pn=res["pn"], ps=res["ps"], dip=res["dip"], usf=res["usf"],
        polar_flux_n=res["fluxN"], polar_flux_s=res["fluxS"], bfly=res["bfly"],
        v0=v0, eta=eta, tau_years=cfg["tau_years"] or 0.0, cap_deg=cap,
    )

    from sft2d.analysis.analysis import cap_areas
    an, _ = cap_areas(grid, cap, cm=True)
    cap_area_cm2 = float(np.sum(an)) * grid["n_phi"]

    m = score(res, cfg["obs"], cap_area_cm2)
    m.update(v0=v0, eta=eta, file=out.name, seconds=time.time() - t_start,
             mb=out.stat().st_size / 1e6)
    return m


# --------------------------------------------------------------- reporting --
COLS = ["v0", "eta", "J", "J_dt", "J_amp", "J_corr", "corrN", "corrS",
        "dtN24", "dtS24", "dtN25", "dtS25", "revN24", "revS24", "revN25", "revS25",
        "ampN", "ampS", "ratioN", "ratioS", "fluxN", "fluxS",
        "peakD", "rmseN", "rmseS", "n_missing_rev", "seconds", "mb", "file"]


def write_outputs(rows, obs, outdir, args):
    outdir = Path(outdir)
    rows = sorted(rows, key=lambda r: (np.inf if not np.isfinite(r["J"]) else r["J"]))

    with open(outdir / "ensemble_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # ---- ranked Markdown table ----
    md = [
        "# Ensemble calibration: Yeates-2020 flow amplitude (du) vs diffusivity (eta)",
        "",
        f"Direct HMI SHARP patch insertion (`SHARPPatchSource`), {args.ntheta}x{args.nphi} grid, "
        f"{START} to {END}, tau = {args.tau:g} yr, cap poleward of {90-CAP:g} deg.",
        f"{len(rows)} members. Ranked by the composite objective **J** (lower = better); "
        "`J = mean|dt|/0.5yr + mean|amp_ratio-1|/0.25 + (1-corr)/0.10`.",
        "",
        "Observed (HMI): reversals N24 = %.2f, S24 = %.2f, N25 = %.2f, S25 = %.2f; "
        "plateau amplitude N = %+.2f G, S = %+.2f G."
        % (obs["revN24"], obs["revS24"], obs["revN25"], obs["revS25"],
           obs["ampN"], obs["ampS"]),
        "",
        "| rank | du (v0) | eta | **J** | corr N / S | dt N24 / S24 | dt N25 / S25 "
        "| amp N / S [G] | ratio N / S | polar flux N / S [1e22 Mx] | peak D [G] |",
        "|---:|---:|---:|---:|:--:|:--:|:--:|:--:|:--:|:--:|---:|",
    ]

    def f(x, spec="%+.2f"):
        return "--" if not np.isfinite(x) else spec % x

    for i, r in enumerate(rows, 1):
        md.append(
            f"| {i} | {r['v0']:.0f} | {r['eta']/1e6:.0f} | **{r['J']:.2f}** "
            f"| {f(r['corrN'],'%.3f')} / {f(r['corrS'],'%.3f')} "
            f"| {f(r['dtN24'])} / {f(r['dtS24'])} "
            f"| {f(r['dtN25'])} / {f(r['dtS25'])} "
            f"| {f(r['ampN'])} / {f(r['ampS'])} "
            f"| {f(r['ratioN'],'%.2f')} / {f(r['ratioS'],'%.2f')} "
            f"| {f(r['fluxN']/1e22)} / {f(r['fluxS']/1e22)} "
            f"| {f(r['peakD'],'%.2f')} |"
        )

    b = rows[0]
    md += [
        "",
        "## Best member",
        "",
        f"**du (v0) = {b['v0']:.0f} m/s, eta = {b['eta']/1e6:.0f} km^2/s** "
        f"(tau = {args.tau:g} yr) -- J = {b['J']:.2f}",
        "",
        f"* correlation with HMI: N {b['corrN']:+.3f}, S {b['corrS']:+.3f}",
        f"* cycle-24 reversal error: N {f(b['dtN24'])} yr, S {f(b['dtS24'])} yr",
        f"* cycle-25 reversal error: N {f(b['dtN25'])} yr, S {f(b['dtS25'])} yr",
        f"* plateau amplitude: N {f(b['ampN'])} G (obs {obs['ampN']:+.2f}, "
        f"ratio {f(b['ratioN'],'%.2f')}), S {f(b['ampS'])} G "
        f"(obs {obs['ampS']:+.2f}, ratio {f(b['ratioS'],'%.2f')})",
        f"* polar flux: N {f(b['fluxN']/1e22)}e22 Mx, S {f(b['fluxS']/1e22)}e22 Mx",
        f"* peak axial dipole: {b['peakD']:.2f} G",
        f"* snapshots: `{b['file']}`",
        "",
        "### Best on each individual criterion",
        "",
        "| criterion | best member (du, eta) | value |",
        "|:--|:--|:--|",
    ]
    crit = [
        ("mean |reversal-time error| [yr]",
         lambda r: np.nanmean([abs(r[k]) for k in ("dtN24", "dtS24", "dtN25", "dtS25")]), "%.3f"),
        ("mean correlation (N,S)", lambda r: -0.5 * (r["corrN"] + r["corrS"]), "%.3f"),
        ("amplitude ratio error |1-r|",
         lambda r: np.nanmean([abs(r["ratioN"] - 1), abs(r["ratioS"] - 1)]), "%.3f"),
        ("|peak dipole - 3.5 G|", lambda r: abs(r["peakD"] - 3.5), "%.3f"),
    ]
    for name, fn_, spec in crit:
        vals = [(fn_(r), r) for r in rows if np.isfinite(fn_(r))]
        if not vals:
            continue
        v, r = min(vals, key=lambda x: x[0])
        shown = -v if name.startswith("mean correlation") else v
        md.append(f"| {name} | du={r['v0']:.0f}, eta={r['eta']/1e6:.0f} | {spec % shown} |")

    md += [
        "",
        "> Note: the HMI polar-cap field carries a known ~1.6-2x line-of-sight/metric",
        "> offset while the axial dipole does not, so read `ratio N / S` together with",
        "> `peak D` before treating an amplitude mismatch as a model error.",
        "",
    ]
    (outdir / "ensemble_summary.md").write_text("\n".join(md))
    return rows


def make_figure(rows, obs, outdir, v0s, etas):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def surf(key, fn_=None):
        A = np.full((len(etas), len(v0s)), np.nan)
        for r in rows:
            try:
                i, j = etas.index(r["eta"]), v0s.index(r["v0"])
            except ValueError:
                continue
            A[i, j] = fn_(r) if fn_ else r[key]
        return A

    panels = [
        ("composite objective J", surf("J"), "viridis_r"),
        ("mean |reversal-time error| [yr]",
         surf(None, lambda r: np.nanmean([abs(r[k]) for k in ("dtN24", "dtS24", "dtN25", "dtS25")])),
         "magma_r"),
        ("mean |amplitude ratio - 1|",
         surf(None, lambda r: np.nanmean([abs(r["ratioN"] - 1), abs(r["ratioS"] - 1)])),
         "cividis_r"),
        ("mean correlation with HMI",
         surf(None, lambda r: 0.5 * (r["corrN"] + r["corrS"])), "viridis"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    for ax, (title, A, cmap) in zip(axes.ravel()[:4], panels):
        im = ax.imshow(A, origin="lower", aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(v0s))); ax.set_xticklabels([f"{v:.0f}" for v in v0s])
        ax.set_yticks(range(len(etas))); ax.set_yticklabels([f"{e/1e6:.0f}" for e in etas])
        ax.set_xlabel("du (v0) [m/s]"); ax.set_ylabel("eta [km^2/s]")
        ax.set_title(title, fontsize=10)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if np.isfinite(A[i, j]):
                    ax.text(j, i, f"{A[i,j]:.2f}", ha="center", va="center",
                            fontsize=7, color="w")
        fig.colorbar(im, ax=ax)

    # best member vs HMI, both hemispheres
    best = rows[0]
    z = np.load(Path(outdir) / best["file"])
    for ax, (lab, mk, ok_) in zip(axes.ravel()[4:],
                                  (("north", "pn", "n"), ("south", "ps", "s"))):
        ax.plot(obs["t"], obs[ok_], "k", lw=1, label="HMI")
        ax.plot(z["yr"], z[mk], "C3", lw=1.4,
                label=f"best: du={best['v0']:.0f}, eta={best['eta']/1e6:.0f}")
        ax.axhline(0, color="k", lw=0.4)
        ax.set_xlabel("year"); ax.set_ylabel(f"{lab} cap field [G]")
        ax.set_title(f"{lab} polar field -- best member vs HMI", fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("Ensemble calibration of du and eta against HMI "
                 "(direct SHARP patch insertion)", y=1.00)
    fig.tight_layout()
    fig.savefig(Path(outdir) / "ensemble_polarfield.png", dpi=140, bbox_inches="tight")


# -------------------------------------------------------------------- main --
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    p.add_argument("--v0", type=float, nargs="+", default=V0S,
                   help="Yeates-2020 meridional-flow peak speeds du [m/s]")
    p.add_argument("--eta", type=float, nargs="+", default=ETAS,
                   help="supergranular diffusivities [m^2/s]")
    p.add_argument("--tau", type=float, default=TAU_YEARS, help="flux-decay time [yr], 0 to disable")
    p.add_argument("--db", default=DEFAULT_DB, help="SHARPS database directory")
    p.add_argument("--outdir", default="ensemble_polarfield_out")
    p.add_argument("--ntheta", type=int, default=NTHETA)
    p.add_argument("--nphi", type=int, default=NPHI)
    p.add_argument("--snap-every", type=int, default=SNAP_EVERY, help="snapshot cadence [days]")
    p.add_argument("--workers", type=int, default=max(1, min(12, os.cpu_count() - 2)))
    p.add_argument("--quick", action="store_true", help="2x2 smoke test on a coarse grid")
    args = p.parse_args(argv)

    if args.quick:
        args.v0, args.eta = [12.0, 16.0], [3.5e8, 5.5e8]
        args.ntheta, args.nphi, args.snap_every = 91, 180, 90
        args.outdir = args.outdir + "_quick"

    catalogue = str(Path(args.db) / "bmrsharps_evol.txt")
    if not Path(catalogue).exists():
        sys.exit(f"catalogue not found: {catalogue}  (set --db or SFT2D_SHARPS_DB)")
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    obs = _observed()
    v0s, etas = list(args.v0), list(args.eta)
    members = [dict(v0=v, eta=e, catalogue=catalogue, nc_dir=args.db,
                    start=START, end=END, tau_years=(args.tau or None),
                    ntheta=args.ntheta, nphi=args.nphi, cap=CAP,
                    rec_every=REC_EVERY, snap_every=args.snap_every,
                    outdir=str(outdir), obs=obs)
               for e in etas for v in v0s]

    print(f"ensemble: {len(members)} members ({len(v0s)} du x {len(etas)} eta), "
          f"{args.ntheta}x{args.nphi}, tau={args.tau:g} yr, {args.workers} workers")
    print(f"  du  [m/s]    : {', '.join(f'{v:g}' for v in v0s)}")
    print(f"  eta [km^2/s] : {', '.join(f'{e/1e6:g}' for e in etas)}")
    print(f"  observed reversals: N24={obs['revN24']:.2f} S24={obs['revS24']:.2f} "
          f"N25={obs['revN25']:.2f} S25={obs['revS25']:.2f}")
    print(f"  output -> {outdir}/\n")

    t0 = time.time()
    rows = []
    if args.workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers) as pool:
            for m in pool.imap_unordered(run_member, members):
                rows.append(m)
                print(f"  [{len(rows):2d}/{len(members)}] du={m['v0']:4.1f} "
                      f"eta={m['eta']/1e6:3.0f}  J={m['J']:6.2f}  "
                      f"corr={0.5*(m['corrN']+m['corrS']):+.3f}  "
                      f"peakD={m['peakD']:.2f}  ({m['seconds']:.0f}s, {m['mb']:.0f} MB)",
                      flush=True)
    else:
        for cfg in members:
            m = run_member(cfg); rows.append(m)
            print(f"  [{len(rows):2d}/{len(members)}] du={m['v0']:4.1f} "
                  f"eta={m['eta']/1e6:3.0f}  J={m['J']:6.2f}", flush=True)

    rows = write_outputs(rows, obs, outdir, args)
    make_figure(rows, obs, outdir, v0s, etas)

    total_mb = sum(r["mb"] for r in rows)
    print(f"\nwall time {(time.time()-t0)/60:.1f} min, {total_mb:.0f} MB of snapshots")
    print(f"wrote {outdir}/ensemble_results.csv, ensemble_summary.md, ensemble_polarfield.png\n")

    b = rows[0]
    print("BEST: du=%.0f m/s, eta=%.0f km^2/s  ->  J=%.2f" % (b["v0"], b["eta"]/1e6, b["J"]))
    print("  corr N/S      %+.3f / %+.3f" % (b["corrN"], b["corrS"]))
    print("  dt reversal   N24 %s  S24 %s  N25 %s  S25 %s yr"
          % tuple("--" if not np.isfinite(b[k]) else "%+.2f" % b[k]
                  for k in ("dtN24", "dtS24", "dtN25", "dtS25")))
    print("  amplitude N/S %+.2f / %+.2f G (obs %+.2f / %+.2f)"
          % (b["ampN"], b["ampS"], obs["ampN"], obs["ampS"]))
    print("  peak dipole   %.2f G" % b["peakD"])


if __name__ == "__main__":
    main()
