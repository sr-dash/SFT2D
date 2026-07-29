"""
ensemble_3param_calibration.py

Three-parameter ensemble calibration of the SFT transport recipe against the
observed HMI polar field, driven by **direct HMI SHARP patch insertion**
(``SHARPPatchSource``).  The free parameters are

* ``du``  -- Yeates (2020) meridional-flow amplitude (entered as peak speed ``v0``)
             -> controls how fast trailing flux reaches the pole (reversal *timing*);
* ``eta`` -- supergranular diffusivity
             -> controls cross-equator cancellation (polar *amplitude*, and timing);
* ``tau`` -- exponential flux-decay time
             -> the only term that *removes* flux rather than moving it, so it sets
                the polar plateau level and how fast an old cycle's polar cap is
                erased before the next one reverses it.

    python examples/ensemble_3param_calibration.py --coarse   # 54 members, ~65 min
    python examples/ensemble_3param_calibration.py            # 100 members, ~2 h
    python examples/ensemble_3param_calibration.py --du 10 12 --eta 4.5e8 --tau 3 5

An optional fourth axis, ``--profile yeates2020 schuessler-baumann``, adds the
meridional-flow *shape*.  It is worth including: across the whole 2-parameter
(du, eta) box the polar over-concentration never fell below 1.83 and the cycle-24
reversal was 1.2-1.9 yr early with only 0.34 yr of spread, so neither defect is
reachable by the continuous parameters -- but the SB profile, which falls off
faster toward the poles, is what fixed the concentration in the earlier
calibration.  ``--coarse`` includes both profiles by default.

Every member is a full 2010-2026 forward run on 181x360.  Each member's snapshots
and diagnostic series go to their own ``.npz`` under ``--outdir``; members are then
scored against HMI and ranked (CSV + Markdown + figure).

Why the objective changed from the 2-parameter scan
---------------------------------------------------
The first (du, eta) ensemble ranked members with a term proportional to the raw
polar-cap amplitude ratio ``model/HMI``.  That term dominated the score (mean 4.67
vs 2.22 for timing and 1.43 for shape) and drove the "best" member to the corner
of the box (du=10, eta=650) -- a member whose **peak axial dipole is 5.13 G**,
well outside the 3-4 G band that the literature and the axial dipole itself
support.  In other words the raw cap amplitude, which carries a known ~1.6-2x HMI
line-of-sight/metric offset, was outvoting the unbiased observables.

This script therefore scores amplitude through quantities that are *not* subject
to that offset, and keeps the raw ratio as a reported diagnostic:

``J_time``
    mean |reversal-epoch error| over the four observed reversals
    (N/S x cycle 24/25), normalised by a 0.5 yr tolerance; +2 per reversal the
    member fails to produce at all.
``J_shape``
    ``(1 - mean corr)`` over both hemispheres, normalised by 0.10.
``J_dip``
    distance of the peak axial dipole from the 3-4 G band (zero inside it),
    normalised by 0.5 G.  The axial dipole is the amplitude observable that does
    *not* carry the HMI cap-field offset.
``J_conc``
    polar over-concentration ``conc = <|B_cap|> / (|D| * 0.933)`` measured at the
    epoch of peak dipole -- 1.0 means the polar cap is distributed like a dipole,
    >1 means flux is piled up in a narrow polar ring.  Penalised above 1.25,
    normalised by 0.5.  Every member of the 2-parameter ensemble sat at 1.83-2.78,
    so this is a real and currently unconstrained defect.

    J = J_time + J_shape + J_dip + J_conc          (``--objective robust``, default)

For continuity, ``--objective rawamp`` reproduces the previous ranking
(``J_time + mean|ratio-1|/0.25 + J_shape``); both values are written to the table
regardless, so the two rankings can always be compared.

Edge pinning
------------
If the best member lands on the boundary of any scanned axis the script says so
explicitly and names the axis to extend -- the failure mode of the first scan.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# metric primitives shared with the 2-parameter scan (same detectors, same windows)
from ensemble_polarfield_calibration import (  # noqa: E402
    CAP, END, PLATEAU_WIN, REC_EVERY, REV_WIN_24, REV_WIN_25, START, T0, YEAR,
    _observed, plateau_amplitude, reversal_time,
)

DEFAULT_DB = os.environ.get(
    "SFT2D_SHARPS_DB",
    "/Users/sdash/NSO/Work/GIT-Projects/sharps-bmrs-db/sharps-bmrs-db",
)

# --- ensemble axes ----------------------------------------------------------
# du extends BELOW the previous box (the old optimum was pinned at its 10 m/s
# edge); eta keeps its span because tau, not eta, should now carry the amplitude.
DUS = [8.0, 10.0, 12.0, 14.0, 16.0]                 # m/s
ETAS = [3.5e8, 4.5e8, 5.5e8, 6.5e8]                 # m^2/s (350..650 km^2/s)
TAUS = [2.0, 3.0, 5.0, 8.0, 15.0]                   # yr   (0 => no decay)

NTHETA, NPHI = 181, 360
SNAP_EVERY = 54                    # days (~2 rotations) -- halves the disk cost
DIPCAP = (1.0 + np.sin(np.deg2rad(60.0))) / 2.0     # <cos> over a >60 deg cap

# tolerances: each term contributes ~1.0 to J at its tolerance
TOL_DT, TOL_CORR, TOL_DIP, TOL_CONC = 0.5, 0.10, 0.5, 0.5
DIP_BAND = (3.0, 4.0)              # expected peak axial dipole [G]
CONC_OK = 1.25                     # concentration below this is unpenalised


# ---------------------------------------------------------------- scoring ---
def score3(res, obs, cap_area_cm2):
    """All metrics for one member, with both the robust and raw-amplitude scores."""
    t, pn, ps, dip = res["yr"], res["pn"], res["ps"], res["dip"]
    out = {}

    for lab, model, o in (("N", pn, obs["n"]), ("S", ps, obs["s"])):
        oi = np.interp(t, obs["t"], o)
        out[f"corr{lab}"] = float(np.corrcoef(model, oi)[0, 1])
        out[f"rmse{lab}"] = float(np.sqrt(np.mean((model - oi) ** 2)))

    for lab, model, win, key in (("N24", pn, REV_WIN_24, "revN24"),
                                 ("S24", ps, REV_WIN_24, "revS24"),
                                 ("N25", pn, REV_WIN_25, "revN25"),
                                 ("S25", ps, REV_WIN_25, "revS25")):
        tm = reversal_time(t, model, win)
        out[f"rev{lab}"] = tm
        out[f"dt{lab}"] = tm - obs[key] if np.isfinite(tm) else np.nan

    for lab, model, oamp in (("N", pn, obs["ampN"]), ("S", ps, obs["ampS"])):
        a, tp = plateau_amplitude(t, model, PLATEAU_WIN)
        out[f"amp{lab}"], out[f"tamp{lab}"] = a, tp
        out[f"ratio{lab}"] = a / oamp if (np.isfinite(a) and oamp) else np.nan
        out[f"flux{lab}"] = a * cap_area_cm2
        out[f"obsflux{lab}"] = oamp * cap_area_cm2

    # amplitude + distribution, on the observables free of the HMI cap offset
    peakD = float(np.nanmax(np.abs(dip)))
    i = int(np.nanargmax(np.abs(dip)))
    capmean = 0.5 * (abs(pn[i]) + abs(ps[i]))
    conc = capmean / (abs(dip[i]) * DIPCAP) if dip[i] else np.nan
    out["peakD"], out["tpeakD"], out["conc"] = peakD, float(t[i]), float(conc)

    dts = [abs(out[f"dt{k}"]) for k in ("N24", "S24", "N25", "S25")
           if np.isfinite(out[f"dt{k}"])]
    n_missing = 4 - len(dts)
    cors = [out["corrN"], out["corrS"]]
    rats = [abs(r - 1.0) for r in (out["ratioN"], out["ratioS"]) if np.isfinite(r)]

    out["n_missing_rev"] = n_missing
    out["J_time"] = (np.mean(dts) if dts else 0.0) / TOL_DT + 2.0 * n_missing
    out["J_shape"] = (1.0 - np.mean(cors)) / TOL_CORR
    out["J_dip"] = max(0.0, abs(peakD - np.mean(DIP_BAND))
                       - 0.5 * (DIP_BAND[1] - DIP_BAND[0])) / TOL_DIP
    out["J_conc"] = (max(0.0, conc - CONC_OK) / TOL_CONC
                     if np.isfinite(conc) else 4.0)
    out["J_amp_raw"] = (np.mean(rats) if rats else 4.0) / 0.25

    out["J_robust"] = out["J_time"] + out["J_shape"] + out["J_dip"] + out["J_conc"]
    out["J_rawamp"] = out["J_time"] + out["J_amp_raw"] + out["J_shape"]
    return out


# ------------------------------------------------------------ one member ----
def member_tag(profile, du, eta, tau, flux_scale=1.0):
    """Filesystem-safe identifier for one ensemble member.

    Shared by the in-repo ensemble driver and the external cluster sweep so both
    agree on output filenames (and so ``--resume`` can recognise finished work).
    """
    pfx = "y20" if str(profile).startswith("yeates") else "sb"
    tag = f"{pfx}_du{du:04.1f}_eta{eta/1e6:03.0f}_tau{tau:04.1f}"
    if not np.isclose(flux_scale, 1.0):
        tag += f"_fs{flux_scale:0.2f}"
    return tag.replace(".", "p")


def run_member3(cfg):
    """Run one (du, eta, tau) member, save snapshots + series, return metrics."""
    import warnings
    warnings.filterwarnings("ignore")

    from sft2d.analysis.analysis import (calculate_dm, calculate_polar_field,
                                         calculate_polar_flux, calculate_usflx,
                                         cap_areas)
    from sft2d.data import HMI_SYNOPTIC_FITS
    from sft2d.src.grid import create_grid
    from sft2d.src.initial_conditions import initialize_field
    from sft2d.src.sharp_patch_driver import SHARPPatchSource
    from sft2d.src.stepper import evolve
    from sft2d.src.transport_profiles import differential_rotation, meridional_flow

    du, eta, tau, profile = cfg["du"], cfg["eta"], cfg["tau"], cfg["profile"]
    flux_scale = float(cfg.get("flux_scale", 1.0))
    t_start = time.time()

    grid = create_grid(cfg["ntheta"], cfg["nphi"])
    mf = meridional_flow(grid, peak_speed=du, profile=profile)
    dr = differential_rotation(grid)
    field = initialize_field(grid, "read", path=str(HMI_SYNOPTIC_FITS))
    src = SHARPPatchSource(cfg["catalogue"], nc_dir=cfg["nc_dir"],
                           start_date=START, end_date=END, flux_scale=flux_scale)

    yr, pn, ps, dip, usf, fn, fs, bfly = [], [], [], [], [], [], [], []
    snap_br, snap_day = [], []
    rec_every, snap_every, cap = cfg["rec_every"], cfg["snap_every"], CAP
    keep_snaps = snap_every > 0

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
            if keep_snaps and day % snap_every == 0:
                snap_br.append(B.astype(np.float32).copy()); snap_day.append(day)

    evolve(field, grid, mf, dr, eta, src.num_days, source=src,
           tau_decay_s=(tau * YEAR if tau else None), recorder=Rec())

    res = dict(yr=np.array(yr), pn=np.array(pn), ps=np.array(ps),
               dip=np.array(dip), usf=np.array(usf),
               fluxN=np.array(fn), fluxS=np.array(fs), bfly=np.array(bfly).T)

    tag = member_tag(profile, du, eta, tau, flux_scale)
    out = Path(cfg["outdir"]) / f"member_{tag}.npz"
    payload = dict(
        yr=res["yr"], pn=res["pn"], ps=res["ps"], dip=res["dip"], usf=res["usf"],
        polar_flux_n=res["fluxN"], polar_flux_s=res["fluxS"], bfly=res["bfly"],
        lat=np.rad2deg(np.pi / 2 - grid["colatitude"]),
        lon=np.rad2deg(grid["longitude"]),
        du=du, eta=eta, tau_years=tau, cap_deg=cap, profile=profile,
        flux_scale=flux_scale,
    )
    if keep_snaps:
        payload.update(br=np.array(snap_br), snap_day=np.asarray(snap_day),
                       snap_year=T0 + np.asarray(snap_day) / 365.25)
    np.savez_compressed(out, **payload)

    an, _ = cap_areas(grid, cap, cm=True)
    m = score3(res, cfg["obs"], float(np.sum(an)) * grid["n_phi"])
    m.update(du=du, eta=eta, tau=tau, profile=profile, flux_scale=flux_scale,
             tag=tag, file=out.name, seconds=time.time() - t_start,
             mb=out.stat().st_size / 1e6)
    return m


# --------------------------------------------------------------- reporting --
COLS = ["profile", "du", "eta", "tau", "J_robust", "J_rawamp", "J_time", "J_shape", "J_dip",
        "J_conc", "J_amp_raw", "corrN", "corrS", "dtN24", "dtS24", "dtN25", "dtS25",
        "revN24", "revS24", "revN25", "revS25", "ampN", "ampS", "ratioN", "ratioS",
        "fluxN", "fluxS", "peakD", "conc", "rmseN", "rmseS", "n_missing_rev",
        "seconds", "mb", "file"]


def _f(x, spec="%+.2f"):
    return "--" if x is None or not np.isfinite(x) else spec % x


def edge_report(best, dus, etas, taus):
    """Name any axis whose best value sits on the scanned boundary.

    Only meaningful for axes with >=3 values: with two points every optimum is
    trivially on an edge, so those are skipped rather than reported as pinned.
    """
    warn = []
    for name, val, axis, unit in (("du", best["du"], dus, "m/s"),
                                  ("eta", best["eta"] / 1e6, [e / 1e6 for e in etas], "km^2/s"),
                                  ("tau", best["tau"], taus, "yr")):
        if len(axis) < 3:
            continue
        if val <= min(axis):
            warn.append(f"{name} is pinned at the LOW edge ({val:g} {unit}) -- extend below it")
        elif val >= max(axis):
            warn.append(f"{name} is pinned at the HIGH edge ({val:g} {unit}) -- extend above it")
    return warn


def write_outputs(rows, obs, outdir, args, dus, etas, taus):
    outdir = Path(outdir)
    key = "J_robust" if args.objective == "robust" else "J_rawamp"
    rows = sorted(rows, key=lambda r: (np.inf if not np.isfinite(r[key]) else r[key]))

    with open(outdir / "ensemble3_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    b = rows[0]
    warn = edge_report(b, dus, etas, taus)
    md = [
        "# Three-parameter ensemble: du x eta x tau vs HMI polar field",
        "",
        f"Direct HMI SHARP patch insertion (`SHARPPatchSource`), {args.ntheta}x{args.nphi}, "
        f"{START} to {END}, cap poleward of {90-CAP:g} deg. {len(rows)} members.",
        f"Ranked by **{key}** (lower = better).",
        "",
        "`J_robust = J_time + J_shape + J_dip + J_conc` -- amplitude is judged by the "
        "axial dipole (3-4 G band) and the polar concentration, both free of the "
        "~1.6-2x HMI cap-field metric offset. `J_rawamp` is the previous "
        "cap-amplitude-driven score, kept for comparison.",
        "",
        "Observed (HMI): reversals N24 = %.2f, S24 = %.2f, N25 = %.2f, S25 = %.2f; "
        "plateau amplitude N = %+.2f G, S = %+.2f G."
        % (obs["revN24"], obs["revS24"], obs["revN25"], obs["revS25"],
           obs["ampN"], obs["ampS"]),
        "",
    ]
    if warn:
        md += ["> **Edge warning:** " + "; ".join(warn) + ".", ""]

    md += [
        "| rank | flow | du | eta | tau | **J_rob** | J_time | J_shape | J_dip | J_conc "
        "| corr N/S | dt N24/S24 | dt N25/S25 | peak D | conc | amp N/S [G] "
        "| ratio N/S | J_raw |",
        "|---:|:--|---:|---:|---:|---:|--:|--:|--:|--:|:--:|:--:|:--:|--:|--:|:--:|:--:|--:|",
    ]
    for i, r in enumerate(rows, 1):
        md.append(
            f"| {i} | {'y20' if r['profile'].startswith('yeates') else 'SB'} "
            f"| {r['du']:.0f} | {r['eta']/1e6:.0f} | {r['tau']:g} "
            f"| **{r['J_robust']:.2f}** | {r['J_time']:.2f} | {r['J_shape']:.2f} "
            f"| {r['J_dip']:.2f} | {r['J_conc']:.2f} "
            f"| {_f(r['corrN'],'%.3f')}/{_f(r['corrS'],'%.3f')} "
            f"| {_f(r['dtN24'])}/{_f(r['dtS24'])} "
            f"| {_f(r['dtN25'])}/{_f(r['dtS25'])} "
            f"| {_f(r['peakD'],'%.2f')} | {_f(r['conc'],'%.2f')} "
            f"| {_f(r['ampN'])}/{_f(r['ampS'])} "
            f"| {_f(r['ratioN'],'%.2f')}/{_f(r['ratioS'],'%.2f')} "
            f"| {r['J_rawamp']:.2f} |"
        )

    md += [
        "",
        "## Best member",
        "",
        f"**{b['profile']} flow, du = {b['du']:g} m/s, eta = {b['eta']/1e6:.0f} km^2/s, "
        f"tau = {b['tau']:g} yr** "
        f"-- J_robust = {b['J_robust']:.2f} (J_rawamp = {b['J_rawamp']:.2f})",
        "",
        f"* correlation with HMI: N {b['corrN']:+.3f}, S {b['corrS']:+.3f}",
        f"* cycle-24 reversal error: N {_f(b['dtN24'])} yr, S {_f(b['dtS24'])} yr",
        f"* cycle-25 reversal error: N {_f(b['dtN25'])} yr, S {_f(b['dtS25'])} yr",
        f"* peak axial dipole: {b['peakD']:.2f} G (target {DIP_BAND[0]:g}-{DIP_BAND[1]:g} G)",
        f"* polar concentration: {b['conc']:.2f} (1.0 = dipole-like)",
        f"* plateau amplitude: N {_f(b['ampN'])} G (obs {obs['ampN']:+.2f}, "
        f"ratio {_f(b['ratioN'],'%.2f')}), S {_f(b['ampS'])} G "
        f"(obs {obs['ampS']:+.2f}, ratio {_f(b['ratioS'],'%.2f')})",
        f"* polar flux: N {_f(b['fluxN']/1e22)}e22 Mx, S {_f(b['fluxS']/1e22)}e22 Mx",
        f"* snapshots: `{b['file']}`",
        "",
        "### Best on each individual criterion",
        "",
        "| criterion | best (du, eta, tau) | value |",
        "|:--|:--|:--|",
    ]
    crit = [
        ("mean |reversal-time error| [yr]",
         lambda r: np.nanmean([abs(r[k]) for k in ("dtN24", "dtS24", "dtN25", "dtS25")]), "%.3f", False),
        ("mean correlation (N,S)", lambda r: -0.5 * (r["corrN"] + r["corrS"]), "%.3f", True),
        ("|peak dipole - 3.5 G|", lambda r: abs(r["peakD"] - 3.5), "%.3f", False),
        ("polar concentration (-> 1)", lambda r: abs(r["conc"] - 1.0), "%.3f", False),
        ("cap-amplitude ratio error", lambda r: np.nanmean(
            [abs(r["ratioN"] - 1), abs(r["ratioS"] - 1)]), "%.3f", False),
    ]
    for name, fn_, spec, neg in crit:
        vals = [(fn_(r), r) for r in rows if np.isfinite(fn_(r))]
        if not vals:
            continue
        v, r = min(vals, key=lambda x: x[0])
        md.append(f"| {name} | {r['profile']}, du={r['du']:g}, eta={r['eta']/1e6:.0f}, "
                  f"tau={r['tau']:g} | {spec % (-v if neg else v)} |")

    # marginal best over each axis -- shows whether the optimum is interior
    md += ["", "### Marginal best J_robust along each axis", "",
           "| axis | value | best J_robust |", "|:--|--:|--:|"]
    axes_spec = [("du [m/s]", lambda r: r["du"], dus),
                 ("eta [km^2/s]", lambda r: r["eta"] / 1e6, [e / 1e6 for e in etas]),
                 ("tau [yr]", lambda r: r["tau"], taus)]
    for name, getter, vals in axes_spec:
        for v in vals:
            sel = [r["J_robust"] for r in rows if np.isclose(getter(r), v)]
            if sel:
                md.append(f"| {name} | {v:g} | {min(sel):.2f} |")
    profiles = sorted({r["profile"] for r in rows})
    if len(profiles) > 1:
        for pname in profiles:
            sel = [r["J_robust"] for r in rows if r["profile"] == pname]
            md.append(f"| flow profile | {pname} | {min(sel):.2f} |")

    md += ["", "> The HMI polar-cap field carries a known ~1.6-2x line-of-sight/metric",
           "> offset; the axial dipole does not. `ratio N/S` is reported but not scored",
           "> in `J_robust` -- see the module docstring for why.", ""]
    (outdir / "ensemble3_summary.md").write_text("\n".join(md))
    return rows, warn


def make_figure(rows, obs, outdir, dus, etas, taus, key="J_robust"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    best = rows[0]

    def slice_surf(xs, ys, xget, yget, fixed):
        A = np.full((len(ys), len(xs)), np.nan)
        for r in rows:
            if r["profile"] != best["profile"]:
                continue
            if not all(np.isclose(g(r), v) for g, v in fixed):
                continue
            try:
                A[ys.index(yget(r)), xs.index(xget(r))] = r[key]
            except ValueError:
                pass
        return A

    etas_km = [e / 1e6 for e in etas]
    panels = [
        ("du x eta  (tau = %g yr)" % best["tau"], dus, etas_km,
         lambda r: r["du"], lambda r: r["eta"] / 1e6,
         [(lambda r: r["tau"], best["tau"])], "du [m/s]", "eta [km^2/s]"),
        ("du x tau  (eta = %g)" % (best["eta"] / 1e6), dus, taus,
         lambda r: r["du"], lambda r: r["tau"],
         [(lambda r: r["eta"], best["eta"])], "du [m/s]", "tau [yr]"),
        ("eta x tau  (du = %g m/s)" % best["du"], etas_km, taus,
         lambda r: r["eta"] / 1e6, lambda r: r["tau"],
         [(lambda r: r["du"], best["du"])], "eta [km^2/s]", "tau [yr]"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))
    for ax, (title, xs, ys, xg, yg, fixed, xl, yl) in zip(axes[0], panels):
        A = slice_surf(xs, ys, xg, yg, fixed)
        im = ax.imshow(A, origin="lower", aspect="auto", cmap="viridis_r")
        ax.set_xticks(range(len(xs))); ax.set_xticklabels([f"{v:g}" for v in xs])
        ax.set_yticks(range(len(ys))); ax.set_yticklabels([f"{v:g}" for v in ys])
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(f"{key}: {title}", fontsize=10)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if np.isfinite(A[i, j]):
                    ax.text(j, i, f"{A[i,j]:.1f}", ha="center", va="center",
                            fontsize=7, color="w")
        fig.colorbar(im, ax=ax)

    # marginals -- an interior minimum here means the box is wide enough
    axm = axes[1, 0]
    for lab, getter, vals in (("du [m/s]", lambda r: r["du"], dus),
                              ("eta [100 km^2/s]", lambda r: r["eta"] / 1e8, [e / 1e8 for e in etas]),
                              ("tau [yr]", lambda r: r["tau"], taus)):
        xs, ys = [], []
        for v in vals:
            sel = [r[key] for r in rows if np.isclose(getter(r), v)]
            if sel:
                xs.append(v); ys.append(min(sel))
        axm.plot(xs, ys, "o-", label=lab)
    axm.set_xlabel("parameter value"); axm.set_ylabel(f"best {key}")
    axm.set_title("marginal best (interior minimum = box wide enough)", fontsize=10)
    profiles = sorted({r["profile"] for r in rows})
    if len(profiles) > 1:
        for k, pname in enumerate(profiles):
            sel = [r[key] for r in rows if r["profile"] == pname]
            axm.axhline(min(sel), ls=":", color=f"C{k+3}",
                        label=f"best {pname}: {min(sel):.2f}")
    axm.legend(fontsize=8)

    z = np.load(Path(outdir) / best["file"])
    for ax, (lab, mk, ok_) in zip(axes[1, 1:], (("north", "pn", "n"), ("south", "ps", "s"))):
        ax.plot(obs["t"], obs[ok_], "k", lw=1, label="HMI")
        ax.plot(z["yr"], z[mk], "C3", lw=1.4,
                label=f"du={best['du']:g}, eta={best['eta']/1e6:.0f}, tau={best['tau']:g}")
        ax.axhline(0, color="k", lw=0.4)
        ax.set_xlabel("year"); ax.set_ylabel(f"{lab} cap field [G]")
        ax.set_title(f"{lab} polar field -- best member vs HMI", fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle("Three-parameter ensemble (du, eta, tau) vs HMI -- "
                 "direct SHARP patch insertion", y=1.00)
    fig.tight_layout()
    fig.savefig(Path(outdir) / "ensemble3_polarfield.png", dpi=140, bbox_inches="tight")


# -------------------------------------------------------------------- main --
def main(argv=None):
    p = argparse.ArgumentParser(description="3-parameter (du, eta, tau) SFT ensemble")
    p.add_argument("--du", type=float, nargs="+", default=DUS)
    p.add_argument("--eta", type=float, nargs="+", default=ETAS)
    p.add_argument("--tau", type=float, nargs="+", default=TAUS,
                   help="flux-decay times [yr]; 0 means no decay")
    p.add_argument("--profile", nargs="+", default=["yeates2020"],
                   choices=["yeates2020", "schuessler-baumann"],
                   help="meridional-flow shape(s); pass both to add it as a 4th axis. "
                        "The SB profile falls off faster near the poles and is the one "
                        "that fixed polar over-concentration in the earlier calibration, "
                        "which (du, eta, tau) alone demonstrably cannot.")
    p.add_argument("--objective", choices=["robust", "rawamp"], default="robust")
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--outdir", default="ensemble3_out")
    p.add_argument("--ntheta", type=int, default=NTHETA)
    p.add_argument("--nphi", type=int, default=NPHI)
    p.add_argument("--snap-every", type=int, default=SNAP_EVERY,
                   help="snapshot cadence [days]; 0 saves series only (no B_r maps)")
    p.add_argument("--workers", type=int, default=max(1, min(12, (os.cpu_count() or 4) - 2)))
    p.add_argument("--coarse", action="store_true",
                   help="3x3x3 first pass at 181x360 (~40 min) to locate the basin")
    p.add_argument("--quick", action="store_true",
                   help="2x2x2 smoke test at 91x180 (~3 min)")
    args = p.parse_args(argv)

    if args.coarse:
        args.du, args.eta, args.tau = [8.0, 12.0, 16.0], [3.5e8, 5.0e8, 6.5e8], [3.0, 8.0, 15.0]
        args.profile = ["yeates2020", "schuessler-baumann"]
        args.outdir = args.outdir + "_coarse"
    if args.quick:
        args.du, args.eta, args.tau = [10.0, 16.0], [3.5e8, 6.5e8], [3.0, 10.0]
        args.ntheta, args.nphi, args.snap_every = 91, 180, 180
        args.outdir = args.outdir + "_quick"

    catalogue = str(Path(args.db) / "bmrsharps_evol.txt")
    if not Path(catalogue).exists():
        sys.exit(f"catalogue not found: {catalogue}  (set --db or SFT2D_SHARPS_DB)")
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    obs = _observed()
    dus, etas, taus, profs = list(args.du), list(args.eta), list(args.tau), list(args.profile)
    members = [dict(du=d, eta=e, tau=tv, profile=pf, catalogue=catalogue, nc_dir=args.db,
                    ntheta=args.ntheta, nphi=args.nphi, rec_every=REC_EVERY,
                    snap_every=args.snap_every, outdir=str(outdir), obs=obs)
               for pf in profs for tv in taus for e in etas for d in dus]

    # cost estimate: ~850 s/member at 181x360, ~110 s at 91x180; snapshots at
    # 54-day cadence are ~25 MB/member at 181x360 and scale inversely with cadence
    per = 850.0 if args.ntheta >= 181 else 110.0
    mb = 25.0 * (54.0 / args.snap_every) if args.snap_every else 1.0
    if args.ntheta < 181:
        mb /= 4.0
    axes_txt = (f"{len(dus)} du x {len(etas)} eta x {len(taus)} tau"
                + (f" x {len(profs)} profile" if len(profs) > 1 else ""))
    print(f"3-parameter ensemble: {len(members)} members ({axes_txt}), "
          f"{args.ntheta}x{args.nphi}, {args.workers} workers")
    print(f"  du  [m/s]    : {', '.join(f'{v:g}' for v in dus)}")
    print(f"  eta [km^2/s] : {', '.join(f'{e/1e6:g}' for e in etas)}")
    print(f"  tau [yr]     : {', '.join(('none' if t == 0 else f'{t:g}') for t in taus)}")
    print(f"  flow profile : {', '.join(profs)}")
    print(f"  objective    : {args.objective}")
    print(f"  observed reversals: N24={obs['revN24']:.2f} S24={obs['revS24']:.2f} "
          f"N25={obs['revN25']:.2f} S25={obs['revS25']:.2f}")
    print(f"  estimated    : ~{len(members)*per/args.workers/60:.0f} min wall, "
          f"~{len(members)*mb/1000:.1f} GB on disk")
    print(f"  output -> {outdir}/\n", flush=True)

    t0 = time.time()
    rows = []
    if args.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for m in pool.imap_unordered(run_member3, members):
                rows.append(m)
                print(f"  [{len(rows):3d}/{len(members)}] "
                      f"{'y20' if m['profile'].startswith('yeates') else ' SB'} "
                      f"du={m['du']:4.1f} eta={m['eta']/1e6:3.0f} tau={m['tau']:4.1f}  "
                      f"J_rob={m['J_robust']:6.2f}  D={m['peakD']:.2f}G  "
                      f"conc={m['conc']:.2f}  corr={0.5*(m['corrN']+m['corrS']):+.3f}"
                      f"  ({m['seconds']:.0f}s)", flush=True)
    else:
        for cfg in members:
            m = run_member3(cfg); rows.append(m)
            print(f"  [{len(rows):3d}/{len(members)}] J_rob={m['J_robust']:.2f}", flush=True)

    rows, warn = write_outputs(rows, obs, outdir, args, dus, etas, taus)
    make_figure(rows, obs, outdir, dus, etas, taus,
                key="J_robust" if args.objective == "robust" else "J_rawamp")

    print(f"\nwall time {(time.time()-t0)/60:.1f} min, "
          f"{sum(r['mb'] for r in rows)/1000:.2f} GB of output")
    print(f"wrote {outdir}/ensemble3_results.csv, ensemble3_summary.md, "
          f"ensemble3_polarfield.png\n")

    b = rows[0]
    print("BEST (%s): %s flow, du=%g m/s, eta=%g km^2/s, tau=%g yr  ->  J_robust=%.2f "
          "(J_rawamp=%.2f)" % (args.objective, b["profile"], b["du"], b["eta"] / 1e6,
                               b["tau"], b["J_robust"], b["J_rawamp"]))
    print("  corr N/S      %+.3f / %+.3f" % (b["corrN"], b["corrS"]))
    print("  dt reversal   N24 %s  S24 %s  N25 %s  S25 %s yr"
          % tuple(_f(b[k]) for k in ("dtN24", "dtS24", "dtN25", "dtS25")))
    print("  peak dipole   %.2f G  (target %g-%g)" % (b["peakD"], *DIP_BAND))
    print("  concentration %.2f    (1.0 = dipole-like)" % b["conc"])
    print("  amplitude N/S %+.2f / %+.2f G (obs %+.2f / %+.2f, ratio %.2f / %.2f)"
          % (b["ampN"], b["ampS"], obs["ampN"], obs["ampS"], b["ratioN"], b["ratioS"]))
    if warn:
        print("\n  EDGE WARNING: " + "; ".join(warn))
        print("  -> re-run with that axis widened; the optimum is outside the box.")


if __name__ == "__main__":
    main()
