"""
scan_sharps_calibration.py

Parameter scan around the calibrated "Yeates" recipe (Schuessler-Baumann
meridional flow + eta=500 km^2/s + tau=10 yr flux decay), driving the model from
a SHARPS catalogue and scoring each run against the observed HMI polar field.

    python examples/scan_sharps_calibration.py path/to/bmrsharps_evol_all.txt

It runs two slices that share the recipe centre (~13 forward runs, a few minutes
each -- budget ~15-20 min total at 91x180):

* an ``(eta, v0)`` grid at the fixed calibrated ``tau``;
* a ``tau`` sensitivity slice at the recipe ``(eta, v0)``.

Metrics per run
---------------
* ``peakD``  : peak axial dipole moment [G] -- the robust amplitude observable;
  the target band is ~3-4 G.
* ``conc``   : polar-cap-mean / dipole-equivalent at cycle max (1.0 = dipole-like,
  >1 = polar flux over-concentration / pile-up).
* ``corr``   : correlation of the model polar field with HMI (N & S averaged) --
  captures reversal timing and shape.
* ``rms*``   : RMS of the model polar field vs HMI after the single best-fit
  scale (scale-robust, so the known HMI-metric amplitude offset does not bias
  the shape/timing comparison).
* ``obj``    : a composite score (lower = better), balancing scaled shape misfit,
  the axial-dipole distance from mid-band (3.5 G), and the concentration.

Outputs a sorted table (also ``scan_results.csv``) and ``scan_sharps.png``
(the (eta, v0) objective surface, the tau slice, and the best run vs HMI).
"""

import sys
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sft2d.analysis.analysis import calculate_dm, calculate_polar_field
from sft2d.data import HMI_SYNOPTIC_FITS, load_hmi_polar_field
from sft2d.src.grid import create_grid
from sft2d.src.initial_conditions import initialize_field
from sft2d.src.sharp_driver import SHARPSource
from sft2d.src.stepper import evolve
from sft2d.src.transport_profiles import differential_rotation, meridional_flow

np.seterr(all="ignore")
YEAR = 365.25 * 86400.0

# --- scan ranges (edit here) ---------------------------------------------
ETAS = [4.0e8, 5.0e8, 6.0e8]        # m^2/s  (400, 500, 600 km^2/s)
V0S = [9.0, 11.0, 13.0]             # m/s    (Schuessler-Baumann peak)
TAUS = [6.0, 8.0, 10.0, 12.0, 15.0]  # yr    (decay slice)
ETA_C, V0_C, TAU_C = 5.0e8, 11.0, 10.0   # recipe centre
CAP = 30.0                          # analyse poleward of 60 deg (HMI definition)
S0 = np.sin(np.deg2rad(60)); DIP_CAPMEAN = (1 + S0) / 2   # >60 mean of unit dipole


def _score(pn, ps, yr, hyr, mn, ms):
    def one(model, obs):
        o = np.interp(yr, hyr, obs)
        corr = np.corrcoef(model, o)[0, 1]
        scale = np.sum(model * o) / np.sum(model * model)
        rms = np.sqrt(np.mean((scale * model - o) ** 2))
        return corr, scale, rms
    cN, scN, rN = one(pn, mn)
    cS, scS, rS = one(ps, ms)
    return dict(corrN=cN, corrS=cS, scaleN=scN, scaleS=scS, rmsN=rN, rmsS=rS,
                corr=0.5 * (cN + cS))


def run_one(grid, dr, field0, src, hmi, v0, eta, tau_years):
    mf = meridional_flow(grid, peak_speed=v0, profile="schuessler-baumann")
    tau_s = tau_years * YEAR if tau_years else None
    yr, pn, ps, dip = [], [], [], []
    box = {"mid": None}
    tmid = int((2016.5 - 2010.33) * 365.25)

    class Rec:
        def record(self, day, B):
            if day % 10 == 0:
                yr.append(2010 + 0.33 + day / 365.25)
                n, s = calculate_polar_field(B, grid, pol_cap_extent_deg=CAP)
                pn.append(n); ps.append(s); dip.append(calculate_dm(B, grid))
            if box["mid"] is None and day >= tmid:
                box["mid"] = B.copy()

    evolve(field0.copy(), grid, mf, dr, eta, src.num_days, source=src,
           tau_decay_s=tau_s, recorder=Rec())
    yr = np.array(yr); pn = np.array(pn); ps = np.array(ps); dip = np.array(dip)

    hyr, mn, ms = hmi
    m = _score(pn, ps, yr, hyr, mn, ms)
    Dmid = np.interp(2016.5, yr, dip); nmid = np.interp(2016.5, yr, pn)
    peakD = np.abs(dip).max()
    conc = nmid / (Dmid * DIP_CAPMEAN)
    obj = 0.5 * (m["rmsN"] + m["rmsS"]) + 0.5 * abs(peakD - 3.5) + 0.3 * abs(conc - 1.0)
    return dict(v0=v0, eta=eta, tau=tau_years, peakD=peakD, conc=conc,
                obj=obj, yr=yr, pn=pn, ps=ps, **m)


def main(catalogue):
    grid = create_grid(91, 180)
    dr = differential_rotation(grid)
    field0 = initialize_field(grid, "read", path=str(HMI_SYNOPTIC_FITS))
    src = SHARPSource(catalogue, start_date="2010-05-01", end_date="2023-09-01", flux_scale=1.0)
    print(src.summary())
    h = load_hmi_polar_field(); idx = h["mean_north"].index
    hmi = (idx.year + (idx.dayofyear - 1) / 365.25, h["mean_north"].values, h["mean_south"].values)

    rows = []
    # slice A: (eta, v0) at tau_C
    for eta in ETAS:
        for v0 in V0S:
            t0 = time.time(); r = run_one(grid, dr, field0, src, hmi, v0, eta, TAU_C)
            rows.append(r)
            print(f"  eta={eta/1e6:3.0f} v0={v0:4.1f} tau={TAU_C:4.1f}  "
                  f"peakD={r['peakD']:.2f} conc={r['conc']:.2f} corr={r['corr']:+.2f} "
                  f"obj={r['obj']:.3f}  ({time.time()-t0:.0f}s)")
    # slice B: tau at (eta_C, v0_C)
    for tau in TAUS:
        if tau == TAU_C:
            continue
        t0 = time.time(); r = run_one(grid, dr, field0, src, hmi, V0_C, ETA_C, tau)
        rows.append(r)
        print(f"  eta={ETA_C/1e6:3.0f} v0={V0_C:4.1f} tau={tau:4.1f}  "
              f"peakD={r['peakD']:.2f} conc={r['conc']:.2f} corr={r['corr']:+.2f} "
              f"obj={r['obj']:.3f}  ({time.time()-t0:.0f}s)")

    rows.sort(key=lambda r: r["obj"])
    _write_csv(rows)
    print("\nranked (best first):")
    print(f"  {'eta':>4} {'v0':>5} {'tau':>5} {'peakD':>6} {'conc':>6} {'corr':>6} "
          f"{'scaleN':>7} {'obj':>6}")
    for r in rows:
        flag = "  <- dipole in band & ~dipole-like" if (3 <= r["peakD"] <= 4 and 0.85 <= r["conc"] <= 1.3) else ""
        print(f"  {r['eta']/1e6:4.0f} {r['v0']:5.1f} {r['tau']:5.1f} {r['peakD']:6.2f} "
              f"{r['conc']:6.2f} {r['corr']:+6.2f} {r['scaleN']:7.3f} {r['obj']:6.3f}{flag}")

    _figure(rows, hmi)


def _write_csv(rows):
    import csv
    keys = ["eta", "v0", "tau", "peakD", "conc", "corr", "corrN", "corrS",
            "scaleN", "scaleS", "rmsN", "rmsS", "obj"]
    with open("scan_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})
    print("wrote scan_results.csv")


def _figure(rows, hmi):
    hyr, mn, ms = hmi
    best = rows[0]
    fig = plt.figure(figsize=(13, 4.5))
    gsA = fig.add_subplot(1, 3, 1)
    gsB = fig.add_subplot(1, 3, 2)
    gsC = fig.add_subplot(1, 3, 3)

    # (eta, v0) objective surface at tau_C
    A = np.full((len(ETAS), len(V0S)), np.nan)
    for r in rows:
        if r["tau"] == TAU_C and r["eta"] in ETAS and r["v0"] in V0S:
            A[ETAS.index(r["eta"]), V0S.index(r["v0"])] = r["obj"]
    im = gsA.imshow(A, origin="lower", aspect="auto", cmap="viridis_r",
                    extent=[min(V0S) - 1, max(V0S) + 1, min(ETAS) / 1e6 - 50, max(ETAS) / 1e6 + 50])
    gsA.set_xticks(V0S); gsA.set_yticks([e / 1e6 for e in ETAS])
    gsA.set_xlabel("v0 [m/s]"); gsA.set_ylabel("eta [km^2/s]")
    gsA.set_title(f"objective (lower=better) at tau={TAU_C:g} yr")
    fig.colorbar(im, ax=gsA)

    # tau slice
    tau_rows = sorted([r for r in rows if r["eta"] == ETA_C and r["v0"] == V0_C],
                      key=lambda r: r["tau"])
    gsB.plot([r["tau"] for r in tau_rows], [r["obj"] for r in tau_rows], "o-", label="objective")
    gsB.plot([r["tau"] for r in tau_rows], [r["peakD"] for r in tau_rows], "s--", label="peak dipole [G]")
    gsB.axhspan(3, 4, color="0.85", alpha=0.6)
    gsB.set_xlabel("tau [yr]"); gsB.set_title(f"tau slice at eta={ETA_C/1e6:g}, v0={V0_C:g}")
    gsB.legend(fontsize=8)

    # best run vs HMI
    gsC.plot(hyr, mn, "k", lw=1, label="HMI N"); gsC.plot(hyr, ms, "k--", lw=1, label="HMI S")
    gsC.plot(best["yr"], best["pn"], "C3", lw=1.8, label="model N")
    gsC.plot(best["yr"], best["ps"], "C0", lw=1.8, label="model S")
    gsC.axhline(0, color="k", lw=0.4); gsC.set_xlabel("year"); gsC.set_ylabel("polar field (>60) [G]")
    gsC.set_title(f"best: eta={best['eta']/1e6:g}, v0={best['v0']:g}, tau={best['tau']:g}")
    gsC.legend(fontsize=7)

    fig.suptitle("SHARPS calibration scan around the Yeates recipe", y=1.02)
    fig.tight_layout(); fig.savefig("scan_sharps.png", dpi=140, bbox_inches="tight")
    print("saved scan_sharps.png")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not args:
        print(__doc__); raise SystemExit("ERROR: provide a bmrsharps_evol catalogue path.")
    main(args[0])
