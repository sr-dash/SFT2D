"""
analyze_sweep.py

Read a finished parameter-sweep output tree and produce the diagnostic figure and
tables that say *what the sweep actually determined* -- which is not the same as
"the member with the lowest objective".

    python examples/analyze_sweep.py "/Volumes/Extreme SSD/SFT2D/sft2d_sweep_out"

The six panels answer six questions, in order:

1. Where is the optimum in (eta, tau)?  A sequential ramp over the objective at
   the best flow profile and du.  Both axes have interior minima, so this surface
   is meaningful.
2. Is du determined?  Marginal best objective against du, one line per flow
   profile.  A monotonic line means the optimum is outside the scanned range and
   the sweep has *not* determined that parameter.
3. Why?  Mean absolute reversal-time error against du.  Reversal timing is the
   dominant term in the objective, and du is its only strong lever.
4. Is the amplitude in range?  Peak axial dipole against du with the 3-4 G band
   shaded -- the amplitude observable free of the HMI cap-field metric offset.
5-6. Does the best member actually track the observations?  North and south polar
   cap field against HMI, for the best member overall and for the best member
   restricted to a physically defensible meridional-flow speed.

Panels 2-4 are deliberately separate small multiples rather than one axes with
three different parameters on a shared x -- that combination is unreadable, which
is what made the sweep's built-in overview figure hard to interpret.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Validated categorical palette (see the dataviz palette checks: lightness band,
# chroma floor, CVD separation, normal-vision floor, contrast all PASS).
C_SB, C_Y20 = "#1b9e77", "#7570b3"        # flow profiles
C_BEST, C_PHYS = "#1f77b4", "#d95f02"     # best overall / best at physical du
INK, MUTED = "#1a1a1a", "#6b6b6b"
DIP_BAND = (3.0, 4.0)
PHYSICAL_DU_MIN = 10.0                     # observed surface flow peaks ~10-20 m/s


def load(sweep_dir):
    import pandas as pd
    d = pd.read_csv(Path(sweep_dir) / "sweep_results.csv")
    if "failed" in d:
        d = d[~d["failed"].astype(bool)]
    d["eta_km"] = (d["eta"] / 1e6).round(0)
    d["flow"] = np.where(d["profile"].str.startswith("yeates"), "y20", "SB")
    d["mean_abs_dt"] = d[["dtN24", "dtS24", "dtN25", "dtS25"]].abs().mean(axis=1)
    return d


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(alpha=0.25, lw=0.6, color=MUTED)
    ax.set_axisbelow(True)
    for lab in (ax.xaxis.label, ax.yaxis.label):
        lab.set_color(INK); lab.set_fontsize(9)
    ax.title.set_color(INK); ax.title.set_fontsize(9.5)


def figure(d, sweep_dir, out, key="J_robust"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sft2d.data import load_hmi_polar_field

    best = d.loc[d[key].idxmin()]
    phys = d[d["du"] >= PHYSICAL_DU_MIN]
    bphys = phys.loc[phys[key].idxmin()] if len(phys) else None

    fig, ax = plt.subplots(2, 3, figsize=(16.5, 9))

    # -- 1. (eta, tau) surface at the best flow + du: sequential, one direction --
    sub = d[(d["flow"] == best["flow"]) & (d["du"] == best["du"])]
    etas = sorted(sub["eta_km"].unique()); taus = sorted(sub["tau"].unique())
    A = np.full((len(taus), len(etas)), np.nan)
    for _, r in sub.iterrows():
        A[taus.index(r["tau"]), etas.index(r["eta_km"])] = r[key]
    a = ax[0, 0]
    vmax = float(np.nanpercentile(A, 70))   # bad members otherwise flatten
    im = a.imshow(A, origin="lower", aspect="auto", cmap="viridis_r",
                  vmin=float(np.nanmin(A)), vmax=vmax)
    a.set_xticks(range(len(etas))); a.set_xticklabels([f"{v:.0f}" for v in etas])
    a.set_yticks(range(len(taus)))
    a.set_yticklabels(["none" if t == 0 else f"{t:g}" for t in taus])
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if np.isfinite(A[i, j]):
                a.text(j, i, f"{A[i,j]:.1f}", ha="center", va="center", fontsize=7.5,
                       color="w" if A[i, j] > 0.55 * vmax else INK)
    a.set_xlabel("eta [km$^2$/s]"); a.set_ylabel(r"tau [yr]")
    a.set_title(f"1. objective at {best['flow']} flow, du = {best['du']:g} m/s\n"
                f"(both axes have interior minima)")
    cb = fig.colorbar(im, ax=a, extend="max")
    cb.set_label(key, color=INK, fontsize=8)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    a.plot(etas.index(best["eta_km"]), taus.index(best["tau"]), "*",
           ms=16, mfc="none", mec="w", mew=2)
    _style(a)

    # -- 2. marginal best vs du, per flow: THE headline problem ---------------
    a = ax[0, 1]
    for flow, col in (("SB", C_SB), ("y20", C_Y20)):
        g = d[d["flow"] == flow].groupby("du")[key].min()
        a.plot(g.index, g.values, "o-", color=col, lw=2, ms=7, label=f"{flow} flow")
        a.annotate(flow, (g.index[-1], g.values[-1]), xytext=(6, 0),
                   textcoords="offset points", color=col, fontsize=9, va="center")
    a.axvspan(d["du"].min() - 0.6, PHYSICAL_DU_MIN, color=MUTED, alpha=0.13, lw=0)
    a.text(d["du"].min() - 0.3, a.get_ylim()[1] * 0.96, "below observed\nflow speeds",
           fontsize=7.5, color=MUTED, va="top")
    a.set_xlabel("du = meridional-flow peak speed [m/s]"); a.set_ylabel(f"best {key}")
    a.set_title("2. du is NOT determined: monotonic to the\nscan floor, which is unphysical")
    a.legend(fontsize=8, frameon=False)
    _style(a)

    # -- 3. why: reversal timing drives the objective ------------------------
    a = ax[0, 2]
    for flow, col in (("SB", C_SB), ("y20", C_Y20)):
        g = d[d["flow"] == flow].groupby("du")["mean_abs_dt"].min()
        a.plot(g.index, g.values, "o-", color=col, lw=2, ms=7, label=f"{flow} flow")
    a.axhline(0.5, color=INK, lw=1, ls="--")
    a.text(d["du"].max(), 0.52, "0.5 yr tolerance", ha="right", fontsize=7.5, color=INK)
    a.axvspan(d["du"].min() - 0.6, PHYSICAL_DU_MIN, color=MUTED, alpha=0.13, lw=0)
    a.set_xlabel("du [m/s]"); a.set_ylabel("mean |reversal-time error| [yr]")
    a.set_title("3. the cause: the model reverses early,\nand du is timing's only strong lever")
    a.legend(fontsize=8, frameon=False)
    _style(a)

    # -- 4. amplitude sanity: peak dipole vs du ------------------------------
    a = ax[1, 0]
    a.axhspan(*DIP_BAND, color=MUTED, alpha=0.18, lw=0)   # reference band:
    #                                        neutral, never a series colour
    a.text(d["du"].max(), DIP_BAND[1] + 0.05, "expected 3-4 G", ha="right",
           fontsize=7.5, color=INK)
    for flow, col in (("SB", C_SB), ("y20", C_Y20)):
        g = d[d["flow"] == flow].groupby("du")["peakD"]
        a.fill_between(g.min().index, g.min().values, g.max().values,
                       color=col, alpha=0.18, lw=0)
        a.plot(g.median().index, g.median().values, "o-", color=col, lw=2, ms=6,
               label=f"{flow} (median, range)")
    a.set_xlabel("du [m/s]"); a.set_ylabel("peak axial dipole [G]")
    a.set_title("4. amplitude is reachable across du\n(so it is not what pins du)")
    a.legend(fontsize=8, frameon=False)
    _style(a)

    # -- 5/6. polar field vs HMI --------------------------------------------
    h = load_hmi_polar_field(); idx = h["mean_north"].index
    ht = np.asarray(idx.year + (idx.dayofyear - 1) / 365.25, float)
    series = [("best overall", best, C_BEST)]
    if bphys is not None and bphys["tag"] != best["tag"]:
        series.append((f"best with du>={PHYSICAL_DU_MIN:g}", bphys, C_PHYS))

    for a, (hemi, mk, obs_key) in zip(ax[1, 1:],
                                      (("north", "pn", "mean_north"),
                                       ("south", "ps", "mean_south"))):
        a.plot(ht, np.asarray(h[obs_key].values, float), color=INK, lw=1.1,
               label="HMI observed")
        for lab, row, col in series:
            f = Path(sweep_dir) / str(row["file"])
            if not f.is_file():
                continue
            z = np.load(f)
            a.plot(z["yr"], z[mk], color=col, lw=1.8,
                   label=f"{lab}: du={row['du']:g}, eta={row['eta_km']:.0f}, "
                         f"tau={row['tau']:g}")
        a.axhline(0, color=MUTED, lw=0.8)
        a.set_xlabel("year"); a.set_ylabel(f"{hemi} cap field (>60$^\\circ$) [G]")
        a.set_title(f"{'5' if hemi=='north' else '6'}. {hemi} polar field vs HMI")
        a.legend(fontsize=7.5, frameon=False, loc="best")
        _style(a)

    fig.suptitle(f"SFT2D parameter sweep: {len(d)} members, direct HMI SHARP patch "
                 f"insertion — ranked by {key}", fontsize=12, color=INK, y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=145, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")
    return best, bphys


def tables(d, key="J_robust"):
    import pandas as pd
    pd.set_option("display.width", 200)
    best = d.loc[d[key].idxmin()]
    print(f"\n=== best overall ===\n{best['flow']} flow, du={best['du']:g} m/s, "
          f"eta={best['eta_km']:.0f} km^2/s, tau={best['tau']:g} yr  ->  {key}={best[key]:.2f}")
    print(f"  corr N/S {best['corrN']:+.3f}/{best['corrS']:+.3f}   peak D {best['peakD']:.2f} G"
          f"   conc {best['conc']:.2f}   mean|dt| {best['mean_abs_dt']:.2f} yr")

    phys = d[d["du"] >= PHYSICAL_DU_MIN]
    if len(phys):
        b = phys.loc[phys[key].idxmin()]
        print(f"\n=== best with a defensible flow speed (du >= {PHYSICAL_DU_MIN:g} m/s) ===")
        print(f"{b['flow']} flow, du={b['du']:g}, eta={b['eta_km']:.0f}, tau={b['tau']:g}"
              f"  ->  {key}={b[key]:.2f}")
        print(f"  corr N/S {b['corrN']:+.3f}/{b['corrS']:+.3f}   peak D {b['peakD']:.2f} G"
              f"   conc {b['conc']:.2f}   mean|dt| {b['mean_abs_dt']:.2f} yr")

    print("\n=== marginal best per axis (monotonic => not determined) ===")
    for ax_ in ("flow", "du", "eta_km", "tau"):
        g = d.groupby(ax_)[key].min().round(2)
        mono = ""
        if ax_ in ("du", "eta_km", "tau"):
            v = g.values
            if np.all(np.diff(v) > 0) or np.all(np.diff(v) < 0):
                mono = "   <- MONOTONIC: optimum outside the scanned range"
        print(f"  {ax_:7s} " + "  ".join(f"{k}={v}" for k, v in g.items()) + mono)


def main(argv=None):
    p = argparse.ArgumentParser(description="Analyse a finished SFT2D sweep")
    p.add_argument("sweep_dir")
    p.add_argument("--key", default="J_robust", choices=["J_robust", "J_rawamp"])
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    sd = Path(args.sweep_dir)
    if not (sd / "sweep_results.csv").is_file():
        sys.exit(f"no sweep_results.csv in {sd}")
    d = load(sd)
    tables(d, args.key)
    figure(d, sd, args.out or str(sd / "sweep_analysis.png"), args.key)


if __name__ == "__main__":
    main()
