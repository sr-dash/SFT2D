"""
plot_ensemble_envelope.py

One panel showing the whole sweep against the observations: the HMI polar-cap
field (raw 12-hourly, its 30-day rolling mean, and the +/-1 sigma band), every
ensemble member's polar-field history as a slim translucent line coloured by a
chosen parameter, and the best-fitting member(s) in bold.

    python examples/plot_ensemble_envelope.py "/Volumes/Extreme SSD/SFT2D/sft2d_sweep_out"
    python examples/plot_ensemble_envelope.py <dir> --color-by tau
    python examples/plot_ensemble_envelope.py <dir> --hemisphere north

Reading the figure
------------------
The ensemble lines are the *reachable range* of the model over the scanned
parameter space, so the plot answers two questions at once: does the observed
curve lie inside what the model can produce, and which part of the parameter
space gets closest.  Colouring by ``du`` (default) is the informative choice --
it is the parameter the sweep failed to determine, and it orders the spread.

Colour decisions: the ensemble carries a *magnitude* (a parameter value), so it
gets a single-hue sequential ramp, light to dark, never a rainbow.  The
observations are ink, not a series colour.  The bold best-fit members share one
highlight hue and are separated by line style, since they are the same kind of
thing (a chosen model run) rather than different entities.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

C_BEST = "#d95f02"          # highlight: model best fits
INK, MUTED = "#1a1a1a", "#6b6b6b"
PHYSICAL_DU_MIN = 10.0
LABELS = {"du": "du [m/s]", "eta_km": r"eta [km$^2$/s]", "tau": "tau [yr]",
          "flow": "flow profile"}


def main(argv=None):
    p = argparse.ArgumentParser(description="Ensemble vs HMI polar field, one panel")
    p.add_argument("sweep_dir")
    p.add_argument("--color-by", default="du", choices=["du", "eta", "tau", "profile"])
    p.add_argument("--hemisphere", default="both", choices=["both", "north", "south"])
    p.add_argument("--key", default="J_robust", choices=["J_robust", "J_rawamp"])
    p.add_argument("--alpha", type=float, default=0.13)
    p.add_argument("--clip", type=float, default=99.0,
                   help="percentile of ensemble values used to set the y-range; "
                        "a few extreme members otherwise squash the panel")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import LineCollection
    from matplotlib.colors import BoundaryNorm, Normalize
    from matplotlib.lines import Line2D

    from sft2d.data import load_hmi_polar_field

    sd = Path(args.sweep_dir)
    csv = sd / "sweep_results.csv"
    if not csv.is_file():
        sys.exit(f"no sweep_results.csv in {sd}")
    d = pd.read_csv(csv)
    if "failed" in d:
        d = d[~d["failed"].astype(bool)]
    d["eta_km"] = (d["eta"] / 1e6).round(0)
    d["flow"] = np.where(d["profile"].str.startswith("yeates"), "y20", "SB")

    cby = {"eta": "eta_km", "profile": "flow"}.get(args.color_by, args.color_by)
    vals = sorted(d[cby].unique())
    # Sequential single hue for a numeric parameter; two well-separated steps of
    # the same ramp when the "parameter" is really a pair of categories.
    ramp = plt.get_cmap("Blues")
    steps = (np.linspace(0.25, 1.0, len(vals)) if len(vals) > 2
             else np.array([0.45, 0.9]))
    cmap_colors = {v: ramp(s) for v, s in zip(vals, steps)}

    # ---- observations -------------------------------------------------------
    h = load_hmi_polar_field()
    idx = h["mean_north"].index
    ht = np.asarray(idx.year + (idx.dayofyear - 1) / 365.25, float)

    hemis = (["north", "south"] if args.hemisphere == "both" else [args.hemisphere])
    fig, ax = plt.subplots(figsize=(14, 7.5))

    for hemi in hemis:
        raw = np.asarray(h[hemi].values, float)
        mean = np.asarray(h[f"mean_{hemi}"].values, float)
        std = np.asarray(h[f"std_{hemi}"].values, float)
        ok = np.isfinite(mean) & np.isfinite(std)
        ax.fill_between(ht[ok], (mean - std)[ok], (mean + std)[ok],
                        color=INK, alpha=0.13, lw=0, zorder=3)
        ax.plot(ht, raw, color=MUTED, lw=0.5, alpha=0.55, zorder=3.1)
        ax.plot(ht, mean, color=INK, lw=2.4, zorder=3.2)

    # ---- ensemble -----------------------------------------------------------
    best = d.loc[d[args.key].idxmin()]
    phys = d[d["du"] >= PHYSICAL_DU_MIN]
    bphys = phys.loc[phys[args.key].idxmin()] if len(phys) else None
    bold_tags = {best["tag"]} | ({bphys["tag"]} if bphys is not None else set())

    segs, cols, n_missing = [], [], 0
    for _, r in d.iterrows():
        f = sd / str(r["file"])
        if not f.is_file():
            n_missing += 1
            continue
        with np.load(f) as z:                      # reads only these arrays
            yr = z["yr"]
            for hemi, mk in (("north", "pn"), ("south", "ps")):
                if hemi not in hemis:
                    continue
                segs.append(np.column_stack([yr, z[mk]]))
                cols.append(cmap_colors[r[cby]])
    if segs:
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=0.45,
                                         alpha=args.alpha, zorder=2))
    if n_missing:
        print(f"warning: {n_missing} member files missing; plotted the rest")

    # ---- bold best fits -----------------------------------------------------
    def overlay(row, ls, label):
        f = sd / str(row["file"])
        if not f.is_file():
            return
        with np.load(f) as z:
            for hemi, mk in (("north", "pn"), ("south", "ps")):
                if hemi in hemis:
                    ax.plot(z["yr"], z[mk], color=C_BEST, lw=2.4, ls=ls, zorder=4)
        return label

    lab_best = (f"best {args.key}: {best['flow']}, du={best['du']:g}, "
                f"eta={best['eta_km']:.0f}, tau={best['tau']:g}")
    overlay(best, "-", lab_best)
    lab_phys = None
    if bphys is not None and bphys["tag"] != best["tag"]:
        lab_phys = (f"best with du>={PHYSICAL_DU_MIN:g}: {bphys['flow']}, "
                    f"du={bphys['du']:g}, eta={bphys['eta_km']:.0f}, tau={bphys['tau']:g}")
        overlay(bphys, "--", lab_phys)

    # ---- frame, legend, colourbar ------------------------------------------
    ax.axhline(0, color=MUTED, lw=0.8, zorder=1.5)
    ax.set_xlim(ht.min() - 0.15, ht.max() + 0.15)
    if segs:
        allv = np.concatenate([s_[:, 1] for s_ in segs])
        lim = np.nanpercentile(np.abs(allv), args.clip)
        hmax = max(np.nanmax(np.abs(np.asarray(h[f'mean_{k}'].values, float)[
            np.isfinite(np.asarray(h[f'mean_{k}'].values, float))])) for k in hemis)
        lim = max(lim, 1.25 * hmax) * 1.08
        ax.set_ylim(-lim, lim)
        n_out = int(np.sum(np.abs(allv) > lim))
        if n_out:
            ax.text(0.995, 0.015, f"y-axis clipped at the {args.clip:g}th percentile "
                    f"of ensemble values ({n_out} of {allv.size} samples outside)",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7.5, color=MUTED)
    ax.set_xlabel("year"); ax.set_ylabel("polar cap field, poleward of 60$^\\circ$ [G]")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(alpha=0.22, lw=0.6, color=MUTED); ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK); ax.yaxis.label.set_color(INK)

    hemi_txt = ("north (upper) and south (lower)" if len(hemis) == 2 else hemis[0])
    ax.set_title(f"SFT2D sweep vs HMI: {len(d)} members, {hemi_txt} polar cap field\n"
                 f"ensemble coloured by {LABELS.get(cby, cby)}",
                 color=INK, fontsize=13)

    handles = [
        Line2D([], [], color=INK, lw=2.4, label="HMI 30-day mean"),
        Line2D([], [], color=MUTED, lw=0.8, alpha=0.7, label="HMI raw (12-hourly)"),
        matplotlib.patches.Patch(color=INK, alpha=0.13, label="HMI $\\pm1\\sigma$"),
        Line2D([], [], color=ramp(0.7), lw=1.6,
               label=f"{len(segs)} ensemble members"),
        Line2D([], [], color=C_BEST, lw=2.4, label=lab_best),
    ]
    if lab_phys:
        handles.append(Line2D([], [], color=C_BEST, lw=2.4, ls="--", label=lab_phys))
    ax.legend(handles=handles, fontsize=9, frameon=False, loc="lower left", ncol=2)

    if len(vals) > 1:
        num = not isinstance(vals[0], str)
        if num:
            edges = np.concatenate([[vals[0] - (vals[1] - vals[0]) / 2],
                                    (np.asarray(vals[:-1]) + np.asarray(vals[1:])) / 2,
                                    [vals[-1] + (vals[-1] - vals[-2]) / 2]])
            sm = ScalarMappable(norm=BoundaryNorm(edges, len(vals)),
                                cmap=matplotlib.colors.ListedColormap(
                                    [cmap_colors[v] for v in vals]))
        else:
            sm = ScalarMappable(norm=Normalize(0, len(vals)),
                                cmap=matplotlib.colors.ListedColormap(
                                    [cmap_colors[v] for v in vals]))
        cb = fig.colorbar(sm, ax=ax, pad=0.015, shrink=0.72,
                          ticks=vals if num else np.arange(len(vals)) + 0.5)
        if not num:
            cb.ax.set_yticklabels(vals)
        cb.set_label(LABELS.get(cby, cby), color=INK, fontsize=10)
        cb.ax.tick_params(colors=MUTED, labelsize=9)

    out = args.out or str(sd / f"ensemble_envelope_by_{cby}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")
    print(f"  ensemble lines: {len(segs)}   coloured by {cby} "
          f"({len(vals)} values: {', '.join(str(v) for v in vals)})")
    print(f"  bold: {lab_best}" + (f"\n        {lab_phys}" if lab_phys else ""))


if __name__ == "__main__":
    main()
