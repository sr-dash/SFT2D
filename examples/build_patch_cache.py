"""
build_patch_cache.py

Precompute every SHARP active-region patch onto a fixed model grid and save one
compact sparse cache, so a parameter sweep does not repeat that work per member.

    python examples/build_patch_cache.py --db /path/to/sharps-bmrs-db \
        --grid 181 360 --out patch_cache_181x360.npz

Why
---
Every member of a sweep inserts the *same* regions on the *same* grid, so each
one re-opens ~3000 ``sharpNNNNN.nc`` files and repeats identical interpolation --
about 3 GB of reads per member.  With tens of members running concurrently off a
shared cluster filesystem that redundant I/O can dominate the wall time (a 686
member sweep re-reads over 2 TB).

The cache is tiny (a few MB: patches are localised, and ``balance_flux`` rescales
rather than offsets, so they stay sparse) and reproduces the uncached driver to
machine precision.  Build it once, put it somewhere fast (node-local scratch or
``/dev/shm`` is ideal), and point the sweep at it with ``--cache``.

The cache is **grid-specific**: build one per (n_theta, n_phi) you intend to run.
``CachedPatchSource`` raises if a cache is used with a grid it was not built for.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(description="Precompute the SHARP patch cache")
    p.add_argument("--db", default=os.environ.get("SFT2D_SHARPS_DB"),
                   help="SHARPS database dir (catalogue + sharpNNNNN.nc maps)")
    p.add_argument("--grid", type=int, nargs=2, default=[181, 360],
                   metavar=("NTHETA", "NPHI"))
    p.add_argument("--out", default=None,
                   help="output .npz (default patch_cache_<ntheta>x<nphi>.npz)")
    p.add_argument("--start", default="2010-05-01")
    p.add_argument("--end", default="2026-06-30")
    p.add_argument("--var", default="br", choices=["br", "br_bipole"])
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    p.add_argument("--verify", action="store_true",
                   help="re-insert every region through both drivers and report "
                        "the maximum difference (a few minutes)")
    args = p.parse_args(argv)

    if not args.db:
        sys.exit("no SHARPS database given: pass --db or set SFT2D_SHARPS_DB")
    db = Path(args.db).expanduser()
    cat = db / "bmrsharps_evol.txt"
    if not cat.is_file():
        sys.exit(f"catalogue not found: {cat}")

    import numpy as np

    from sft2d.src.grid import create_grid
    from sft2d.src.sharp_patch_driver import build_patch_cache

    nt, npz_ = args.grid
    out = args.out or f"patch_cache_{nt}x{npz_}.npz"
    grid = create_grid(nt, npz_)

    print(f"building patch cache for a {nt}x{npz_} grid from {db}")
    t0 = time.time()
    build_patch_cache(str(cat), str(db), grid, out, start_date=args.start,
                      end_date=args.end, var=args.var, n_jobs=args.jobs)
    print(f"built in {time.time()-t0:.0f}s")

    if args.verify:
        from sft2d.src.sharp_patch_driver import CachedPatchSource, SHARPPatchSource
        raw = SHARPPatchSource(str(cat), nc_dir=str(db), start_date=args.start,
                               end_date=args.end, var=args.var)
        cch = CachedPatchSource(out, str(cat), start_date=args.start, end_date=args.end)
        a = np.zeros((grid["n_theta"], grid["n_phi"])); b = a.copy()
        for d in sorted(raw.events):
            a = raw(d, a, grid)
        for d in sorted(cch.events):
            b = cch(d, b, grid)
        err = float(np.abs(a - b).max()); rel = err / float(np.abs(a).max())
        print(f"verify: max |cached - uncached| = {err:.3e} G (relative {rel:.2e}) "
              f"-> {'IDENTICAL' if rel < 1e-12 else 'MISMATCH'}")
        if rel >= 1e-12:
            sys.exit("cache does NOT reproduce the uncached driver")


if __name__ == "__main__":
    main()
