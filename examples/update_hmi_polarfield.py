"""
update_hmi_polarfield.py

Refresh the bundled HMI polar-field reference (`sft2d/data/hmi_polar_field.p`)
to the current date by downloading the mean polar-cap field from JSOC.

    python examples/update_hmi_polarfield.py
    python examples/update_hmi_polarfield.py --end 2027.01.01

Requires the `drms` package and internet access to JSOC
(jsoc.stanford.edu). Following the SunPy/drms polar-field example, it queries the
series `hmi.meanpf_720s` for the north/south polar-cap mean radial field
(`CAPN2`/`CAPS2`, poleward of ~60 deg) at 12-hour cadence, and stores the raw
series plus a 30-day centred rolling mean and standard deviation — the format
`sft2d.data.load_hmi_polar_field()` returns.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sft2d.data import HMI_POLAR_FIELD

START = "2010.05.01_TAI"
DEFAULT_END = "2026.07.27_TAI"      # override with --end YYYY.MM.DD


def main(end=DEFAULT_END):
    import drms

    qstr = f"hmi.meanpf_720s[{START}-{end}@12h]"
    print("querying", qstr, "...")
    r = drms.Client().query(qstr, key=["T_REC", "CAPN2", "CAPS2"])
    print(f"got {len(r)} records")

    r.index = drms.to_datetime(r.pop("T_REC"))
    r = r.sort_index()
    north = pd.to_numeric(r["CAPN2"], errors="coerce").rename("CAPN2")
    south = pd.to_numeric(r["CAPS2"], errors="coerce").rename("CAPS2")

    dt = (north.index[1] - north.index[0]).total_seconds()
    win = max(int(30 * 24 * 3600 / dt), 1)          # 30-day boxcar
    data = dict(
        north=north, south=south,
        mean_north=north.rolling(win, min_periods=1, center=True).mean(),
        mean_south=south.rolling(win, min_periods=1, center=True).mean(),
        std_north=north.rolling(win, min_periods=1, center=True).std(),
        std_south=south.rolling(win, min_periods=1, center=True).std(),
        time=np.array([d.to_pydatetime() for d in north.index], dtype=object),
    )
    with open(HMI_POLAR_FIELD, "wb") as fh:
        pickle.dump(data, fh)
    print(f"wrote {HMI_POLAR_FIELD} ({Path(HMI_POLAR_FIELD).stat().st_size/1e6:.2f} MB)")
    print(f"coverage: {north.index[0].date()} -> {north.index[-1].date()}, {len(north)} pts")


if __name__ == "__main__":
    end = DEFAULT_END
    if "--end" in sys.argv:
        end = sys.argv[sys.argv.index("--end") + 1].replace("-", ".") + "_TAI"
    main(end)
