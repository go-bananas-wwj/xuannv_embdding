from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import xarray as xr


def _valid_ratio_for_slice(b: np.ndarray) -> float:
    """Return the fraction of valid (non-nodata) pixels in a band slice.

    For floating-point bands we mirror the heuristic in download_pc.py:
    finite and non-zero pixels are considered valid. Integer bands are
    assumed to use 0 as the nodata sentinel; this is a heuristic that may
    need adjustment if future sources adopt an explicit nodata value.
    """
    if np.issubdtype(b.dtype, np.floating):
        valid = np.isfinite(b) & (b != 0)
    else:
        valid = b != 0
    return float(valid.mean())


def check_file(path: Path) -> dict:
    print(f"checking {path.name}...")
    report: dict = {"file": str(path), "ok": True, "issues": []}
    try:
        ds = xr.open_dataset(path, chunks={"time": 1})
    except Exception as e:
        report["ok"] = False
        report["issues"].append(f"cannot open: {e}")
        return report

    try:
        if not ds.data_vars:
            report["ok"] = False
            report["issues"].append("dataset has no data variables")
            return report

        var = list(ds.data_vars)[0]
        arr = ds[var]
        report["dims"] = dict(arr.sizes)

        # Check per-time-step valid pixel ratio using the first band.
        if "band" not in arr.dims:
            report["ok"] = False
            report["issues"].append("missing 'band' dimension")
            return report

        band0 = arr.isel(band=0)
        valid_ratios = []
        for t in range(arr.sizes["time"]):
            b = band0.isel(time=t).values
            valid_ratios.append(_valid_ratio_for_slice(b))

        report["valid_ratio_min"] = float(np.min(valid_ratios))
        report["valid_ratio_median"] = float(np.median(valid_ratios))
        report["valid_ratio_max"] = float(np.max(valid_ratios))

        if report["valid_ratio_median"] < 0.05:
            report["ok"] = False
            report["issues"].append(
                f"median valid ratio too low: {report['valid_ratio_median']:.2%}"
            )

        if report["valid_ratio_max"] == 0:
            report["ok"] = False
            report["issues"].append("all time slices are invalid")
    finally:
        ds.close()

    return report


def main(
    region: str,
    data_root: str = "/data/xuannv_embedding/raw",
    output: str | None = None,
) -> None:
    data_path = Path(data_root)
    out = {"region": region, "sources": {}}
    for source in ["s2", "s1", "landsat"]:
        files = sorted(glob.glob(str(data_path / region / source / "*.nc")))
        if not files:
            out["sources"][source] = [
                {
                    "ok": False,
                    "issues": [f"no .nc files found in {data_path / region / source}"],
                }
            ]
            continue
        out["sources"][source] = [check_file(Path(f)) for f in files]

    out_path = Path(output) if output else Path(
        f"/data/xuannv_embedding/qa/qa_download_{region}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"QA report written to {out_path}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
