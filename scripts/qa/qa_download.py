import json
import glob
from pathlib import Path
import numpy as np
import xarray as xr


def _valid_ratio_for_slice(b: np.ndarray) -> float:
    if np.issubdtype(b.dtype, np.floating):
        valid = np.isfinite(b)
    else:
        valid = b != 0
    return float(valid.mean())


def check_file(path: Path) -> dict:
    report = {"file": str(path), "ok": True, "issues": []}
    try:
        ds = xr.open_dataset(path, chunks={"time": 1})
    except Exception as e:
        report["ok"] = False
        report["issues"].append(f"cannot open: {e}")
        return report

    try:
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


def main(region: str):
    out = {"region": region, "sources": {}}
    for source in ["s2", "s1", "landsat"]:
        files = sorted(
            glob.glob(str(Path("/data/xuannv_embedding/raw") / region / source / "*.nc"))
        )
        out["sources"][source] = [check_file(Path(f)) for f in files]

    out_path = Path(f"/data/xuannv_embedding/qa/qa_download_{region}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"QA report written to {out_path}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
