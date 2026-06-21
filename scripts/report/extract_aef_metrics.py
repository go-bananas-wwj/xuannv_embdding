import json
import re
from pathlib import Path


def parse_metric(value_str: str):
    """Parse '0.7461±0.1637' or '+11.90pp' into a dict."""
    value_str = value_str.strip()
    if value_str.endswith("pp"):
        # percentage point delta
        num = float(value_str[:-2])
        return {"value": num, "unit": "pp"}
    m = re.match(r"([0-9.]+)(?:±([0-9.]+))?", value_str)
    if not m:
        return {"value": None, "std": None}
    val = float(m.group(1))
    std = float(m.group(2)) if m.group(2) else None
    return {"value": val, "std": std}


def main():
    src = Path("/data/xuannv_embedding/experiments/aef_benchmark/AEF_BENCHMARK_REPORT.md")
    out = Path("/root/workspace/report/data/aef_benchmark_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [l.strip() for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = []
    for line in lines:
        if not line.startswith("|"):
            continue
        if "任务" in line or "---" in line or line.count("|") < 5:
            continue
        parts = [p.strip() for p in line.split("|")][1:-1]
        if len(parts) < 5:
            continue
        task = parts[0]
        version = parts[1]
        if task == "" and version.startswith("Δ"):
            # Delta row belongs to previous task; skip or attach
            continue
        rows.append({
            "task": task,
            "version": version,
            "auc_roc": parse_metric(parts[2]),
            "f1_best": parse_metric(parts[3]),
            "f1_at_0_5": parse_metric(parts[4]),
            "miou": parse_metric(parts[5]) if len(parts) > 5 else {"value": None},
        })

    out.write_text(json.dumps({"tasks": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out} with {len(rows)} rows")


if __name__ == "__main__":
    main()
