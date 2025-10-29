from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import csv, json
from typing import Any, Dict
from .paths import RESULTS

def write_json(obj: Dict[str, Any], name: str) -> str:
    p = RESULTS / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2))
    return str(p)

def write_csv(rows, headers, name: str) -> str:
    p = RESULTS / name
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', newline='') as f:
        w = csv.writer(f)
        if headers: w.writerow(headers)
        w.writerows(rows)
    return str(p)

def bundle_results(bundle: Dict[str, Any], prefix: str = "run") -> str:
    """Write a timestamped JSON bundle into results/."""
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return write_json(bundle, f"{prefix}_{stamp}.json")