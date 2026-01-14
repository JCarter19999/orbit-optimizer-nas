from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Iterable

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
