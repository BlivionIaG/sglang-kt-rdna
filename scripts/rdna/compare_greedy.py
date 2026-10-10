#!/usr/bin/env python3
"""Compare two greedy /generate dumps from smoke_qwen3_moe.sh.

Prints whether output_ids match and, if not, the first diverging index.
Exit 0 when the token ids match. Exit 1 when they differ or a file is missing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(path: Path) -> dict:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        data = data[0]
    return data


def ids_of(payload: dict) -> list[int]:
    if "output_ids" in payload:
        return list(payload["output_ids"])
    meta = payload.get("meta_info") or {}
    if "output_ids" in meta:
        return list(meta["output_ids"])
    raise SystemExit(f"no output_ids in payload keys {sorted(payload)}")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} gfx1030.json gfx1100.json")
    left_path, right_path = Path(sys.argv[1]), Path(sys.argv[2])
    left, right = ids_of(load(left_path)), ids_of(load(right_path))
    limit = min(len(left), len(right))
    diverge = next((i for i in range(limit) if left[i] != right[i]), None)
    if diverge is None and len(left) == len(right):
        print(f"match {len(left)} tokens")
        return
    if diverge is None:
        diverge = limit
    print(
        f"diverge at index {diverge} "
        f"len {left_path.name}={len(left)} {right_path.name}={len(right)}"
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
