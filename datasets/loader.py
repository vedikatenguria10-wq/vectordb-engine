"""Load AG News records from HuggingFace."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_LABEL_TO_CATEGORY: dict[int, str] = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}


def _hf_load_dataset(*args: Any, **kwargs: Any) -> Any:
    """Import HuggingFace ``datasets.load_dataset`` despite the local ``datasets`` package name."""
    project_root = str(Path(__file__).resolve().parent.parent)
    removed = False
    if project_root in sys.path:
        sys.path.remove(project_root)
        removed = True

    saved_modules = {
        key: sys.modules[key]
        for key in list(sys.modules)
        if key == "datasets" or key.startswith("datasets.")
    }
    for key in saved_modules:
        del sys.modules[key]

    try:
        from datasets import load_dataset

        return load_dataset(*args, **kwargs)
    finally:
        for key in list(sys.modules):
            if key == "datasets" or key.startswith("datasets."):
                del sys.modules[key]
        sys.modules.update(saved_modules)
        if removed:
            sys.path.insert(0, project_root)


def load_agnews(max_records: int = 2000) -> list[dict[str, Any]]:
    """Load the first ``max_records`` AG News train examples as id/text/category dicts."""
    ds = _hf_load_dataset("ag_news", split="train")
    n = min(max_records, len(ds))
    records: list[dict[str, Any]] = []

    for i in range(n):
        row = ds[i]
        label = int(row["label"])
        records.append(
            {
                "id": i,
                "text": str(row["text"]),
                "category": _LABEL_TO_CATEGORY[label],
            }
        )

    return records
