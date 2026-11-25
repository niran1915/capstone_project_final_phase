"""Published external validation status used by mapping and VQA reports."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SUMMARY_PATH = Path(__file__).resolve().parents[2] / "validation_reports" / "class_validation_summary.json"
CLASS_ORDER = ("crack", "fissures", "spalling", "rebar")


def class_validation_summary() -> dict[str, Any]:
    """Load benchmark-level validation; never infer a score from image telemetry."""
    try:
        with SUMMARY_PATH.open("r", encoding="utf-8") as stream:
            summary = json.load(stream)
        classes = summary.get("classes")
        if not isinstance(classes, dict) or any(name not in classes for name in CLASS_ORDER):
            raise ValueError("Validation summary is missing a supported label.")
        summary["classes"] = {name: classes[name] for name in CLASS_ORDER}
        return summary
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        return {
            "dataset": None,
            "dataset_url": None,
            "paper_url": None,
            "dataset_scope": "No external class validation summary is installed.",
            "interpretation": "Validation is unavailable; do not interpret detector evidence as validated mapping performance.",
            "classes": {
                name: {
                    "status": "validation_summary_unavailable",
                    "summary": "No external class-matched validation result is available.",
                    "dataset": None,
                    "limitations": "Do not infer class accuracy from this inspection.",
                }
                for name in CLASS_ORDER
            },
        }
