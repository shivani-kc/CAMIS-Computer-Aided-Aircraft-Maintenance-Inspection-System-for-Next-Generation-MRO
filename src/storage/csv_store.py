from __future__ import annotations

import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

CSV_COLUMNS = [
    "inspection_id",
    "flight_id",
    "aircraft_id",
    "timestamp",
    "image_name",
    "preprocessing_applied",
    "defect_count",
    "critical_count",
    "highest_severity",
    "avg_confidence",
    "defect_types",
    "zones_affected",
    "dip_operations",
    "gemma_summary",
    "annotated_image_path",
    "fly_status",
    "estimated_cost_usd",
]

AIRCRAFT_POOL = [
    "AIRCRAFT-A320-001",
    "AIRCRAFT-A320-002",
    "AIRCRAFT-B737-001",
    "AIRCRAFT-B737-002",
    "AIRCRAFT-ATR72-001",
]


def _default_csv_path() -> str:
    return os.getenv("CSV_PATH", "/teamspace/studios/this_studio/CAMIS_AI/data/csv/everything_file.csv")


def _norm_list(v) -> str:
    if v is None:
        return "[]"
    if isinstance(v, list):
        return json.dumps(v, ensure_ascii=False)
    return json.dumps([str(v)], ensure_ascii=False)


def _derive_fly_status(highest_severity: str, urgent_required: bool) -> str:
    sev = (highest_severity or "").lower()
    if sev == "critical":
        return "do_not_fly"
    if sev == "high" or urgent_required:
        return "maintenance_required"
    if sev == "medium":
        return "inspection_pending"
    if sev == "low":
        return "fly_with_monitoring"
    return "clear"


def _now_timestamp() -> str:
    from datetime import timedelta
    import random
    base = datetime.now()
    dt = base - timedelta(
        days=random.randint(0, 7),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
    )
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _auto_ids():
    now = datetime.now()
    rand_tail = random.randint(100, 999)
    inspection_id = f"INSP-{now.strftime('%Y%m%d')}-{rand_tail}"
    flight_id = f"FLIGHT-{random.randint(1000, 9999)}"
    aircraft_id = random.choice(AIRCRAFT_POOL)
    return inspection_id, flight_id, aircraft_id


def build_csv_row(
    image_name: str,
    inference_result: Dict,
    dip_result: Optional[Dict] = None,
    gemma_result: Optional[Dict] = None,
    flight_id: Optional[str] = None,
    aircraft_id: Optional[str] = None,
    zones_affected: Optional[List[str]] = None,
) -> Dict:
    detections = inference_result.get("detections", []) or []
    defect_types = [d.get("class_name") or d.get("label") or "unknown" for d in detections]
    critical_count = sum(1 for d in detections if str(d.get("severity", "")).lower() == "critical")
    highest_severity = inference_result.get("highest_severity", "none")
    urgent_required = bool(inference_result.get("urgent_required", False))
    avg_confidence = float(inference_result.get("avg_confidence", 0.0) or 0.0)
    dip_operations = (dip_result or {}).get("applied_operations", []) or []
    preprocessing_applied = "yes" if dip_operations else "no"
    gemma_summary = str((gemma_result or {}).get("summary", "")).replace("\n", " ").strip()

    timestamp = inference_result.get("timestamp") or _now_timestamp()
    inspection_id, auto_flight_id, auto_aircraft_id = _auto_ids()

    return {
        "inspection_id": inspection_id,
        "flight_id": flight_id or auto_flight_id,
        "aircraft_id": aircraft_id or auto_aircraft_id,
        "timestamp": timestamp,
        "image_name": image_name,
        "preprocessing_applied": preprocessing_applied,
        "defect_count": int(inference_result.get("defect_count", len(detections))),
        "critical_count": int(critical_count),
        "highest_severity": highest_severity,
        "avg_confidence": avg_confidence,
        "defect_types": _norm_list(sorted(list(set(defect_types)))),
        "zones_affected": _norm_list(zones_affected or []),
        "dip_operations": _norm_list(dip_operations),
        "gemma_summary": gemma_summary,
        "annotated_image_path": inference_result.get("annotated_image_path", ""),
        "fly_status": _derive_fly_status(highest_severity, urgent_required),
        "estimated_cost_usd": float((gemma_result or {}).get("estimated_cost_usd", 0.0) or 0.0),
    }


def append_row_to_csv(row: Dict, csv_path: Optional[str] = None) -> str:
    csv_file = csv_path or _default_csv_path()
    Path(csv_file).parent.mkdir(parents=True, exist_ok=True)

    if Path(csv_file).exists():
        try:
            existing = pd.read_csv(csv_file)
        except Exception:
            existing = pd.DataFrame(columns=CSV_COLUMNS)
    else:
        existing = pd.DataFrame(columns=CSV_COLUMNS)

    for col in CSV_COLUMNS:
        if col not in existing.columns:
            existing[col] = None

    new_df = pd.DataFrame([row], columns=CSV_COLUMNS)
    out = pd.concat([existing[CSV_COLUMNS], new_df], ignore_index=True)
    out.to_csv(csv_file, index=False)
    return csv_file


def append_pipeline_results(
    image_name: str,
    inference_result: Dict,
    dip_result: Optional[Dict] = None,
    gemma_result: Optional[Dict] = None,
    csv_path: Optional[str] = None,
    flight_id: Optional[str] = None,
    aircraft_id: Optional[str] = None,
    zones_affected: Optional[List[str]] = None,
) -> Dict:
    row = build_csv_row(
        image_name=image_name,
        inference_result=inference_result,
        dip_result=dip_result,
        gemma_result=gemma_result,
        flight_id=flight_id,
        aircraft_id=aircraft_id,
        zones_affected=zones_affected,
    )
    path = append_row_to_csv(row, csv_path)
    return {"csv_path": path, "row": row}
