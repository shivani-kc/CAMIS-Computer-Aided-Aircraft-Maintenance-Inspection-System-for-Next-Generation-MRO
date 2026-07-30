from __future__ import annotations

from typing import Dict, List
import json
import requests

POWERBI_STREAMING_URL = ""


def _to_text(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def _to_number(v):
    try:
        return float(v)
    except Exception:
        return 0.0


def _normalize_row(row: Dict) -> Dict:
    return {
        "inspection_id": _to_text(row.get("inspection_id")),
        "flight_id": _to_text(row.get("flight_id")),
        "aircraft_id": _to_text(row.get("aircraft_id")),
        "timestamp": _to_text(row.get("timestamp")),
        "image_name": _to_text(row.get("image_name")),
        "preprocessing_applied": _to_text(row.get("preprocessing_applied")),
        "defect_count": int(_to_number(row.get("defect_count"))),
        "critical_count": int(_to_number(row.get("critical_count"))),
        "highest_severity": _to_text(row.get("highest_severity")),
        "avg_confidence": float(_to_number(row.get("avg_confidence"))),
        "defect_types": _to_text(row.get("defect_types")),
        "zones_affected": _to_text(row.get("zones_affected")),
        "dip_operations": _to_text(row.get("dip_operations")),
        "gemma_summary": _to_text(row.get("gemma_summary")),
        "annotated_image_path": _to_text(row.get("annotated_image_path")),
        "fly_status": _to_text(row.get("fly_status")),
        "estimated_cost_usd": float(_to_number(row.get("estimated_cost_usd", 0.0))),
    }


class PowerBIStreamingClient:
    def __init__(self, timeout: int = 20):
        self.push_url = POWERBI_STREAMING_URL
        self.timeout = timeout

    def post_rows(self, rows: List[Dict]) -> Dict:
        try:
            clean_rows = [_normalize_row(r) for r in rows]
            response = requests.post(
                self.push_url,
                json=clean_rows,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )
            return {
                "success": 200 <= response.status_code < 300,
                "status_code": response.status_code,
                "message": response.text[:1000],
                "payload_count": len(clean_rows),
                "sample_payload": clean_rows[0] if clean_rows else {},
            }
        except Exception as e:
            return {
                "success": False,
                "status_code": None,
                "message": str(e),
                "payload_count": len(rows),
            }

    def post_single_row(self, row: Dict) -> Dict:
        return self.post_rows([row])
