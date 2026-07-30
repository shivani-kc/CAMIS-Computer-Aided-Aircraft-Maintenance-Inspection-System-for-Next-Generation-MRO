from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
from ultralytics import YOLO


CLASS_NAME_MAP = {
    0: "Dent",
    1: "Fastener Damage",
    2: "Rupture",
}


SEVERITY_RULES = {
    "Rupture": "critical",
    "Fastener Damage": "medium",
    "Dent": "low",
}


@dataclass
class DetectionItem:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: List[int]
    severity: str
    urgent: bool


@dataclass
class InferenceResult:
    input_path: str
    annotated_image_path: str
    json_path: str
    timestamp: str
    defect_count: int
    avg_confidence: float
    highest_severity: str
    urgent_required: bool
    detections: List[DetectionItem]


class YOLOInference:
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model = YOLO(model_path)

    def _class_name(self, class_id: int) -> str:
        if hasattr(self.model, "names") and class_id in self.model.names:
            return str(self.model.names[class_id])
        return CLASS_NAME_MAP.get(class_id, f"class_{class_id}")

    def _severity_from_class(self, class_name: str, confidence: float) -> str:
        base = SEVERITY_RULES.get(class_name, "medium")
        if class_name == "Dent" and confidence >= 0.85:
            return "medium"
        return base

    def _highest_severity(self, detections: List[DetectionItem]) -> str:
        order = {"critical": 4, "high": 3, "medium": 2, "low": 1, "none": 0}
        highest = "none"
        for d in detections:
            if order[d.severity] > order[highest]:
                highest = d.severity
        return highest

    def predict(
        self,
        input_path: str,
        annotated_output_path: str,
        json_output_path: str,
    ) -> Dict:
        results = self.model.predict(
            source=input_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        if not results:
            raise ValueError("No inference results returned by YOLO model.")

        result = results[0]
        detections: List[DetectionItem] = []

        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = [int(v) for v in box.xyxy[0].tolist()]
                class_name = self._class_name(cls_id)
                severity = self._severity_from_class(class_name, conf)
                urgent = severity in {"critical", "high"}

                detections.append(
                    DetectionItem(
                        class_id=cls_id,
                        class_name=class_name,
                        confidence=round(conf, 4),
                        bbox_xyxy=xyxy,
                        severity=severity,
                        urgent=urgent,
                    )
                )

        plotted = result.plot()
        Path(annotated_output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(annotated_output_path, plotted)

        highest = self._highest_severity(detections)
        urgent_required = any(d.urgent for d in detections)
        avg_conf = round(sum(d.confidence for d in detections) / len(detections), 4) if detections else 0.0

        output = InferenceResult(
            input_path=input_path,
            annotated_image_path=annotated_output_path,
            json_path=json_output_path,
            timestamp=datetime.now().isoformat(),
            defect_count=len(detections),
            avg_confidence=avg_conf,
            highest_severity=highest,
            urgent_required=urgent_required,
            detections=detections,
        )

        payload = asdict(output)
        payload["detections"] = [asdict(d) for d in detections]

        Path(json_output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return payload


def run_inference(
    model_path: str,
    input_path: str,
    annotated_output_path: str,
    json_output_path: str,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    device: Optional[str] = None,
) -> Dict:
    engine = YOLOInference(
        model_path=model_path,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        device=device,
    )
    return engine.predict(
        input_path=input_path,
        annotated_output_path=annotated_output_path,
        json_output_path=json_output_path,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 5:
        raise SystemExit(
            "Usage: python -m src.inference.yolo_infer <model_path> <input_path> <annotated_output_path> <json_output_path>"
        )

    result = run_inference(
        model_path=sys.argv[1],
        input_path=sys.argv[2],
        annotated_output_path=sys.argv[3],
        json_output_path=sys.argv[4],
    )
    print(json.dumps(result, indent=2))
