from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


@dataclass
class DIPResult:
    input_path: str
    output_path: str
    blur_score: float
    brightness_score: float
    contrast_score: float
    applied_operations: List[str]
    quality_flags: Dict[str, bool]


class DIPPipeline:
    def __init__(
        self,
        blur_threshold: float = 100.0,
        dark_threshold: float = 90.0,
        bright_threshold: float = 190.0,
        contrast_threshold: float = 35.0,
    ):
        self.blur_threshold = blur_threshold
        self.dark_threshold = dark_threshold
        self.bright_threshold = bright_threshold
        self.contrast_threshold = contrast_threshold

    def load_image(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return image

    def save_image(self, image: np.ndarray, output_path: str) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, image)

    def laplacian_variance(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def brightness_score(self, image: np.ndarray) -> float:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        return float(np.mean(v))

    def contrast_score(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(np.std(gray))

    def needs_processing(self, image: np.ndarray) -> Tuple[Dict[str, bool], Dict[str, float]]:
        blur = self.laplacian_variance(image)
        brightness = self.brightness_score(image)
        contrast = self.contrast_score(image)

        flags = {
            "is_blurry": blur < self.blur_threshold,
            "is_dark": brightness < self.dark_threshold,
            "is_too_bright": brightness > self.bright_threshold,
            "is_low_contrast": contrast < self.contrast_threshold,
        }

        scores = {
            "blur_score": blur,
            "brightness_score": brightness,
            "contrast_score": contrast,
        }
        return flags, scores

    def white_balance_gray_world(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32)
        b_avg, g_avg, r_avg = np.mean(img[:, :, 0]), np.mean(img[:, :, 1]), np.mean(img[:, :, 2])
        gray_avg = (b_avg + g_avg + r_avg) / 3.0

        scale_b = gray_avg / (b_avg + 1e-6)
        scale_g = gray_avg / (g_avg + 1e-6)
        scale_r = gray_avg / (r_avg + 1e-6)

        img[:, :, 0] *= scale_b
        img[:, :, 1] *= scale_g
        img[:, :, 2] *= scale_r

        return np.clip(img, 0, 255).astype(np.uint8)

    def gamma_correction(self, image: np.ndarray, gamma: float) -> np.ndarray:
        gamma = max(gamma, 0.1)
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
        return cv2.LUT(image, table)

    def apply_clahe(self, image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l = clahe.apply(l)
        merged = cv2.merge((l, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def bilateral_filter(self, image: np.ndarray, d: int = 9, sigma_color: int = 75, sigma_space: int = 75) -> np.ndarray:
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    def unsharp_mask(
        self,
        image: np.ndarray,
        sigma: float = 1.0,
        strength: float = 1.5,
    ) -> np.ndarray:
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def process(self, input_path: str, output_path: str) -> DIPResult:
        image = self.load_image(input_path)
        flags, scores = self.needs_processing(image)

        processed = image.copy()
        operations = []

        if any(flags.values()):
            processed = self.white_balance_gray_world(processed)
            operations.append("white_balance")

            if flags["is_dark"]:
                processed = self.gamma_correction(processed, gamma=1.4)
                operations.append("gamma_brighten")

            elif flags["is_too_bright"]:
                processed = self.gamma_correction(processed, gamma=0.8)
                operations.append("gamma_darken")

            if flags["is_low_contrast"]:
                processed = self.apply_clahe(processed, clip_limit=2.0, tile_grid_size=(8, 8))
                operations.append("clahe")

            if flags["is_blurry"]:
                processed = self.bilateral_filter(processed, d=9, sigma_color=75, sigma_space=75)
                operations.append("bilateral_filter")
                processed = self.unsharp_mask(processed, sigma=1.0, strength=1.2)
                operations.append("unsharp_mask")

        self.save_image(processed, output_path)

        return DIPResult(
            input_path=input_path,
            output_path=output_path,
            blur_score=scores["blur_score"],
            brightness_score=scores["brightness_score"],
            contrast_score=scores["contrast_score"],
            applied_operations=operations,
            quality_flags=flags,
        )


def process_single_image(input_path: str, output_path: str) -> Dict:
    pipeline = DIPPipeline()
    result = pipeline.process(input_path, output_path)
    return asdict(result)


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) != 3:
        raise SystemExit("Usage: python -m src.dip.pipeline <input_path> <output_path>")

    result = process_single_image(sys.argv[1], sys.argv[2])
    print(json.dumps(result, indent=2))
