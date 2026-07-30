from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from transformers import BitsAndBytesConfig
    BNB_AVAILABLE = True
except Exception:
    BitsAndBytesConfig = None
    BNB_AVAILABLE = False

load_dotenv()

SYSTEM_INSTRUCTION = """
You are CAMIS Damage Analysis Assistant for aircraft maintenance inspection.

STRICT RULES:
1. Use only the provided structured JSON for defect facts.
2. Write summary as ONE smooth human-readable paragraph. No bullet points.
3. Estimate repair cost in USD based on defect severity and type. Use realistic aviation MRO cost ranges:
   - critical defect: $8000 to $25000
   - high severity: $3000 to $8000
   - medium severity: $800 to $3000
   - low severity: $150 to $800
   - no defect: $0
4. Return valid JSON only.

Required JSON format:
{
  "summary": "...",
  "urgency": "immediate|scheduled|routine|none",
  "recommended_action": "...",
  "detected_defects": ["..."],
  "estimated_cost_usd": <number>
}
""".strip()


class GemmaClient:
    def __init__(self, model_id: Optional[str] = None, max_new_tokens: int = 180, temperature: float = 0.05):
        self.model_id = model_id or os.getenv("GEMMA_MODEL_ID", "google/gemma-4-E2B-it").strip()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.hf_token = os.getenv("HF_TOKEN", "").strip()

        if not self.hf_token:
            raise ValueError("HF_TOKEN is missing in .env")

        self.cuda_available = torch.cuda.is_available()
        self.device_mode = "gpu" if self.cuda_available else "cpu"

        print(f"[GemmaClient] torch.cuda.is_available() = {self.cuda_available}")
        if self.cuda_available:
            for i in range(torch.cuda.device_count()):
                print(f"[GemmaClient] GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"[GemmaClient] Selected device mode = {self.device_mode}")

        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=self.hf_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.cuda_available:
            if BNB_AVAILABLE:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    token=self.hf_token,
                    quantization_config=quant_config,
                    device_map="auto",
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    token=self.hf_token,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                token=self.hf_token,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
            self.model.to("cpu")

        self.model.eval()

    def _build_messages(self, detection_json: Dict) -> list:
        payload = json.dumps(detection_json, indent=2)
        return [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {
                "role": "user",
                "content": (
                    "Analyze this aircraft defect detection result. "
                    "Write one smooth paragraph summary. "
                    "Estimate repair cost based on severity. "
                    "Return JSON only.\n\n" + payload
                ),
            },
        ]

    def _fallback_cost(self, detection_json: Dict) -> float:
        sev = str(detection_json.get("highest_severity", "")).lower()
        mapping = {
            "critical": 15000.0,
            "high": 5000.0,
            "medium": 1500.0,
            "low": 400.0,
        }
        return mapping.get(sev, 0.0)

    def _safe_parse_json(self, text: str, detection_json: Dict) -> Dict:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]

        try:
            parsed = json.loads(text)
        except Exception:
            parsed = {}

        defaults = {
            "summary": "Inspection completed. Structural review recommended based on detected anomalies.",
            "urgency": "scheduled",
            "recommended_action": "Refer to maintenance team for detailed assessment.",
            "detected_defects": [],
            "estimated_cost_usd": self._fallback_cost(detection_json),
        }

        for key, default in defaults.items():
            if key not in parsed or parsed[key] is None:
                parsed[key] = default

        if not isinstance(parsed["detected_defects"], list):
            parsed["detected_defects"] = [str(parsed["detected_defects"])]

        parsed["summary"] = str(parsed["summary"]).replace("\n", " ").strip()
        parsed["recommended_action"] = str(parsed["recommended_action"]).replace("\n", " ").strip()

        try:
            parsed["estimated_cost_usd"] = float(parsed["estimated_cost_usd"])
        except Exception:
            parsed["estimated_cost_usd"] = self._fallback_cost(detection_json)

        parsed["device_mode"] = self.device_mode
        parsed["model_id"] = self.model_id
        return parsed

    def analyze_detection_json(self, detection_json: Dict) -> Dict:
        messages = self._build_messages(detection_json)

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = self.tokenizer(
            [prompt_text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        if self.cuda_available:
            model_inputs = {k: v.to(self.model.device) for k, v in model_inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=self.temperature,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        prompt_len = model_inputs["input_ids"].shape[1]
        generated_ids = outputs[0][prompt_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return self._safe_parse_json(text, detection_json)

    def analyze_json_file(self, json_path: str, output_summary_path: Optional[str] = None) -> Dict:
        with open(json_path, "r", encoding="utf-8") as f:
            detection_json = json.load(f)

        result = self.analyze_detection_json(detection_json)

        if output_summary_path:
            Path(output_summary_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_summary_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

        return result
