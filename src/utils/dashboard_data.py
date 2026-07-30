from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import plotly.express as px
from dotenv import load_dotenv

load_dotenv()
PLOT_THEME = "plotly_dark"


def get_csv_path() -> str:
    return os.getenv(
        "CSV_PATH",
        "/teamspace/studios/this_studio/CAMIS_AI/data/csv/everything_file.csv",
    )


def _parse_listish(cell) -> List[str]:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    if isinstance(cell, list):
        return [str(x) for x in cell if str(x).strip()]

    s = str(cell).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return []

    for parser in (json.loads, ast.literal_eval):
        try:
            val = parser(s)
            if isinstance(val, list):
                return [str(x) for x in val if str(x).strip()]
        except Exception:
            pass

    if "," in s:
        return [x.strip() for x in s.split(",") if x.strip()]

    return [s]


def load_history() -> pd.DataFrame:
    csv_path = get_csv_path()
    if not Path(csv_path).exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return pd.DataFrame()

    defaults = {
        "inspection_id": "",
        "flight_id": "",
        "aircraft_id": "AIRCRAFT-DEMO-001",
        "timestamp": "",
        "image_name": "",
        "preprocessing_applied": "no",
        "defect_count": 0,
        "critical_count": 0,
        "highest_severity": "none",
        "avg_confidence": 0.0,
        "defect_types": "[]",
        "zones_affected": "[]",
        "dip_operations": "[]",
        "gemma_summary": "",
        "annotated_image_path": "",
        "fly_status": "clear",
    }

    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    df["timestamp"] = df["timestamp"].replace(["None", "none", "nan", "NaT", ""], pd.NA)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    for col in ["defect_count", "critical_count", "avg_confidence"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if df["avg_confidence"].max() <= 1.0:
        df["avg_confidence_pct"] = df["avg_confidence"] * 100.0
    else:
        df["avg_confidence_pct"] = df["avg_confidence"]

    df["defect_types"] = df["defect_types"].apply(_parse_listish)
    df["zones_affected"] = df["zones_affected"].apply(_parse_listish)
    df["dip_operations"] = df["dip_operations"].apply(_parse_listish)

    df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d").fillna("Unknown")
    df["month"] = df["timestamp"].dt.to_period("M").astype(str).replace("NaT", "Unknown")

    return df


def filter_history(df: pd.DataFrame, aircraft_ids=None, severities=None, defect_types=None) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()

    if aircraft_ids:
        out = out[out["aircraft_id"].astype(str).isin(aircraft_ids)]

    if severities:
        out = out[out["highest_severity"].astype(str).isin(severities)]

    if defect_types:
        out = out[out["defect_types"].apply(lambda xs: any(x in xs for x in defect_types))]

    return out


def available_filter_values(df: pd.DataFrame) -> Tuple[list, list, list]:
    if df.empty:
        return [], [], []

    aircraft = sorted(df["aircraft_id"].dropna().astype(str).unique().tolist()) if "aircraft_id" in df.columns else []
    severity = sorted(df["highest_severity"].dropna().astype(str).unique().tolist()) if "highest_severity" in df.columns else []

    defects = set()
    if "defect_types" in df.columns:
        for items in df["defect_types"]:
            for x in items:
                defects.add(str(x))

    return aircraft, severity, sorted(defects)


def make_kpis(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "total_inspections": 0,
            "total_defects": 0,
            "critical_defects": 0,
            "avg_confidence": "0.0%",
            "no_fly_aircraft": 0,
        }

    no_fly = df[df["fly_status"] == "do_not_fly"]["aircraft_id"].nunique()

    return {
        "total_inspections": int(len(df)),
        "total_defects": int(df["defect_count"].sum()),
        "critical_defects": int(df["critical_count"].sum()),
        "avg_confidence": f"{df['avg_confidence_pct'].mean():.1f}%",
        "no_fly_aircraft": int(no_fly),
    }


def defect_trend_chart(df: pd.DataFrame):
    temp = pd.DataFrame({"date": ["No Data"], "defect_count": [0]}) if df.empty else df.groupby("date", as_index=False)["defect_count"].sum()
    fig = px.line(temp, x="date", y="defect_count", markers=True, title="Defect Trend Over Time")
    fig.update_traces(line=dict(width=3, color="#00c2ff"))
    fig.update_layout(template=PLOT_THEME, height=360, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def severity_stacked_chart(df: pd.DataFrame):
    temp = pd.DataFrame({"month": ["No Data"], "highest_severity": ["none"], "count": [0]}) if df.empty else df.groupby(["month", "highest_severity"]).size().reset_index(name="count")
    fig = px.bar(
        temp,
        x="month",
        y="count",
        color="highest_severity",
        barmode="stack",
        title="Severity by Month",
        color_discrete_map={
            "critical": "#ff4b6e",
            "high": "#ff9f43",
            "medium": "#f6c343",
            "low": "#2ecc71",
            "none": "#6c757d",
        },
    )
    fig.update_layout(template=PLOT_THEME, height=360, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def defect_type_distribution_chart(df: pd.DataFrame):
    vals = []
    if not df.empty:
        for items in df["defect_types"]:
            vals.extend(items)

    if not vals:
        vals = ["No Data"]

    temp = pd.DataFrame({"defect_type": vals}).value_counts("defect_type").reset_index(name="count").sort_values("count", ascending=False)
    fig = px.bar(
        temp,
        x="count",
        y="defect_type",
        orientation="h",
        title="Damage Type Distribution",
        text_auto=True,
        color="count",
        color_continuous_scale="Turbo",
    )
    fig.update_layout(
        template=PLOT_THEME,
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis={"categoryorder": "total ascending"},
        coloraxis_showscale=False,
    )
    return fig


def aircraft_risk_chart(df: pd.DataFrame):
    if df.empty:
        temp = pd.DataFrame({"aircraft_id": ["No Data"], "risk_score": [0]})
    else:
        temp = df.groupby("aircraft_id", as_index=False).agg(
            total_defects=("defect_count", "sum"),
            critical_defects=("critical_count", "sum"),
        )
        temp["risk_score"] = temp["total_defects"] + temp["critical_defects"] * 3
        temp = temp.sort_values("risk_score", ascending=False).head(10)

    fig = px.bar(
        temp,
        x="aircraft_id",
        y="risk_score",
        title="Aircraft Risk Ranking",
        text_auto=True,
        color="risk_score",
        color_continuous_scale="Reds",
    )
    fig.update_layout(template=PLOT_THEME, height=360, margin=dict(l=20, r=20, t=50, b=20), coloraxis_showscale=False)
    return fig


def preprocessing_chart(df: pd.DataFrame):
    temp = pd.DataFrame({"preprocessing_applied": ["no_data"], "count": [1]}) if df.empty else df.groupby("preprocessing_applied", as_index=False).size().rename(columns={"size": "count"})
    fig = px.pie(
        temp,
        names="preprocessing_applied",
        values="count",
        title="Preprocessing Usage",
        hole=0.58,
        color="preprocessing_applied",
        color_discrete_map={"yes": "#00c2ff", "no": "#8b949e", "no_data": "#444"},
    )
    fig.update_layout(template=PLOT_THEME, height=360, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def fly_status_chart(df: pd.DataFrame):
    temp = pd.DataFrame({"fly_status": ["no_data"], "count": [1]}) if df.empty else df.groupby("fly_status", as_index=False).size().rename(columns={"size": "count"})
    fig = px.bar(
        temp,
        x="fly_status",
        y="count",
        title="Fleet Readiness Status",
        text_auto=True,
        color="fly_status",
        color_discrete_map={
            "do_not_fly": "#ff4b6e",
            "maintenance_required": "#ff9f43",
            "inspection_pending": "#f6c343",
            "fly_with_monitoring": "#2ecc71",
            "clear": "#00c2ff",
            "no_data": "#666666",
        },
    )
    fig.update_layout(template=PLOT_THEME, height=360, margin=dict(l=20, r=20, t=50, b=20), showlegend=False)
    return fig


def confidence_trend_chart(df: pd.DataFrame):
    temp = pd.DataFrame({"date": ["No Data"], "avg_confidence_pct": [0]}) if df.empty else df.groupby("date", as_index=False)["avg_confidence_pct"].mean()
    fig = px.area(temp, x="date", y="avg_confidence_pct", title="Model Confidence Trend")
    fig.update_traces(line=dict(color="#29b6f6", width=3))
    fig.update_layout(template=PLOT_THEME, height=360, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def recent_inspection_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        c for c in [
            "inspection_id",
            "flight_id",
            "aircraft_id",
            "timestamp",
            "defect_count",
            "critical_count",
            "highest_severity",
            "avg_confidence_pct",
            "fly_status",
            "estimated_cost_usd",
            "gemma_summary",
        ] if c in df.columns
    ]

    if not cols:
        return pd.DataFrame()

    out = df[cols].copy()

    if "timestamp" in out.columns:
        out["timestamp"] = out["timestamp"].replace(["None", "none", "nan", "NaT", ""], pd.NA)
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        out = out.sort_values("timestamp", ascending=False)
        out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        out["timestamp"] = out["timestamp"].fillna("N/A").replace({"NaT": "N/A", "None": "N/A", "nan": "N/A"})

    for col in out.columns:
        out[col] = out[col].fillna("N/A")

    return out.head(50)
    return df[cols].copy().sort_values("timestamp", ascending=False).head(50)