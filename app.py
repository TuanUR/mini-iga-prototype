from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
try:
    from streamlit_plotly_events import plotly_events
except Exception:
    plotly_events = None

from scoring import compute_confidence, evaluate_case

DATA_DIR = Path("data")
CASES_PATH = DATA_DIR / "review_cases.csv"
DECISIONS_PATH = DATA_DIR / "decisions.csv"
EVALUATION_PATH = DATA_DIR / "evaluation_log.csv"
MAIN_TAB_LABELS = ["Übersicht", "Struktur", "Fallprüfung", "Audit Log", "Evaluation"]
DECISION_COLUMNS = [
    "timestamp",
    "case_id",
    "reviewer",
    "recommended",
    "reviewer_decision",
    "final_decision",
    "action_type",
    "comment",
]
EVALUATION_COLUMNS = [
    "timestamp",
    "participant_id",
    "task_1_completed",
    "task_2_completed",
    "task_3_completed",
    "likert_q1",
    "likert_q2",
    "likert_q3",
    "likert_q4",
    "likert_q5",
    "comment",
]
MISSING_TEXT_VALUES = {"", "nan", "nat", "none", "null"}


@st.cache_data
def load_cases(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def ensure_decision_schema(decisions_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize decision log columns to the expected runtime schema."""
    if decisions_df is None or len(decisions_df.columns) == 0:
        return pd.DataFrame(columns=DECISION_COLUMNS)

    normalized = decisions_df.copy()
    source_columns = set(normalized.columns)
    for col in DECISION_COLUMNS:
        if col not in normalized.columns:
            normalized[col] = ""

    # Backward compatibility for older files with only one of these fields.
    if "reviewer_decision" not in source_columns and "final_decision" in source_columns:
        normalized["reviewer_decision"] = normalized["final_decision"]
    if "final_decision" not in source_columns and "reviewer_decision" in source_columns:
        normalized["final_decision"] = normalized["reviewer_decision"]

    normalized = normalized[DECISION_COLUMNS].copy()
    return normalized


def load_decisions(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=DECISION_COLUMNS)

    raw_df = pd.read_csv(path)
    schema_changed = list(raw_df.columns) != DECISION_COLUMNS
    decisions_df = ensure_decision_schema(raw_df)
    decisions_df["recommended"] = decisions_df["recommended"].fillna("").astype(str)
    decisions_df["reviewer_decision"] = decisions_df["reviewer_decision"].fillna("").astype(str)

    def _normalize_reviewer_decision(row: pd.Series) -> str:
        rd = str(row["reviewer_decision"]).strip()
        recommended = str(row["recommended"]).strip()
        if rd == "escalated":
            return "escalate"
        if rd == "confirm":
            return "confirm" if recommended in {"retain", "revoke"} else "escalate"
        if rd == "escalate" or rd.startswith("override_"):
            return rd
        if rd in {"retain", "review", "revoke"}:
            if rd == "review":
                return "escalate"
            if rd == recommended:
                return "confirm"
            return f"override_{rd}"
        return rd

    decisions_df["reviewer_decision"] = decisions_df.apply(_normalize_reviewer_decision, axis=1)

    def _normalize_final_decision(row: pd.Series) -> str:
        reviewer_decision = str(row["reviewer_decision"]).strip()
        recommended = str(row["recommended"]).strip()
        if reviewer_decision == "confirm":
            if recommended in {"retain", "revoke"}:
                return recommended
            return "escalated"
        if reviewer_decision == "override_retain":
            return "retain"
        if reviewer_decision == "override_revoke":
            return "revoke"
        if reviewer_decision == "escalate":
            return "escalated"
        return str(row.get("final_decision", "")).strip()

    decisions_df["final_decision"] = decisions_df.apply(_normalize_final_decision, axis=1)
    decisions_df["action_type"] = decisions_df["reviewer_decision"]

    if len(decisions_df) == 0:
        return decisions_df

    normalized = (
        decisions_df.sort_values("timestamp")
        .drop_duplicates(subset=["case_id"], keep="last")
        .reset_index(drop=True)
    )
    if schema_changed or len(normalized) != len(decisions_df):
        normalized.to_csv(path, index=False)
    return normalized


def is_missing_value(value: object) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    value_str = str(value).strip().lower()
    return value_str in MISSING_TEXT_VALUES


def safe_text(value: object, fallback: str = "–") -> str:
    return fallback if is_missing_value(value) else str(value).strip()


def safe_int(value: object, fallback: int = 0) -> int:
    if is_missing_value(value):
        return fallback
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return fallback


def safe_float(value: object, fallback: float = 0.0) -> float:
    if is_missing_value(value):
        return fallback
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def format_confidence_display(confidence_value: object) -> str:
    ui = get_confidence_ui(safe_float(confidence_value, 0.0))
    return f"{float(ui['value']):.1f}% ({safe_text(ui['label'])})"


def save_decision(entry: dict) -> str:
    decisions_df = load_decisions(DECISIONS_PATH)
    case_id = str(entry["case_id"])
    existing_mask = decisions_df["case_id"].astype(str) == case_id

    if not existing_mask.any():
        updated_df = pd.concat([decisions_df, pd.DataFrame([entry])], ignore_index=True)
        updated_df = updated_df[DECISION_COLUMNS].copy()
        updated_df.to_csv(DECISIONS_PATH, index=False)
        return "created"

    existing_row = decisions_df.loc[existing_mask].iloc[0]
    existing_decision = str(existing_row["reviewer_decision"])
    existing_comment = str(existing_row["comment"]).strip()
    new_decision = str(entry["reviewer_decision"])
    new_comment = str(entry["comment"]).strip()

    if existing_decision == new_decision and existing_comment == new_comment:
        return "unchanged"

    updated_df = decisions_df.loc[~existing_mask].copy()
    updated_df = pd.concat([updated_df, pd.DataFrame([entry])], ignore_index=True)
    updated_df = (
        updated_df.sort_values("timestamp")
        .drop_duplicates(subset=["case_id"], keep="last")
        .reset_index(drop=True)
    )
    updated_df = updated_df[DECISION_COLUMNS].copy()
    updated_df.to_csv(DECISIONS_PATH, index=False)
    return "updated"


def get_current_decision(decisions_df: pd.DataFrame, case_id: str) -> pd.Series | None:
    existing_mask = decisions_df["case_id"].astype(str) == str(case_id)
    if not existing_mask.any():
        return None
    return decisions_df.loc[existing_mask].iloc[0]


def map_action_to_decision(action: str, recommendation_label: str) -> tuple[str, str, str]:
    recommendation = str(recommendation_label).strip()
    if action == "confirm":
        if recommendation not in {"retain", "revoke"}:
            raise ValueError(
                f"Ungültige Kombination: confirm ist für recommendation_label='{recommendation}' nicht zulässig."
            )
        return "confirm", recommendation, "confirm"
    if action == "override_retain":
        return "override_retain", "retain", "override_retain"
    if action == "override_revoke":
        return "override_revoke", "revoke", "override_revoke"
    if action == "escalate":
        return "escalate", "escalated", "escalate"
    raise ValueError(f"Unbekannte Aktion: '{action}'")


def format_system_recommendation(label: str) -> str:
    mapping = {
        "retain": "Beibehalten",
        "review": "Prüfen",
        "revoke": "Entziehen",
    }
    return mapping.get(str(label).strip(), str(label).strip())


def format_reviewer_decision(decision: str) -> str:
    mapping = {
        "confirm": "Bestätigen",
        "override_retain": "Überstimmen: Beibehalten",
        "override_revoke": "Überstimmen: Entziehen",
        "escalate": "Eskalieren",
    }
    return mapping.get(str(decision).strip(), str(decision).strip())


def format_final_decision(decision: str) -> str:
    mapping = {
        "retain": "Beibehalten",
        "revoke": "Entziehen",
        "escalated": "Eskaliert",
    }
    return mapping.get(str(decision).strip(), str(decision).strip())


def format_case_status_label(status: object) -> str:
    return {
        "open": "🟡 Offen",
        "decided": "🟢 Entschieden",
        "escalated": "🔴 Eskaliert",
    }.get(str(status), "🟡 Offen")


def format_factor_label(factor: str) -> str:
    factor_map = {
        "Inactive login > 120 days": "Inaktive Nutzung > 120 Tage",
        "Inactive login > 60 days": "Inaktive Nutzung > 60 Tage",
        "Inactive login > 30 days": "Inaktive Nutzung > 30 Tage",
        "Very recent login (<=7 days)": "Sehr aktuelle Nutzung (<= 7 Tage)",
        "Recent login (<=30 days)": "Aktuelle Nutzung (<= 30 Tage)",
        "Stale access > 365 days": "Unbenutzte Zuweisung > 365 Tage",
        "Stale access > 180 days": "Unbenutzte Zuweisung > 180 Tage",
        "Stale access > 90 days": "Unbenutzte Zuweisung > 90 Tage",
        "Access recently used (<=30 days)": "Berechtigung kürzlich genutzt (<= 30 Tage)",
        "Access used in last quarter": "Berechtigung im letzten Quartal genutzt",
        "High privilege": "Hohes Privileg",
        "Medium privilege": "Mittleres Privileg",
        "Low privilege": "Niedriges Privileg",
        "Low privilege profile": "Niedriges Berechtigungsprofil",
        "Terminated user (orphan account)": "Beendeter Benutzer / Orphan Account",
        "Inactive user": "Inaktiver Benutzer",
        "Active user": "Aktiver Benutzer",
        "SoD conflict (toxic combination)": "SoD-Konflikt",
        "Sensitive role": "Sensible Rolle",
        "Standard role profile": "Standard-Rollenprofil",
        "Department change (privilege creep risk)": "Abteilungswechsel / Privilege Creep-Risiko",
        "No SoD conflict detected": "Kein SoD-Konflikt erkannt",
        "No department change": "Kein Abteilungswechsel",
        "No major risk driver; low-risk baseline": "Keine dominanten Risikotreiber; niedriges Basisrisiko",
    }
    return factor_map.get(str(factor).strip(), str(factor).strip())


def localize_reason_text(reason: object) -> str:
    reason_text = safe_text(reason, fallback="–")
    scored_match = re.match(r"^(.*)\s\(([+-]\d+)\)$", reason_text)
    if scored_match:
        factor_text = format_factor_label(scored_match.group(1).strip())
        return f"{factor_text} ({scored_match.group(2)})"
    factor_localized = format_factor_label(reason_text)
    if factor_localized != reason_text:
        return factor_localized
    replacements = {
        "Revoke recommendation": "Empfehlung: Zuweisung entziehen",
        "Review recommendation": "Empfehlung: Zuweisung prüfen",
        "Retain recommendation": "Empfehlung: Zuweisung beibehalten",
        "high risk score": "hoher Risiko-Score",
        "medium risk score": "mittlerer Risiko-Score",
        "low risk score": "niedriger Risiko-Score",
        "No major risk driver; low-risk baseline": "Keine dominanten Risikotreiber; niedriges Basisrisiko",
    }
    localized = reason_text
    for src, dst in replacements.items():
        localized = localized.replace(src, dst)
    return localized


def derive_decision_drivers(
    contrib_df: pd.DataFrame,
    max_items: int = 4,
    top_reasons: list[str] | None = None,
) -> tuple[list[str], bool]:
    fallback = "Keine dominanten Einzeltreiber; Entscheidung basiert auf dem Gesamtbild."

    if top_reasons:
        localized = [localize_reason_text(r) for r in top_reasons if safe_text(r, "").strip()]
        localized = localized[:max_items]
        if localized:
            return localized, True

    if contrib_df is None or len(contrib_df) == 0 or "points" not in contrib_df.columns:
        return [fallback], False

    drivers_df = contrib_df.copy()
    drivers_df["points"] = pd.to_numeric(drivers_df["points"], errors="coerce").fillna(0.0)
    drivers_df["abs_points"] = drivers_df["points"].abs()
    drivers_df = drivers_df[drivers_df["abs_points"] > 0]
    if len(drivers_df) == 0:
        return [fallback], False

    drivers_df = drivers_df.sort_values("abs_points", ascending=False)
    drivers = [
        format_factor_label(str(row["factor"]))
        for _, row in drivers_df.head(max_items).iterrows()
    ]
    return drivers, True


def format_recommendation_badge(label: str) -> str:
    badge_map = {
        "retain": "🟩 Beibehalten",
        "review": "🟨 Prüfen",
        "revoke": "🟥 Entziehen",
    }
    return badge_map.get(str(label).strip(), str(label).strip())


def get_confidence_ui(confidence_value: float) -> dict[str, object]:
    value = safe_float(confidence_value, 0.0)
    if value >= 85:
        return {
            "level": "hoch",
            "label": "hoch",
            "bg": "#EAF3FF",
            "border": "#2F6FED",
            "text": "#163B7A",
            "value": value,
        }
    if value >= 65:
        return {
            "level": "mittel",
            "label": "mittel",
            "bg": "#EEF3F8",
            "border": "#7A8CA5",
            "text": "#425466",
            "value": value,
        }
    return {
        "level": "unsicher",
        "label": "unsicher – manuelle Prüfung empfohlen",
        "bg": "#F5F7FA",
        "border": "#98A2B3",
        "text": "#475467",
        "value": value,
    }


def compute_similarity_score(row: pd.Series) -> float:
    # Weighted deterministic score: role + entitlement + application + score proximity.
    structural = (
        int(row.get("same_role", 0)) * 35
        + int(row.get("same_entitlement", 0)) * 40
        + int(row.get("same_application", 0)) * 20
    )
    score_distance = float(row.get("score_distance", 100))
    proximity = max(0.0, 25.0 - min(25.0, score_distance))
    raw = structural + proximity  # max 120
    return round((raw / 120.0) * 100.0, 1)


def classify_peer_delta(metric_key: str, selected_value: float, peer_value: float) -> tuple[str, str, str, bool]:
    delta = float(selected_value) - float(peer_value)
    abs_delta = abs(delta)

    cfg = {
        "recommendation_score": {"near": 5.0, "high": 15.0, "risk_when_higher": True},
        "confidence": {"near": 5.0, "high": 12.0, "risk_when_higher": False},
        "last_login_days": {"near": 15.0, "high": 45.0, "risk_when_higher": True},
        "stale_access_days": {"near": 20.0, "high": 60.0, "risk_when_higher": True},
    }[metric_key]

    if abs_delta <= cfg["near"]:
        return "≈ nahe am Peer-Group-Durchschnitt", "nahe", "◑", False

    direction = "über" if delta > 0 else "unter"
    if abs_delta >= cfg["high"]:
        critical = (delta > 0 and cfg["risk_when_higher"]) or (delta < 0 and not cfg["risk_when_higher"])
        if critical:
            return f"▲ deutlich {direction} Peer-Group-Durchschnitt (kritisch)", "kritisch", "⚠️", True
        return f"▲ deutlich {direction} Peer-Group-Durchschnitt", "deutlich", "▲", False

    return f"△ leicht {direction} Peer-Group-Durchschnitt", "leicht", "△", False


def render_case_summary_bar(
    case_row: pd.Series,
    recommendation_label: str,
    recommendation_score: int,
    confidence: float,
    contrib_df: pd.DataFrame,
    top_reasons: list[str] | None = None,
) -> None:
    def _safe_int(value: object) -> int | None:
        try:
            if value is None or str(value).strip() == "":
                return None
            return int(float(value))
        except (TypeError, ValueError):
            return None

    recommendation_text = format_system_recommendation(recommendation_label)
    rec_status_class = {
        "revoke": "status-danger",
        "review": "status-warn",
        "retain": "status-ok",
    }.get(str(recommendation_label), "status-neutral")

    score_val = _safe_int(recommendation_score) or 0
    if score_val >= 70:
        risk_text = "Hohes Risiko"
        risk_class = "status-danger"
        risk_threshold_text = "Score >= 70"
    elif score_val >= 35:
        risk_text = "Mittleres Risiko"
        risk_class = "status-warn"
        risk_threshold_text = "Score >= 35 und < 70"
    else:
        risk_text = "Niedriges Risiko"
        risk_class = "status-ok"
        risk_threshold_text = "Score < 35"

    confidence_ui = get_confidence_ui(float(confidence) if confidence is not None else 0.0)
    confidence_val = float(confidence_ui["value"])

    st.markdown(
        """
        <style>
        .iam-summary-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(140px, 1fr));
            gap: 0.4rem;
            margin-bottom: 0.35rem;
        }
        .iam-summary-card {
            border: 1px solid #d9dce3;
            border-radius: 8px;
            padding: 0.48rem 0.58rem;
            background: #ffffff;
            min-height: 74px;
        }
        .iam-summary-card.primary {
            grid-column: span 1;
            min-height: 74px;
        }
        .iam-summary-card.primary .iam-summary-value {
            font-size: 1.02rem;
            font-weight: 700;
        }
        .iam-summary-card.primary .iam-summary-label {
            font-size: 0.72rem;
        }
        .iam-summary-label {
            font-size: 0.7rem;
            color: #4b5563;
            margin-bottom: 0.18rem;
        }
        .iam-summary-value {
            font-size: 0.9rem;
            font-weight: 600;
            line-height: 1.2;
            color: #111827;
        }
        .iam-summary-sub {
            margin-top: 0.12rem;
            font-size: 0.72rem;
            color: #374151;
        }
        .status-danger { border-left: 4px solid #b91c1c; background: #fef2f2; }
        .status-warn { border-left: 4px solid #ca8a04; background: #fefce8; }
        .status-ok { border-left: 4px solid #15803d; background: #f0fdf4; }
        .status-neutral { border-left: 4px solid #4b5563; background: #f9fafb; }
        @media (max-width: 1200px) {
            .iam-summary-grid {
                grid-template-columns: repeat(2, minmax(132px, 1fr));
            }
            .iam-summary-card.primary {
                grid-column: span 1;
            }
        }
        @media (max-width: 680px) {
            .iam-summary-grid { grid-template-columns: 1fr; }
            .iam-summary-card,
            .iam-summary-card.primary {
                grid-column: span 1;
                min-height: auto;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="iam-summary-grid">
            <div class="iam-summary-card primary {rec_status_class}">
                <div class="iam-summary-label">KI-Empfehlung</div>
                <div class="iam-summary-value">{recommendation_text}</div>
                <div class="iam-summary-sub">Score: {score_val}</div>
            </div>
            <div class="iam-summary-card {risk_class}">
                <div class="iam-summary-label">Risikostufe</div>
                <div class="iam-summary-value" style="font-weight:700;">{risk_text}</div>
                <div class="iam-summary-sub">{risk_threshold_text}</div>
            </div>
            <div class="iam-summary-card"
                 style="background:{confidence_ui['bg']}; border-left:4px solid {confidence_ui['border']};">
                <div class="iam-summary-label">KI-Empfehlungssicherheit</div>
                <div class="iam-summary-value" style="color:{confidence_ui['text']}; font-weight:700;">{confidence_val:.1f}%</div>
                <div class="iam-summary-sub" style="color:{confidence_ui['text']};">Stufe: {confidence_ui['label']}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def clear_case_dialog_state() -> None:
    st.session_state["active_main_tab"] = "Fallprüfung"
    st.session_state.pop("edit_case_id", None)
    st.session_state.pop("pending_case_decision", None)
    active_case_id = str(st.session_state.get("worklist_selected_case_id", "")).strip()
    if active_case_id:
        st.session_state.pop(f"dialog_action_{active_case_id}", None)
        st.session_state.pop(f"dialog_primary_action_{active_case_id}", None)
        st.session_state.pop(f"dialog_secondary_action_{active_case_id}", None)
        st.session_state.pop(f"dialog_comment_{active_case_id}", None)


def clear_confirm_dialog_state() -> None:
    st.session_state["active_main_tab"] = "Fallprüfung"
    st.session_state.pop("confirm_case_id", None)
    st.session_state.pop("pending_case_decision", None)


def return_to_case_dialog_from_confirm() -> None:
    pending = st.session_state.get("pending_case_decision")
    case_id = ""
    if isinstance(pending, dict):
        case_id = str(pending.get("case_id", "")).strip()
    st.session_state.pop("confirm_case_id", None)
    st.session_state.pop("pending_case_decision", None)
    if case_id:
        st.session_state["edit_case_id"] = case_id
    st.session_state["active_main_tab"] = "Fallprüfung"


def sync_main_tab_state() -> None:
    selected_tab = st.session_state.get("main_tabs")
    if isinstance(selected_tab, str) and selected_tab in MAIN_TAB_LABELS:
        st.session_state["active_main_tab"] = selected_tab


@st.dialog("Rezertifizierungsprüfung - Fall bearbeiten", width="large", on_dismiss=clear_case_dialog_state)
def render_case_edit_dialog(
    worklist_df: pd.DataFrame,
    analysis_df: pd.DataFrame,
    decisions_df: pd.DataFrame,
    case_id: str,
) -> None:
    selected = worklist_df.loc[worklist_df["case_id"].astype(str) == str(case_id)]
    if len(selected) == 0:
        st.warning("Der ausgewählte Fall ist im aktuellen Filterkontext nicht mehr verfügbar.")
        if st.button("Schließen", key="close_case_dialog_missing"):
            st.session_state["active_main_tab"] = "Fallprüfung"
            st.session_state.pop("edit_case_id", None)
            st.session_state.pop("pending_case_decision", None)
            st.rerun()
        return

    case_row = selected.iloc[0]
    st.caption(
        "Ein Fall entspricht einer konkreten Nutzer-Berechtigungs-Zuweisung, die im Rahmen der Rezertifizierung geprüft wird."
    )

    recommendation_label, recommendation_score, confidence, top_reasons, contrib_df = explain_case(
        case_row
    )
    current_decision = get_current_decision(decisions_df, str(case_row["case_id"]))
    criticality = (
        "hoch"
        if int(recommendation_score) >= 70
        else "mittel"
        if int(recommendation_score) >= 35
        else "niedrig"
    )

    st.markdown(
        """
        <style>
        .fallkontext-box {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            background: #ffffff;
            padding: 0.35rem 0.5rem;
            margin-bottom: 0.25rem;
        }
        .fallkontext-row {
            display: grid;
            grid-template-columns: 120px 1fr;
            gap: 0.4rem;
            padding: 0.16rem 0;
            border-bottom: 1px solid #f3f4f6;
            align-items: baseline;
        }
        .fallkontext-row:last-child {
            border-bottom: none;
        }
        .fallkontext-key {
            font-size: 0.8rem;
            color: #6b7280;
            font-weight: 400;
            line-height: 1.15;
        }
        .fallkontext-value {
            font-size: 0.95rem;
            color: #111827;
            font-weight: 400;
            line-height: 1.2;
            word-break: break-word;
        }
        .fallkontext-row.highlight {
            background: #f8fafc;
            border-radius: 6px;
            padding: 0.22rem 0.28rem;
            margin: 0.06rem 0;
            border: 1px solid #e5e7eb;
        }
        .fallkontext-row.highlight .fallkontext-key {
            color: #334155;
            font-weight: 700;
        }
        .fallkontext-row.highlight .fallkontext-value {
            color: #0f172a;
            font-weight: 700;
        }
        @media (max-width: 920px) {
            .fallkontext-row {
                grid-template-columns: 1fr;
                gap: 0.12rem;
                padding: 0.2rem 0;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    context_left_col, context_right_col = st.columns([1, 1], gap="large")

    with context_left_col:
        context_rows = [
            ("Fall-ID", safe_text(case_row["case_id"]), ""),
            ("Benutzer", safe_text(case_row["user_id"]), "highlight"),
            ("Abteilung", safe_text(case_row["department"]), ""),
            ("Rolle", safe_text(case_row["role"]), ""),
            ("Anwendung", safe_text(case_row["application"]), ""),
            ("Berechtigung", safe_text(case_row["entitlement"]), "highlight"),
            ("Kritikalität", criticality, ""),
        ]
        context_html = ['<div class="fallkontext-box">']
        context_html.extend(
            [
                (
                    f'<div class="fallkontext-row {row_class}">'
                    f'<div class="fallkontext-key">{key}</div>'
                    f'<div class="fallkontext-value">{value}</div>'
                    "</div>"
                )
                for key, value, row_class in context_rows
            ]
        )
        context_html.append("</div>")
        st.markdown("".join(context_html), unsafe_allow_html=True)

        def _format_effective_permission(value: object) -> str:
            if is_missing_value(value):
                return "Nicht angegeben"
            normalized = str(value).strip().lower()
            if normalized in {"true", "1", "ja", "yes"}:
                return "Ja"
            if normalized in {"false", "0", "nein", "no"}:
                return "Nein"
            return safe_text(value, "Nicht angegeben")

    with context_right_col:
        governance_rows = [
            ("Fachliche Notwendigkeit", safe_text(case_row.get("business_need"), "Nicht angegeben"), ""),
            ("Berechtigungs-Owner", safe_text(case_row.get("entitlement_owner"), "Nicht angegeben"), ""),
            ("Manager", safe_text(case_row.get("manager_name"), "Nicht angegeben"), ""),
            ("Zuweisungstyp", safe_text(case_row.get("assignment_type"), "Nicht angegeben"), ""),
            ("Quellrolle", safe_text(case_row.get("source_role"), "Nicht angegeben"), ""),
            (
                "Effektive Berechtigung",
                _format_effective_permission(case_row.get("effective_permission")),
                "",
            ),
        ]
        governance_html = ['<div class="fallkontext-box">']
        governance_html.extend(
            [
                (
                    f'<div class="fallkontext-row {row_class}">'
                    f'<div class="fallkontext-key">{key}</div>'
                    f'<div class="fallkontext-value">{value}</div>'
                    "</div>"
                )
                for key, value, row_class in governance_rows
            ]
        )
        governance_html.append("</div>")
        st.markdown("".join(governance_html), unsafe_allow_html=True)

    render_case_summary_bar(
        case_row=case_row,
        recommendation_label=recommendation_label,
        recommendation_score=recommendation_score,
        confidence=confidence,
        contrib_df=contrib_df,
        top_reasons=top_reasons,
    )
    st.divider()

    with st.expander("Einfluss einzelner Faktoren", expanded=True):
        contrib_plot_df = contrib_df.copy()
        contrib_plot_df["factor"] = contrib_plot_df["factor"].apply(format_factor_label)

        def _truncate_label(text: str, max_len: int = 34) -> str:
            if len(text) <= max_len:
                return text
            return text[: max_len - 1] + "…"

        contrib_plot_df["direction_text"] = contrib_plot_df["direction"].map(
            {"Risk-up": "▲ Risikoerhöhend", "Risk-down": "▼ Risikomindernd"}
        )
        contrib_plot_df["direction_group"] = contrib_plot_df["direction"].map(
            {"Risk-up": "Risikoerhöhend", "Risk-down": "Risikomindernd"}
        )
        contrib_plot_df["factor_full"] = contrib_plot_df["factor"].astype(str)
        contrib_plot_df["factor_short"] = contrib_plot_df["factor_full"].apply(_truncate_label)
        contrib_plot_df["points_text"] = contrib_plot_df["points"].apply(
            lambda x: f"{int(x):+d}" if pd.notna(x) else "0"
        )
        fig_contrib = px.bar(
            contrib_plot_df,
            x="points",
            y="factor_short",
            orientation="h",
            color="direction_group",
            pattern_shape="direction_group",
            pattern_shape_map={"Risikoerhöhend": "/", "Risikomindernd": "x"},
            color_discrete_map={"Risikoerhöhend": "#b91c1c", "Risikomindernd": "#166534"},
            labels={
                "points": "Punkte",
                "factor_short": "Faktor",
                "direction_group": "",
            },
            title="Faktoren für die KI-Empfehlung",
            text="points_text",
            hover_data={
                "factor_full": True,
                "direction_text": True,
                "points": ":.0f",
                "factor_short": False,
                "direction_group": False,
            },
        )
        fig_contrib.add_vline(x=0, line_width=1, line_dash="dash", line_color="#666666")
        max_abs_points = float(contrib_plot_df["points"].abs().max()) if len(contrib_plot_df) else 1.0
        x_pad = max(6.0, max_abs_points * 0.25)
        fig_contrib.update_traces(
            textposition="outside",
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Richtung: %{customdata[1]}<br>"
                "Beitrag: %{x:+.0f} Punkte<extra></extra>"
            ),
        )
        fig_contrib.update_layout(
            height=440,
            margin=dict(l=180, r=70, t=50, b=120),
            font=dict(size=14),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.22,
                xanchor="center",
                x=0.5,
            ),
        )
        fig_contrib.update_xaxes(
            title_font=dict(size=14),
            tickfont=dict(size=13),
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor="#374151",
            range=[-(max_abs_points + x_pad), max_abs_points + x_pad],
            automargin=True,
        )
        fig_contrib.update_yaxes(
            title_font=dict(size=14),
            tickfont=dict(size=13),
            automargin=True,
        )
        st.plotly_chart(fig_contrib, width="stretch")

    with st.expander("Vergleich mit Peer-Group", expanded=False):
        st.caption(
            "Der Vergleich basiert auf dem vollständigen Datensatz und ist unabhängig von den aktiven Sidebar-Filtern."
        )
        render_peer_group_comparison(analysis_df, case_row)

    with st.expander("Zeitstrahl", expanded=False):
        render_timeline(case_row)

    with st.expander("Aktion", expanded=True):
        st.caption(
            "Die KI gibt eine Empfehlung. Sie treffen die finale Entscheidung: Empfehlung übernehmen, abweichend entscheiden oder zur Klärung eskalieren."
        )
        if current_decision is not None:
            current_comment = safe_text(current_decision["comment"], fallback="").strip()
            st.write("Aktueller Entscheidungsstand")
            st.write(f"- KI-Empfehlung: `{format_system_recommendation(recommendation_label)}`")
            st.write(
                "- Aktion: "
                f"`{format_reviewer_decision(str(current_decision['reviewer_decision']))}`"
            )
            st.write(
                "- Finale Entscheidung: "
                f"`{format_final_decision(safe_text(current_decision.get('final_decision'), '–'))}`"
            )
            st.write(f"- Gespeichert am: `{safe_text(current_decision['timestamp'])}`")
            st.write(f"- Kommentar: `{current_comment if current_comment else '–'}`")
        if recommendation_label == "retain":
            action_options = ["confirm", "override_revoke", "escalate"]
        elif recommendation_label == "revoke":
            action_options = ["confirm", "override_retain", "escalate"]
        else:
            action_options = ["override_retain", "override_revoke", "escalate"]

        primary_labels = {
            "adopt": "KI-Empfehlung übernehmen",
            "override": "Abweichend entscheiden",
            "escalate": "Zur Klärung eskalieren",
        }
        secondary_labels = {
            "retain": "Zuweisung beibehalten",
            "revoke": "Zuweisung entziehen",
        }
        default_primary: str | None = None
        default_secondary: str | None = None
        default_comment = ""
        reviewer = "reviewer_1"
        if current_decision is not None:
            current_action = str(current_decision["reviewer_decision"])
            if current_action == "confirm":
                default_primary = "adopt"
            elif current_action == "override_retain":
                default_primary = "override"
                default_secondary = "retain"
            elif current_action == "override_revoke":
                default_primary = "override"
                default_secondary = "revoke"
            elif current_action == "escalate":
                default_primary = "escalate"
            default_comment = safe_text(current_decision["comment"], fallback="")
            reviewer_value = safe_text(current_decision["reviewer"], fallback="")
            if reviewer_value:
                reviewer = reviewer_value

        primary_key = f"dialog_primary_action_{case_id}"
        secondary_key = f"dialog_secondary_action_{case_id}"
        action_key = f"dialog_action_{case_id}"
        comment_key = f"dialog_comment_{case_id}"
        if primary_key not in st.session_state:
            st.session_state[primary_key] = default_primary
        elif st.session_state[primary_key] not in primary_labels:
            st.session_state[primary_key] = default_primary
        if secondary_key not in st.session_state:
            st.session_state[secondary_key] = default_secondary
        elif st.session_state[secondary_key] not in secondary_labels:
            st.session_state[secondary_key] = default_secondary
        if comment_key not in st.session_state:
            st.session_state[comment_key] = default_comment

        adopt_available = recommendation_label in {"retain", "revoke"}
        primary_cols = st.columns(3)
        for idx, primary_action in enumerate(["adopt", "override", "escalate"]):
            with primary_cols[idx]:
                if st.button(
                    primary_labels[primary_action],
                    key=f"{primary_key}_btn_{primary_action}",
                    type=(
                        "primary"
                        if st.session_state.get(primary_key) == primary_action
                        else "secondary"
                    ),
                    disabled=primary_action == "adopt" and not adopt_available,
                    width="stretch",
                ):
                    st.session_state[primary_key] = primary_action
                    st.rerun()

        if not adopt_available:
            st.caption(
                "Hinweis: Bei KI-Empfehlung „Prüfen“ ist keine direkte Übernahme möglich. Bitte abweichend entscheiden oder eskalieren."
            )

        if st.session_state.get(primary_key) == "override":
            st.write("**Finale Entscheidung bei Abweichung**")
            secondary_cols = st.columns(2)
            for idx, secondary_action in enumerate(["retain", "revoke"]):
                with secondary_cols[idx]:
                    if st.button(
                        secondary_labels[secondary_action],
                        key=f"{secondary_key}_btn_{secondary_action}",
                        type=(
                            "primary"
                            if st.session_state.get(secondary_key) == secondary_action
                            else "secondary"
                        ),
                        width="stretch",
                    ):
                        st.session_state[secondary_key] = secondary_action
                        st.rerun()

        selected_primary = st.session_state.get(primary_key)
        selected_secondary = st.session_state.get(secondary_key)
        selected_action: str | None = None
        selected_primary_label = "Nicht gewählt"
        selected_final_label = "Nicht gewählt"

        if selected_primary == "adopt":
            selected_primary_label = primary_labels["adopt"]
            selected_action = "confirm" if adopt_available else None
        elif selected_primary == "override":
            selected_primary_label = primary_labels["override"]
            if selected_secondary in {"retain", "revoke"}:
                selected_action = f"override_{selected_secondary}"
                selected_final_label = format_final_decision(str(selected_secondary))
        elif selected_primary == "escalate":
            selected_primary_label = primary_labels["escalate"]
            selected_action = "escalate"

        if selected_action is not None:
            try:
                _, final_decision_preview, _ = map_action_to_decision(
                    str(selected_action), recommendation_label
                )
                selected_final_label = format_final_decision(final_decision_preview)
            except ValueError:
                selected_action = None

        if isinstance(selected_action, str) and selected_action in action_options:
            st.info(f"Aktion: {selected_primary_label}")
            if selected_final_label == "Beibehalten":
                st.success(f"Finale Entscheidung: {selected_final_label}")
            elif selected_final_label == "Entziehen":
                st.error(f"Finale Entscheidung: {selected_final_label}")
            elif selected_final_label == "Eskaliert":
                st.warning(f"Finale Entscheidung: {selected_final_label}")
            else:
                st.info(f"Finale Entscheidung: {selected_final_label}")
        else:
            st.info("Bitte wählen Sie eine gültige Aktion.")

        with st.form(key=f"decision_form_{case_id}", clear_on_submit=False):
            comment_input = st.text_area(
                "Kommentar",
                key=comment_key,
                max_chars=500,
                help="Begründung für Audit.",
            )
            submit_pressed = st.form_submit_button("Entscheidung speichern", type="primary")

        if submit_pressed:
            if not isinstance(selected_action, str) or selected_action not in action_options:
                st.warning("Bitte wählen Sie zuerst eine gültige Aktion.")
                st.session_state.pop("pending_case_decision", None)
                return
            normalized_comment = str(comment_input).strip()
            try:
                reviewer_decision, final_decision, action_type = map_action_to_decision(
                    str(selected_action), recommendation_label
                )
            except ValueError as exc:
                st.error(str(exc))
                return
            if current_decision is not None:
                old_decision = str(current_decision["reviewer_decision"]).strip()
                old_comment = safe_text(current_decision["comment"], fallback="").strip()
                if old_decision == reviewer_decision and old_comment == normalized_comment:
                    st.warning("Keine Änderungen zur Speicherung vorhanden.")
                    st.session_state.pop("pending_case_decision", None)
                    return
            st.session_state["pending_case_decision"] = {
                "case_id": safe_text(case_row["case_id"]),
                "user_id": safe_text(case_row["user_id"]),
                "application": safe_text(case_row["application"]),
                "entitlement": safe_text(case_row["entitlement"]),
                "recommendation_label": format_system_recommendation(recommendation_label),
                "reviewer_decision_label": format_reviewer_decision(reviewer_decision),
                "final_decision_label": format_final_decision(final_decision),
                "entry": {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "case_id": case_row["case_id"],
                    "reviewer": reviewer,
                    "recommended": recommendation_label,
                    "reviewer_decision": reviewer_decision,
                    "final_decision": final_decision,
                    "action_type": action_type,
                    "comment": normalized_comment,
                },
            }
            st.session_state["active_main_tab"] = "Fallprüfung"
            st.session_state["confirm_case_id"] = str(case_row["case_id"])
            st.session_state.pop("edit_case_id", None)
            st.rerun()


@st.dialog("Entscheidung bestätigen", width="small", on_dismiss=return_to_case_dialog_from_confirm)
def render_decision_confirm_dialog() -> None:
    pending = st.session_state.get("pending_case_decision")
    if not isinstance(pending, dict):
        st.info("Keine ausstehende Entscheidung vorhanden.")
        return

    pending_entry = pending.get("entry", {})
    if not isinstance(pending_entry, dict):
        st.info("Keine ausstehende Entscheidung vorhanden.")
        return

    st.write("Prüfen Sie die Angaben und speichern Sie final.")
    st.write(f"- Fall-ID: `{safe_text(pending.get('case_id'))}`")
    st.write(f"- Benutzer: `{safe_text(pending.get('user_id'))}`")
    st.write(f"- Anwendung: `{safe_text(pending.get('application'))}`")
    st.write(f"- Berechtigung: `{safe_text(pending.get('entitlement'))}`")
    st.write(f"- KI-Empfehlung: `{safe_text(pending.get('recommendation_label'))}`")
    st.write(
        f"- Aktion: "
        f"`{safe_text(pending.get('reviewer_decision_label', format_reviewer_decision(pending_entry.get('reviewer_decision', '–'))))}`"
    )
    st.write(
        f"- Finale Entscheidung: "
        f"`{safe_text(pending.get('final_decision_label', format_final_decision(pending_entry.get('final_decision', '–'))))}`"
    )
    st.write(f"- Kommentar: `{safe_text(pending_entry.get('comment'), '–')}`")
    _, confirm_col, cancel_col, _ = st.columns([1, 2, 2, 1])
    with confirm_col:
        if st.button("Speichern", type="primary", key="confirm_save_decision", width="stretch"):
            save_result = save_decision(pending_entry)
            if save_result == "unchanged":
                st.session_state["decision_feedback"] = {
                    "type": "warning",
                    "text": "Keine Änderungen zur Speicherung vorhanden.",
                }
            elif str(pending_entry.get("reviewer_decision", "")) == "escalate":
                st.session_state["decision_feedback"] = {
                    "type": "warning",
                    "text": "Der Fall wurde zur weiteren Prüfung eskaliert.",
                }
            elif save_result == "created":
                st.session_state["decision_feedback"] = {
                    "type": "success",
                    "text": "Aktion erfolgreich gespeichert.",
                }
            else:
                st.session_state["decision_feedback"] = {
                    "type": "success",
                    "text": "Aktion erfolgreich aktualisiert.",
                }

            case_id = str(pending.get("case_id", "")).strip()
            if case_id:
                st.session_state.pop(f"dialog_action_{case_id}", None)
                st.session_state.pop(f"dialog_primary_action_{case_id}", None)
                st.session_state.pop(f"dialog_secondary_action_{case_id}", None)
                st.session_state.pop(f"dialog_comment_{case_id}", None)
            st.session_state["active_main_tab"] = "Fallprüfung"
            clear_confirm_dialog_state()
            st.rerun()
    with cancel_col:
        if st.button("Abbrechen", key="cancel_save_decision", width="stretch"):
            st.session_state["active_main_tab"] = "Fallprüfung"
            return_to_case_dialog_from_confirm()
            st.rerun()


def load_evaluations(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=EVALUATION_COLUMNS)
    eval_df = pd.read_csv(path)
    for col in EVALUATION_COLUMNS:
        if col not in eval_df.columns:
            eval_df[col] = ""
    return eval_df[EVALUATION_COLUMNS].copy()


def save_evaluation(entry: dict) -> None:
    eval_df = load_evaluations(EVALUATION_PATH)
    eval_df = pd.concat([eval_df, pd.DataFrame([entry])], ignore_index=True)
    eval_df.to_csv(EVALUATION_PATH, index=False)


def write_empty_csv(path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=columns).to_csv(path, index=False)


def reset_demo_and_evaluation_state() -> dict[str, int]:
    decisions_count = len(load_decisions(DECISIONS_PATH))
    evaluations_count = len(load_evaluations(EVALUATION_PATH))
    write_empty_csv(DECISIONS_PATH, DECISION_COLUMNS)
    write_empty_csv(EVALUATION_PATH, EVALUATION_COLUMNS)
    return {
        "decisions_removed": int(decisions_count),
        "evaluations_removed": int(evaluations_count),
    }


def clear_session_state_after_reset() -> None:
    removable_keys = {
        "confirm_case_id",
        "edit_case_id",
        "pending_case_decision",
        "decision_feedback",
        "worklist_selected_case_id",
        "heatmap_dialog_open",
        "heatmap_selected_cell",
        "case_review_worklist_table",
    }
    for key in list(st.session_state.keys()):
        if (
            key in removable_keys
            or key.startswith("dialog_action_")
            or key.startswith("dialog_primary_action_")
            or key.startswith("dialog_secondary_action_")
            or key.startswith("dialog_comment_")
        ):
            st.session_state.pop(key, None)


def kpi_value(label: str, value: int) -> None:
    st.metric(label=label, value=f"{value:,}")


def _recommendation_cell_style(value: object) -> str:
    text = safe_text(value, "").strip().lower()
    if text == "beibehalten":
        return "background-color: #dcfce7; color: #166534; font-weight: 700;"
    if text == "prüfen":
        return "background-color: #fef9c3; color: #854d0e; font-weight: 700;"
    if text == "entziehen":
        return "background-color: #fee2e2; color: #991b1b; font-weight: 700;"
    return ""


def style_recommendation_column(df: pd.DataFrame, columns: list[str]) -> pd.io.formats.style.Styler | pd.DataFrame:
    if len(df) == 0:
        return df
    existing_cols = [c for c in columns if c in df.columns]
    if not existing_cols:
        return df
    styler = df.style
    for col in existing_cols:
        styler = styler.map(_recommendation_cell_style, subset=[col])
    return styler


def explain_case(case_row: pd.Series) -> tuple[str, int, float, list[str], pd.DataFrame]:
    """Local explanation with score-based recommendation and clear factor view."""
    result = evaluate_case(case_row.to_dict())
    recommendation_label = str(result["recommendation"])
    recommendation_score = int(result["score"])
    confidence = float(result["confidence"])
    top_reasons = [str(r) for r in result.get("top_reasons", [])]

    contrib_rows: list[dict[str, object]] = []
    for contribution in result.get("contributions", []):
        points = int(contribution["points"])
        contrib_rows.append(
            {
                "factor": str(contribution["factor"]),
                "points": points,
                "direction": "Risk-up" if points >= 0 else "Risk-down",
            }
        )

    if not contrib_rows:
        contrib_rows = [
            {
                "factor": "No major risk driver; low-risk baseline",
                "points": 0,
                "direction": "Risk-down",
            }
        ]

    contrib_df = pd.DataFrame(contrib_rows).sort_values("points", ascending=True)

    # Guard: charted factor sum must match model score.
    if int(contrib_df["points"].sum()) != recommendation_score:
        recommendation_score = int(contrib_df["points"].sum())
        recommendation_label = (
            "revoke" if recommendation_score >= 70 else "review" if recommendation_score >= 35 else "retain"
        )
        confidence = float(compute_confidence(recommendation_score, recommendation_label))

    return recommendation_label, recommendation_score, confidence, top_reasons, contrib_df


def confidence_label(confidence: float) -> tuple[str, str]:
    """Return (label, color) for a confidence value."""
    ui = get_confidence_ui(confidence)
    return str(ui["label"]), "🔵"


def render_timeline(case_row: pd.Series) -> None:
    """Render a visual timeline of lifecycle events for a case."""
    history_str = str(case_row.get("history_events", "")).strip()
    if not history_str:
        st.info("Keine Berechtigungshistorie für diesen Fall verfügbar.")
        return

    events = []
    for entry in history_str.split(";"):
        parts = entry.strip().split("|", 2)
        if len(parts) == 3:
            events.append({"date": parts[0], "type": parts[1], "description": parts[2]})

    if not events:
        st.info("Keine Berechtigungshistorie für diesen Fall verfügbar.")
        return

    events_df = pd.DataFrame(events)
    events_df["date"] = pd.to_datetime(events_df["date"], errors="coerce")
    events_df = events_df.dropna(subset=["date"]).sort_values("date")
    if len(events_df) == 0:
        st.info("Keine Berechtigungshistorie für diesen Fall verfügbar.")
        return

    event_meta = {
        "Joiner": {"label": "Joiner", "color": "#2563eb", "priority": 2},
        "Grant": {"label": "Grant", "color": "#0284c7", "priority": 3},
        "Mover": {"label": "Rollenwechsel", "color": "#d97706", "priority": 3},
        "Recert": {"label": "Rezertifizierung", "color": "#7c3aed", "priority": 2},
        "Leaver": {"label": "Leaver", "color": "#b91c1c", "priority": 3},
        "Login": {"label": "Login", "color": "#475569", "priority": 2},
    }
    default_meta = {"label": "Event", "color": "#6b7280", "priority": 1}

    events_df["meta"] = events_df["type"].map(event_meta).apply(
        lambda x: x if isinstance(x, dict) else default_meta
    )
    events_df["event_label"] = events_df["meta"].apply(lambda m: str(m["label"]))
    events_df["color"] = events_df["meta"].apply(lambda m: str(m["color"]))
    events_df["priority"] = events_df["meta"].apply(lambda m: int(m["priority"]))

    y_levels = [0.35, -0.35, 0.2, -0.2]
    events_df["y"] = [y_levels[i % len(y_levels)] for i in range(len(events_df))]
    events_df["text_position"] = events_df["y"].apply(
        lambda y: "top center" if float(y) > 0 else "bottom center"
    )
    events_df["is_last_per_type"] = (
        events_df.groupby("type")["date"].transform("max") == events_df["date"]
    )
    events_df["is_key_event"] = (
        events_df["priority"] >= 3
    ) | (events_df.index == events_df.index.min()) | (events_df.index == events_df.index.max())
    events_df["is_key_event"] = events_df["is_key_event"] | (
        events_df["is_last_per_type"] & events_df["type"].isin(["Login", "Recert", "Mover"])
    )
    events_df["display_text"] = events_df.apply(
        lambda r: str(r["event_label"]) if bool(r["is_key_event"]) else "",
        axis=1,
    )
    events_df["marker_symbol"] = events_df["priority"].map({3: "diamond", 2: "circle", 1: "circle-open"})
    events_df["marker_size"] = events_df["priority"].map({3: 16, 2: 13, 1: 10})

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=events_df["date"],
            y=[0] * len(events_df),
            mode="lines",
            line=dict(color="#cbd5e1", width=2),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    for _, row in events_df.iterrows():
        fig.add_shape(
            type="line",
            x0=row["date"],
            y0=0,
            x1=row["date"],
            y1=float(row["y"]),
            line=dict(color="#cbd5e1", width=1),
        )

    fig.add_trace(
        go.Scatter(
            x=events_df["date"],
            y=events_df["y"],
            mode="markers+text",
            marker=dict(
                size=events_df["marker_size"].tolist(),
                color=events_df["color"].tolist(),
                symbol=events_df["marker_symbol"].tolist(),
                line=dict(width=1, color="#1f2937"),
            ),
            text=events_df["display_text"],
            textposition=events_df["text_position"],
            textfont=dict(size=11),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Datum: %{x|%d.%m.%Y}<br>"
                "Ereignis: %{customdata[1]}<br>"
                "Details: %{customdata[2]}<extra></extra>"
            ),
            customdata=list(
                zip(
                    events_df["type"],
                    events_df["event_label"],
                    events_df["description"],
                )
            ),
            showlegend=False,
        )
    )
    fig.update_layout(
        title="Berechtigungshistorie mit priorisierten Ereignissen",
        xaxis_title="Zeitachse",
        yaxis=dict(visible=False, range=[-0.55, 0.55]),
        height=300,
        margin=dict(l=40, r=40, t=60, b=40),
        showlegend=False,
    )
    fig.add_hline(y=0, line_width=1, line_color="#94a3b8", line_dash="dot")
    st.plotly_chart(fig, width="stretch")
    st.caption("Diamant-Markierung kennzeichnet priorisierte Ereignisse.")


def build_case_scope(cases_df: pd.DataFrame, decisions_df: pd.DataFrame) -> pd.DataFrame:
    scoped = cases_df.copy()
    scoped["recommendation_label"] = scoped["recommendation"].astype(str)
    scoped["recommendation_score"] = pd.to_numeric(scoped["risk_score"], errors="coerce").fillna(0)
    scoped["confidence"] = scoped.apply(
        lambda r: compute_confidence(int(r["recommendation_score"]), str(r["recommendation_label"])),
        axis=1,
    )
    scoped["only_sod_conflicts"] = scoped["toxic_combo"].astype(int) == 1
    decisions_view = decisions_df[["case_id", "reviewer_decision", "action_type", "timestamp", "comment"]].copy()
    decisions_view = decisions_view.rename(
        columns={
            "timestamp": "decision_timestamp",
            "comment": "decision_comment",
        }
    )
    scoped = scoped.merge(decisions_view, how="left", on="case_id")
    scoped["reviewer_decision"] = scoped["reviewer_decision"].fillna("").astype(str)
    scoped["action_type"] = scoped["action_type"].fillna("").astype(str)

    def _status(row: pd.Series) -> str:
        if row["reviewer_decision"] == "escalate":
            return "escalated"
        if row["reviewer_decision"] == "confirm" or row["reviewer_decision"].startswith("override_"):
            return "decided"
        return "open"

    scoped["case_status"] = scoped.apply(_status, axis=1)
    scoped["only_open_cases"] = scoped["case_status"] == "open"
    return scoped


def render_global_filters(scope_df: pd.DataFrame) -> dict:
    st.sidebar.info("Die Filter wirken auf alle Hauptansichten.")

    departments = sorted(
        {
            safe_text(v)
            for v in scope_df["department"].tolist()
            if not is_missing_value(v)
        }
    )
    applications = sorted(
        {
            safe_text(v)
            for v in scope_df["application"].tolist()
            if not is_missing_value(v)
        }
    )
    roles = sorted(
        {
            safe_text(v)
            for v in scope_df["role"].tolist()
            if not is_missing_value(v)
        }
    )
    entitlements = sorted(
        {
            safe_text(v)
            for v in scope_df["entitlement"].tolist()
            if not is_missing_value(v)
        }
    )
    employees = sorted(
        {
            safe_text(v)
            for v in scope_df["user_id"].tolist()
            if not is_missing_value(v)
        }
    )
    labels = ["retain", "review", "revoke"]

    min_score = int(scope_df["recommendation_score"].min()) if len(scope_df) else 0
    max_score = int(scope_df["recommendation_score"].max()) if len(scope_df) else 100
    min_conf = float(scope_df["confidence"].min()) if len(scope_df) else 50.0
    max_conf = float(scope_df["confidence"].max()) if len(scope_df) else 99.0

    with st.sidebar.expander("Rolle, Berechtigung, Mitarbeiter", expanded=False):
        selected_roles = st.multiselect("Rolle", roles, default=roles)
        selected_entitlements = st.multiselect(
            "Berechtigung", entitlements, default=entitlements
        )
        selected_users = st.multiselect("Mitarbeiter", employees, default=employees)

    return {
        "department": st.sidebar.multiselect("Abteilung", departments, default=departments),
        "application": st.sidebar.multiselect("Anwendung", applications, default=applications),
        "role": selected_roles,
        "entitlement": selected_entitlements,
        "user_id": selected_users,
        "recommendation_label": st.sidebar.multiselect(
            "KI-Empfehlung",
            labels,
            default=labels,
            format_func=lambda x: {
                "retain": "Beibehalten",
                "review": "Prüfen",
                "revoke": "Entziehen",
            }.get(x, x),
        ),
        "recommendation_score_range": st.sidebar.slider(
            "KI-Empfehlungs-Score (Bereich)",
            min_value=min_score,
            max_value=max_score,
            value=(min_score, max_score),
            help="Der KI-Empfehlungs-Score ist ein regelbasierter Auffälligkeitswert. Höhere Werte deuten auf einen kritischeren Rezertifizierungsfall hin.",
        ),
        "confidence_range": st.sidebar.slider(
            "KI-Empfehlungssicherheit (Bereich)",
            min_value=float(min_conf),
            max_value=float(max_conf),
            value=(float(min_conf), float(max_conf)),
            step=0.1,
            help="Heuristisch aus der Regel-/Scoring-Logik abgeleitet; kein kalibriertes Unsicherheitsmaß eines trainierten Modells.",
        ),
        "only_sod_conflicts": st.sidebar.checkbox("Nur SoD-Konflikte", value=False),
        "only_open_cases": st.sidebar.checkbox("Nur offene Fälle (zusätzlich)", value=False),
        "case_status": st.sidebar.selectbox(
            "Fallstatus",
            options=["all", "open", "decided", "escalated"],
            index=0,
            format_func=lambda x: {
                "all": "alle",
                "open": "offen",
                "decided": "entschieden",
                "escalated": "eskaliert",
            }[x],
        ),
    }


def apply_global_filters(scope_df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    filtered = scope_df.copy()
    filtered = filtered[filtered["department"].astype(str).isin(filters["department"])]
    filtered = filtered[filtered["application"].astype(str).isin(filters["application"])]
    filtered = filtered[filtered["role"].astype(str).isin(filters["role"])]
    filtered = filtered[filtered["entitlement"].astype(str).isin(filters["entitlement"])]
    filtered = filtered[filtered["user_id"].astype(str).isin(filters["user_id"])]
    filtered = filtered[
        filtered["recommendation_label"].astype(str).isin(filters["recommendation_label"])
    ]
    s_min, s_max = filters["recommendation_score_range"]
    filtered = filtered[
        (filtered["recommendation_score"] >= s_min) & (filtered["recommendation_score"] <= s_max)
    ]
    c_min, c_max = filters["confidence_range"]
    filtered = filtered[(filtered["confidence"] >= c_min) & (filtered["confidence"] <= c_max)]
    if filters["only_sod_conflicts"]:
        filtered = filtered[filtered["only_sod_conflicts"]]
    if filters["only_open_cases"]:
        filtered = filtered[filtered["only_open_cases"]]
    if filters["case_status"] != "all":
        filtered = filtered[filtered["case_status"] == filters["case_status"]]
    return filtered


def _safe_get(obj: object, key: str) -> object | None:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    val = getattr(obj, key, None)
    if val is not None:
        return val
    if hasattr(obj, "get"):
        try:
            return obj.get(key)
        except Exception:
            return None
    return None


def build_worklist_table(cases_df: pd.DataFrame) -> pd.DataFrame:
    table_df = cases_df[
        [
            "case_id",
            "user_id",
            "department",
            "role",
            "application",
            "entitlement",
            "recommendation_label",
            "recommendation_score",
            "confidence",
            "case_status",
        ]
    ].copy()
    table_df = table_df.rename(columns={"user_id": "user_name"})
    for col in ["case_id", "user_name", "department", "role", "application", "entitlement"]:
        table_df[col] = table_df[col].apply(lambda v: safe_text(v, fallback="–"))
    table_df["recommendation_label"] = table_df["recommendation_label"].apply(
        lambda v: safe_text(v, fallback="–")
    )
    table_df["recommendation_score"] = table_df["recommendation_score"].apply(
        lambda v: safe_int(v, 0)
    )
    table_df["confidence"] = table_df["confidence"].apply(safe_float)
    table_df["case_status"] = table_df["case_status"].apply(format_case_status_label)
    table_df = table_df.sort_values("recommendation_score", ascending=False).reset_index(drop=True)
    return table_df


def extract_single_selected_case_id(
    selection_event: object, table_df: pd.DataFrame, widget_key: str
) -> str | None:
    selection = _safe_get(selection_event, "selection")
    rows = _safe_get(selection, "rows")
    if not isinstance(rows, (list, tuple)):
        widget_state = st.session_state.get(widget_key, {})
        rows = _safe_get(_safe_get(widget_state, "selection"), "rows")
    if not isinstance(rows, (list, tuple)) or len(rows) != 1:
        return None

    row_idx = rows[0]
    try:
        row_idx_int = int(row_idx)
    except (TypeError, ValueError):
        return None
    if row_idx_int < 0 or row_idx_int >= len(table_df):
        return None
    return str(table_df.iloc[row_idx_int]["case_id"])


def clear_heatmap_selection_state() -> None:
    st.session_state["heatmap_dialog_open"] = False
    st.session_state.pop("heatmap_selected_cell", None)


def compute_cell_metrics(cell_df: pd.DataFrame) -> dict:
    total_cases = int(len(cell_df))
    retain_cases = int((cell_df["recommendation_label"] == "retain").sum()) if total_cases else 0
    review_cases = int((cell_df["recommendation_label"] == "review").sum()) if total_cases else 0
    revoke_cases = int((cell_df["recommendation_label"] == "revoke").sum()) if total_cases else 0
    critical_rate = ((review_cases + revoke_cases) / total_cases) if total_cases else 0.0
    mean_score = float(cell_df["recommendation_score"].mean()) if total_cases else 0.0
    sod_conflicts = int((cell_df["toxic_combo"].astype(int) == 1).sum()) if total_cases else 0
    sod_rate = (sod_conflicts / total_cases) if total_cases else 0.0
    return {
        "total_cases": total_cases,
        "retain_cases": retain_cases,
        "review_cases": review_cases,
        "revoke_cases": revoke_cases,
        "critical_rate": critical_rate,
        "mean_score": mean_score,
        "sod_conflicts": sod_conflicts,
        "sod_rate": sod_rate,
    }


@st.dialog("Heatmap-Detailansicht", width="large", on_dismiss=clear_heatmap_selection_state)
def show_heatmap_cell_dialog(
    cases_df: pd.DataFrame,
    row_dim: str,
    col_dim: str,
    row_value: str,
    col_value: str,
    heatmap_mode: str,
    metric_name: str,
) -> None:
    dialog_df = cases_df.copy()
    dialog_df[row_dim] = dialog_df[row_dim].apply(lambda v: safe_text(v, "–"))
    dialog_df[col_dim] = dialog_df[col_dim].apply(lambda v: safe_text(v, "–"))
    cell_df = dialog_df[
        (dialog_df[row_dim].astype(str) == str(row_value))
        & (dialog_df[col_dim].astype(str) == str(col_value))
    ].copy()

    if len(cell_df) == 0:
        st.warning("Für die ausgewählte Zelle sind im aktuellen Filterkontext keine Fälle vorhanden.")
        if st.button("Schließen", key="close_heatmap_dialog_empty"):
            clear_heatmap_selection_state()
            st.rerun()
        return

    metric_texts = {
        "Anteil auffälliger Fälle": "Diese Zelle zeigt den Anteil der Fälle mit den Empfehlungen „prüfen“ oder „entziehen“ in der ausgewählten Kombination.",
        "Durchschnittlicher Risiko-Score": "Diese Zelle zeigt den durchschnittlichen Risiko-Score aller Fälle in der ausgewählten Kombination.",
    }
    metric_info = metric_texts.get(metric_name)
    if metric_info:
        st.info(metric_info)

    if heatmap_mode == "Abteilung × Anwendung":
        st.caption(
            "Diese Detailansicht unterstützt die Priorisierung von Rezertifizierungsaktivitäten auf Gruppenebene."
        )
        row_label, col_label = "Abteilung", "Anwendung"
    else:
        st.caption(
            "Diese Detailansicht unterstützt die strukturelle Governance-Analyse von Rollen-Berechtigungs-Kombinationen."
        )
        row_label, col_label = "Rolle", "Berechtigung"

    st.markdown(
        (
            f"<div style='font-size:1.0rem; margin-bottom:0.2rem;'>"
            f"<strong>{row_label}:</strong> "
            f"<span style='font-size:1.35rem; font-weight:700;'>{safe_text(row_value)}</span>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        (
            f"<div style='font-size:1.0rem; margin-bottom:0.6rem;'>"
            f"<strong>{col_label}:</strong> "
            f"<span style='font-size:1.35rem; font-weight:700;'>{safe_text(col_value)}</span>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    metrics = compute_cell_metrics(cell_df)
    m_col_1, m_col_2, m_col_3 = st.columns(3)
    with m_col_1:
        st.metric("Fälle gesamt", metrics["total_cases"])
        st.metric("Anzahl beibehalten", metrics["retain_cases"])
        st.metric("Anzahl prüfen", metrics["review_cases"])
    with m_col_2:
        st.metric("Anzahl entziehen", metrics["revoke_cases"])
        st.metric(
            "Anteil auffälliger Fälle",
            f"{metrics['critical_rate']:.1%}",
            help="Anteil der Fälle mit KI-Empfehlung „Prüfen“ oder „Entziehen“.",
        )
        st.metric(
            "Durchschnittlicher Risiko-Score",
            f"{metrics['mean_score']:.1f}",
            help="Mittelwert des KI-Risiko-Scores aller Fälle.",
        )
    with m_col_3:
        st.metric("SoD-Konflikte gesamt", metrics["sod_conflicts"])
        st.metric("SoD-Konfliktrate", f"{metrics['sod_rate']:.1%}")

    detail_table = cell_df[
        [
            "case_id",
            "user_id",
            "department",
            "application",
            "role",
            "entitlement",
            "recommendation_label",
            "recommendation_score",
            "confidence",
            "toxic_combo",
            "case_status",
        ]
    ].copy()
    detail_table["user_id"] = detail_table["user_id"].apply(lambda v: safe_text(v, "–"))
    detail_table["department"] = detail_table["department"].apply(lambda v: safe_text(v, "–"))
    detail_table["application"] = detail_table["application"].apply(lambda v: safe_text(v, "–"))
    detail_table["role"] = detail_table["role"].apply(lambda v: safe_text(v, "–"))
    detail_table["entitlement"] = detail_table["entitlement"].apply(lambda v: safe_text(v, "–"))
    detail_table["recommendation_label"] = detail_table["recommendation_label"].apply(
        lambda v: safe_text(v, "–")
    )
    detail_table["recommendation_score"] = detail_table["recommendation_score"].apply(
        lambda v: safe_int(v, 0)
    )
    detail_table["confidence"] = detail_table["confidence"].apply(format_confidence_display)
    detail_table["toxic_combo"] = detail_table["toxic_combo"].apply(lambda v: safe_int(v, 0))
    detail_table = detail_table.sort_values("recommendation_score", ascending=False).rename(
        columns={
            "case_id": "Fall-ID",
            "user_id": "Benutzer",
            "department": "Abteilung",
            "application": "Anwendung",
            "role": "Rolle",
            "entitlement": "Berechtigung",
            "recommendation_label": "Empfehlung",
            "recommendation_score": "Risiko-Score",
            "confidence": "KI-Empfehlungssicherheit",
            "toxic_combo": "SoD-Konflikt",
            "case_status": "Fallstatus",
        }
    )
    detail_table["Fallstatus"] = detail_table["Fallstatus"].map(
        {"open": "🟡 Offen", "decided": "🟢 Entschieden", "escalated": "🔴 Eskaliert"}
    ).fillna("🟡 Offen")
    detail_table["Empfehlung"] = detail_table["Empfehlung"].map(
        {"retain": "beibehalten", "review": "prüfen", "revoke": "entziehen"}
    ).fillna("–")
    st.dataframe(
        style_recommendation_column(detail_table, ["Empfehlung"]),
        width="stretch",
        hide_index=True,
    )

    if st.button("Schließen", key="close_heatmap_dialog"):
        clear_heatmap_selection_state()
        st.rerun()


def build_similar_cases(cases_df: pd.DataFrame, case_row: pd.Series, top_n: int = 5) -> pd.DataFrame:
    candidates = cases_df[cases_df["case_id"] != case_row["case_id"]].copy()
    if len(candidates) == 0:
        return candidates

    candidates["same_role"] = (candidates["role"] == case_row["role"]).astype(int)
    candidates["same_entitlement"] = (
        candidates["entitlement"] == case_row["entitlement"]
    ).astype(int)
    candidates["same_application"] = (
        candidates["application"] == case_row["application"]
    ).astype(int)
    candidates["score_distance"] = (
        candidates["recommendation_score"] - int(case_row["recommendation_score"])
    ).abs()
    candidates["similarity_score"] = candidates.apply(compute_similarity_score, axis=1)

    candidates = candidates.sort_values(
        by=[
            "similarity_score",
            "same_role",
            "same_entitlement",
            "same_application",
            "score_distance",
            "case_id",
        ],
        ascending=[False, False, False, False, True, True],
    )

    top = candidates.head(top_n).copy()

    def similarity_reason(row: pd.Series) -> str:
        tags: list[str] = []
        if int(row["same_role"]) == 1:
            tags.append("gleiche Rolle")
        if int(row["same_entitlement"]) == 1:
            tags.append("gleiches Entitlement")
        if int(row["same_application"]) == 1:
            tags.append("gleiche Anwendung")
        if not tags:
            tags.append("nahe Score-Ausprägung")
        return f"{' + '.join(tags)}; Score-Distanz {int(row['score_distance'])}"

    top["similarity_reason"] = top.apply(similarity_reason, axis=1)
    top["recommendation_badge"] = top["recommendation_label"].apply(format_recommendation_badge)
    top["relevance_badge"] = "• Relevant"
    if len(top) > 0:
        top.loc[top.index[0], "relevance_badge"] = "⭐ Höchste Relevanz"
    return top


def render_peer_group_comparison(cases_df: pd.DataFrame, case_row: pd.Series) -> None:
    def _extract_clicked_point_customdata(event_payload: object) -> list[object] | None:
        selection = _safe_get(event_payload, "selection")
        points = _safe_get(selection, "points")
        if isinstance(points, list) and len(points) > 0:
            first = points[0]
            if isinstance(first, dict):
                cd = first.get("customdata")
                if isinstance(cd, (list, tuple)):
                    return list(cd)
        return None

    def _format_case_status(status: object) -> str:
        return format_case_status_label(status)

    peer_df = cases_df[
        (cases_df["case_id"] != case_row["case_id"])
        & (cases_df["role"] == case_row["role"])
        & (cases_df["application"] == case_row["application"])
    ].copy()
    comparison_mode = "gleiche Rolle + gleiche Anwendung"
    if len(peer_df) == 0:
        peer_df = cases_df[
            (cases_df["case_id"] != case_row["case_id"]) & (cases_df["role"] == case_row["role"])
        ].copy()
        comparison_mode = "gleiche Rolle (Fallback)"

    if len(peer_df) < 3:
        st.info("Für diese Kombination steht keine ausreichend große Peer-Group zur Verfügung.")
        return
    comparison_group_key = f"role={safe_text(case_row['role'])}|app={safe_text(case_row['application'])}|mode={comparison_mode}"
    current_case_id = str(case_row["case_id"])

    selected_conf = compute_confidence(
        int(case_row["recommendation_score"]), str(case_row["recommendation_label"])
    )
    peer_conf = peer_df.apply(
        lambda r: compute_confidence(int(r["recommendation_score"]), str(r["recommendation_label"])),
        axis=1,
    ).mean()

    metrics_df = pd.DataFrame(
        [
            {
                "metric_key": "recommendation_score",
                "Metrik": "KI-Empfehlungs-Score",
                "Fallwert": float(case_row["recommendation_score"]),
                "Peer-Group-Durchschnitt": float(peer_df["recommendation_score"].mean()),
                "Einheit": "score",
            },
            {
                "metric_key": "last_login_days",
                "Metrik": "Tage seit letzter Nutzung",
                "Fallwert": float(case_row["last_login_days"]),
                "Peer-Group-Durchschnitt": float(peer_df["last_login_days"].mean()),
                "Einheit": "days",
            },
            {
                "metric_key": "stale_access_days",
                "Metrik": "Tage seit letzter Berechtigungsnutzung",
                "Fallwert": float(case_row["stale_access_days"]),
                "Peer-Group-Durchschnitt": float(peer_df["stale_access_days"].mean()),
                "Einheit": "days",
            },
        ]
    )
    metrics_df["Delta_abs"] = metrics_df["Fallwert"] - metrics_df["Peer-Group-Durchschnitt"]
    metrics_df["Delta_%"] = metrics_df.apply(
        lambda r: 0.0
        if abs(float(r["Peer-Group-Durchschnitt"])) < 1e-9
        else (float(r["Delta_abs"]) / float(r["Peer-Group-Durchschnitt"])) * 100.0,
        axis=1,
    )
    class_info = metrics_df.apply(
        lambda r: classify_peer_delta(
            metric_key=str(r["metric_key"]),
            selected_value=float(r["Fallwert"]),
            peer_value=float(r["Peer-Group-Durchschnitt"]),
        ),
        axis=1,
    )
    metrics_df["Einordnung"] = class_info.apply(lambda x: x[0])
    metrics_df["Einordnungsklasse"] = class_info.apply(lambda x: x[1])
    metrics_df["Kritisch"] = class_info.apply(lambda x: bool(x[3]))
    metrics_df["Auffälligkeit"] = metrics_df["Delta_%"].abs()
    metrics_df = metrics_df.sort_values(
        by=["Kritisch", "Auffälligkeit", "Metrik"], ascending=[False, False, True]
    ).reset_index(drop=True)

    def _fmt_value(value: float, unit: str) -> str:
        if unit == "days":
            return f"{int(round(value))} Tage"
        if unit == "percent":
            return f"{value:.1f}%"
        return f"{int(round(value))}"

    def _fmt_delta_abs(value: float, unit: str) -> str:
        if unit == "days":
            return f"{int(round(value)):+d} Tage"
        if unit == "percent":
            return f"{value:+.1f}%"
        return f"{value:+.1f} Punkte"

    metrics_df["Fallwert (fmt)"] = metrics_df.apply(
        lambda r: _fmt_value(float(r["Fallwert"]), str(r["Einheit"])), axis=1
    )
    metrics_df["Peer-Group-Durchschnitt (fmt)"] = metrics_df.apply(
        lambda r: _fmt_value(float(r["Peer-Group-Durchschnitt"]), str(r["Einheit"])), axis=1
    )
    metrics_df["Delta_abs (fmt)"] = metrics_df.apply(
        lambda r: _fmt_delta_abs(float(r["Delta_abs"]), str(r["Einheit"])), axis=1
    )
    metrics_df["Delta_%_label"] = metrics_df["Delta_%"].map(lambda x: f"{x:+.1f}%")
    metrics_df["Einordnung_anzeige"] = metrics_df["Einordnung"].astype(str)
    metrics_df["Fallwert_Textposition"] = metrics_df.apply(
        lambda r: "middle right"
        if float(r["Fallwert"]) >= float(r["Peer-Group-Durchschnitt"])
        else "middle left",
        axis=1,
    )
    metrics_df["Vergleich_Textposition"] = metrics_df.apply(
        lambda r: "middle left"
        if float(r["Fallwert"]) >= float(r["Peer-Group-Durchschnitt"])
        else "middle right",
        axis=1,
    )

    total_metrics = len(metrics_df)
    critical_count = int((metrics_df["Einordnungsklasse"] == "kritisch").sum())
    clear_count = int((metrics_df["Einordnungsklasse"] == "deutlich").sum())
    if critical_count > 0:
        summary_line_1 = (
            f"Der Fall ist in {critical_count} von {total_metrics} Vergleichsmetriken kritisch erhöht."
        )
    elif clear_count > 0:
        summary_line_1 = (
            f"Der Fall weicht in {clear_count} von {total_metrics} Vergleichsmetriken deutlich vom Gruppenmittel ab."
        )
    else:
        summary_line_1 = (
            "Der Fall liegt über alle Vergleichsmetriken nahe am Gruppenmittel und wirkt insgesamt unauffällig."
        )

    st.markdown("**Peer-Group-Karte**")
    st.markdown(
        f'<div style="margin:0;padding:0;color:#374151;font-size:0.875rem;">{summary_line_1}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <style>
        .peer-card-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(150px, 1fr));
            gap: 0.55rem;
            margin: 0 0 0.5rem 0;
        }
        .peer-card {
            border-radius: 10px;
            border: 1px solid #d1d5db;
            padding: 0.55rem 0.65rem;
        }
        .peer-card-title {
            font-size: 0.78rem;
            color: #334155;
            margin-bottom: 0.2rem;
            font-weight: 600;
        }
        .peer-card-value {
            font-size: 0.88rem;
            color: #0f172a;
            font-weight: 700;
        }
        .peer-card-sub {
            font-size: 0.78rem;
            color: #334155;
            margin-top: 0.1rem;
        }
        .peer-card-eval {
            margin-top: 0.2rem;
            font-size: 0.78rem;
            color: #111827;
            font-weight: 600;
        }
        .peer-ok {
            background: #f0fdf4;
            border-left: 4px solid #15803d;
        }
        .peer-warn {
            background: #fff7ed;
            border-left: 4px solid #c2410c;
        }
        .peer-critical {
            background: #fef2f2;
            border-left: 4px solid #b91c1c;
        }
        @media (max-width: 1200px) {
            .peer-card-grid {
                grid-template-columns: repeat(2, minmax(180px, 1fr));
            }
        }
        @media (max-width: 700px) {
            .peer-card-grid { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    def _peer_card_class(class_name: str) -> str:
        if class_name == "kritisch":
            return "peer-critical"
        if class_name == "deutlich":
            return "peer-warn"
        return "peer-ok"

    def _peer_card_eval(class_name: str) -> str:
        if class_name == "kritisch":
            return "Kritisch"
        if class_name == "deutlich":
            return "Deutlich"
        return "Unauffällig"

    card_order = ["recommendation_score", "last_login_days", "stale_access_days"]
    order_map = {k: i for i, k in enumerate(card_order)}
    card_df = metrics_df.copy()
    card_df["order"] = card_df["metric_key"].map(order_map).fillna(99)
    card_df = card_df.sort_values(
        by=["Kritisch", "Auffälligkeit", "order"], ascending=[False, False, True]
    )

    card_html_parts: list[str] = ['<div class="peer-card-grid">']
    for _, r in card_df.iterrows():
        metric_key = str(r["metric_key"])
        if metric_key == "confidence":
            conf_ui = get_confidence_ui(float(r["Fallwert"]))
            card_class = "peer-card"
            card_style = f' style="background:{conf_ui["bg"]}; border-left:4px solid {conf_ui["border"]};"'
            eval_text = f'Stufe: {safe_text(conf_ui["label"])}'
            value_style = f' style="color:{conf_ui["text"]};"'
        else:
            css_class = _peer_card_class(str(r["Einordnungsklasse"]))
            card_class = f"peer-card {css_class}"
            card_style = ""
            eval_text = _peer_card_eval(str(r["Einordnungsklasse"]))
            value_style = ""
        card_html_parts.append(
            (
                f'<div class="{card_class}"{card_style}>'
                f'<div class="peer-card-title">{safe_text(r["Metrik"])}</div>'
                f'<div class="peer-card-value"{value_style}>Fallwert: {safe_text(r["Fallwert (fmt)"])}</div>'
                f'<div class="peer-card-eval">{eval_text}</div>'
                "</div>"
            )
        )
    card_html_parts.append("</div>")
    st.markdown("".join(card_html_parts), unsafe_allow_html=True)
    fig = go.Figure()
    for _, row in metrics_df.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[float(row["Peer-Group-Durchschnitt"]), float(row["Fallwert"])],
                y=[str(row["Metrik"]), str(row["Metrik"])],
                mode="lines",
                line=dict(
                    color="#b91c1c" if bool(row["Kritisch"]) else "#94a3b8",
                    width=2 if bool(row["Kritisch"]) else 1.5,
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=metrics_df["Peer-Group-Durchschnitt"],
            y=metrics_df["Metrik"],
            mode="markers+text",
            name="Peer-Group-Durchschnitt",
            marker=dict(size=10, color="#f59e0b", symbol="circle", line=dict(width=1, color="#92400e")),
            text=metrics_df["Peer-Group-Durchschnitt (fmt)"],
            textposition=metrics_df["Vergleich_Textposition"],
            textfont=dict(size=11, color="#334155"),
            cliponaxis=False,
            customdata=list(
                zip(
                    metrics_df["Fallwert (fmt)"],
                    metrics_df["Peer-Group-Durchschnitt (fmt)"],
                    metrics_df["Delta_abs (fmt)"],
                    metrics_df["Delta_%_label"],
                    metrics_df["Einordnung_anzeige"],
                    ["peer_group"] * len(metrics_df),
                    metrics_df["metric_key"],
                    [current_case_id] * len(metrics_df),
                    [comparison_group_key] * len(metrics_df),
                    [comparison_mode] * len(metrics_df),
                )
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Fallwert: %{customdata[0]}<br>"
                "Peer-Group-Durchschnitt: %{customdata[1]}<br>"
                "Abweichung: %{customdata[2]} (%{customdata[3]})<br>"
                "Einordnung: %{customdata[4]}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=metrics_df["Fallwert"],
            y=metrics_df["Metrik"],
            mode="markers+text",
            name="Fallwert",
            marker=dict(
                size=12,
                color="#1d4ed8",
                symbol="circle",
                line=dict(width=1.2, color="#1e293b"),
            ),
            text=metrics_df["Fallwert (fmt)"],
            textposition=metrics_df["Fallwert_Textposition"],
            textfont=dict(size=11, color="#1d4ed8"),
            cliponaxis=False,
            customdata=list(
                zip(
                    metrics_df["Fallwert (fmt)"],
                    metrics_df["Peer-Group-Durchschnitt (fmt)"],
                    metrics_df["Delta_abs (fmt)"],
                    metrics_df["Delta_%_label"],
                    metrics_df["Einordnung_anzeige"],
                    ["case"] * len(metrics_df),
                    metrics_df["metric_key"],
                    [current_case_id] * len(metrics_df),
                    [comparison_group_key] * len(metrics_df),
                    [comparison_mode] * len(metrics_df),
                )
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Fallwert: %{customdata[0]}<br>"
                "Peer-Group-Durchschnitt: %{customdata[1]}<br>"
                "Abweichung: %{customdata[2]} (%{customdata[3]})<br>"
                "Einordnung: %{customdata[4]}<extra></extra>"
            ),
        )
    )

    min_value = float(
        min(
            metrics_df["Fallwert"].min(),
            metrics_df["Peer-Group-Durchschnitt"].min(),
        )
    )
    max_value = float(
        max(
            metrics_df["Fallwert"].max(),
            metrics_df["Peer-Group-Durchschnitt"].max(),
        )
    )
    span = max(1.0, max_value - min_value)
    x_range = [min_value - span * 0.18, max_value + span * 0.28]

    fig.update_layout(
        title="Direkter Vergleich: Fallwert vs. Peer-Group-Durchschnitt",
        height=350,
        margin=dict(l=170, r=135, t=55, b=105),
        font=dict(size=13),
        legend=dict(orientation="h", yanchor="top", y=-0.28, xanchor="center", x=0.5),
    )
    fig.update_xaxes(
        title="Wert (metrikspezifisch)",
        tickfont=dict(size=12),
        title_font=dict(size=13),
        automargin=True,
        range=x_range,
    )
    fig.update_yaxes(
        tickfont=dict(size=12),
        title_font=dict(size=13),
        automargin=True,
        categoryorder="array",
        categoryarray=list(metrics_df["Metrik"])[::-1],
    )
    fig.update_layout(clickmode="event+select")

    clicked_customdata: list[object] | None = None
    if plotly_events is not None:
        events = plotly_events(
            fig,
            click_event=True,
            hover_event=False,
            select_event=False,
            key=f"peer_group_plot_events_{current_case_id}",
        )
        if isinstance(events, list) and len(events) > 0 and isinstance(events[0], dict):
            cd = events[0].get("customdata")
            if isinstance(cd, (list, tuple)):
                clicked_customdata = list(cd)
    else:
        selection_event = st.plotly_chart(
            fig,
            width="stretch",
            on_select="rerun",
            key=f"peer_group_plot_select_{current_case_id}",
        )
        clicked_customdata = _extract_clicked_point_customdata(selection_event)

    if isinstance(clicked_customdata, list) and len(clicked_customdata) >= 10:
        point_type = str(clicked_customdata[5])
        if point_type == "peer_group":
            st.session_state["peer_group_marker_selection"] = {
                "case_id": str(clicked_customdata[7]),
                "metric_key": str(clicked_customdata[6]),
                "comparison_group_key": str(clicked_customdata[8]),
                "comparison_mode": str(clicked_customdata[9]),
            }

    selected_meta = st.session_state.get("peer_group_marker_selection")
    is_selection_valid = (
        isinstance(selected_meta, dict)
        and str(selected_meta.get("case_id", "")) == current_case_id
        and str(selected_meta.get("comparison_group_key", "")) == comparison_group_key
    )

    if not is_selection_valid:
        st.info(
            "Klicken Sie auf das Peer-Group-Symbol einer Metrik, um die Vergleichsgruppe anzuzeigen."
        )
    else:
        selected_metric_key = str(selected_meta.get("metric_key", ""))
        selected_metric_row = metrics_df[metrics_df["metric_key"].astype(str) == selected_metric_key]
        selected_metric_label = (
            safe_text(selected_metric_row.iloc[0]["Metrik"]) if len(selected_metric_row) else "–"
        )
        st.markdown("**Peer-Group-Metadaten zur gewählten Vergleichsmetrik**")
        st.markdown(
            (
                '<div style="margin:0;padding:0;color:#374151;font-size:0.80rem;line-height:1.2;">'
                f"Ausgewählte Metrik: {selected_metric_label}<br>"
                f"Vergleich anhand: {safe_text(selected_meta.get('comparison_mode'))}<br>"
                f"Gruppengröße: {len(peer_df)}"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        st.markdown('<div style="height:0.6rem;"></div>', unsafe_allow_html=True)

        peer_table = peer_df.copy()
        peer_table["user_id"] = peer_table["user_id"].apply(lambda v: safe_text(v, "–"))
        peer_table["department"] = peer_table["department"].apply(lambda v: safe_text(v, "–"))
        peer_table["role"] = peer_table["role"].apply(lambda v: safe_text(v, "–"))
        peer_table["application"] = peer_table["application"].apply(lambda v: safe_text(v, "–"))
        peer_table["entitlement"] = peer_table["entitlement"].apply(lambda v: safe_text(v, "–"))
        peer_table["recommendation_label"] = peer_table["recommendation_label"].map(
            {"retain": "Beibehalten", "review": "Prüfen", "revoke": "Entziehen"}
        ).fillna("–")
        if "final_decision" in peer_table.columns:
            peer_table["final_decision"] = peer_table["final_decision"].apply(
                lambda v: format_final_decision(safe_text(v, "–"))
            )
        else:
            peer_table["final_decision"] = "–"
        peer_table["recommendation_score"] = peer_table["recommendation_score"].apply(
            lambda v: safe_int(v, 0)
        )
        peer_table["confidence"] = peer_table["confidence"].apply(format_confidence_display)
        peer_table["last_login_days"] = peer_table["last_login_days"].apply(
            lambda v: f"{safe_int(v, 0)} Tage"
        )
        peer_table["stale_access_days"] = peer_table["stale_access_days"].apply(
            lambda v: f"{safe_int(v, 0)} Tage"
        )
        peer_table["toxic_combo"] = peer_table["toxic_combo"].apply(
            lambda v: "Ja" if safe_int(v, 0) == 1 else "Nein"
        )
        peer_table["case_status"] = peer_table["case_status"].apply(_format_case_status)

        peer_table_display = peer_table[
            [
                "case_id",
                "user_id",
                "department",
                "role",
                "application",
                "entitlement",
                "recommendation_label",
                "final_decision",
                "recommendation_score",
                "confidence",
                "last_login_days",
                "stale_access_days",
                "toxic_combo",
                "case_status",
            ]
        ].rename(
            columns={
                "case_id": "Fall-ID",
                "user_id": "Benutzer",
                "department": "Abteilung",
                "role": "Rolle",
                "application": "Anwendung",
                "entitlement": "Berechtigung",
                "recommendation_label": "Empfehlung",
                "final_decision": "Finale Entscheidung",
                "recommendation_score": "KI-Empfehlungs-Score",
                "confidence": "KI-Empfehlungssicherheit",
                "last_login_days": "Tage seit letzter Nutzung",
                "stale_access_days": "Tage seit letzter Berechtigungsnutzung",
                "toxic_combo": "SoD-Konflikt",
                "case_status": "Status",
            }
        ).sort_values("KI-Empfehlungs-Score", ascending=False)
        st.dataframe(
            style_recommendation_column(peer_table_display, ["Empfehlung"]),
            width="stretch",
            hide_index=True,
        )
        if st.button(
            "Auswahl zurücksetzen",
            key=f"peer_group_reset_{current_case_id}",
            width="content",
        ):
            st.session_state.pop("peer_group_marker_selection", None)
            st.rerun()


def render_overview(cases_df: pd.DataFrame, decisions_df: pd.DataFrame) -> None:
    st.subheader("Übersicht")
    if len(cases_df) == 0:
        st.warning("Keine Fälle entsprechen den aktuellen globalen Filtern.")
        return

    total_cases = len(cases_df)
    rec_counts = cases_df["recommendation_label"].value_counts()
    revoke_count = int(rec_counts.get("revoke", 0))
    sod_conflicts = int(cases_df["toxic_combo"].astype(int).sum())
    status_counts = cases_df["case_status"].value_counts()
    open_count = int(status_counts.get("open", 0))
    decided_count = int(status_counts.get("decided", 0))
    escalated_count = int(status_counts.get("escalated", 0))

    kpi_row = st.columns(6)
    with kpi_row[0]:
        kpi_value("Fälle gesamt", total_cases)
    with kpi_row[1]:
        kpi_value("Offen", open_count)
    with kpi_row[2]:
        kpi_value("Entschieden", decided_count)
    with kpi_row[3]:
        kpi_value("Eskaliert", escalated_count)
    with kpi_row[4]:
        kpi_value("Entziehen empfohlen", revoke_count)
    with kpi_row[5]:
        kpi_value("SoD-Konflikte", sod_conflicts)

    charts_col_1, charts_col_2 = st.columns(2)

    with charts_col_1:
        rec_dist = (
            cases_df["recommendation_label"]
            .value_counts()
            .reindex(["retain", "review", "revoke"], fill_value=0)
            .rename_axis("recommendation")
            .reset_index(name="count")
        )
        rec_label_map = {"retain": "Beibehalten", "review": "Prüfen", "revoke": "Entziehen"}
        color_map = {"Beibehalten": "#16a34a", "Prüfen": "#eab308", "Entziehen": "#dc2626"}
        categories = ["Beibehalten", "Prüfen", "Entziehen"]
        rec_dist["Empfehlung"] = rec_dist["recommendation"].map(rec_label_map).fillna("–")
        rec_dist["Anzahl"] = rec_dist["count"].astype(int)
        rec_dist = rec_dist.set_index("Empfehlung").reindex(categories, fill_value=0).reset_index()
        fig_rec = go.Figure()
        fig_rec.add_trace(
            go.Bar(
                x=rec_dist["Empfehlung"],
                y=rec_dist["Anzahl"],
                marker=dict(color=[color_map.get(v, "#9ca3af") for v in rec_dist["Empfehlung"]]),
                showlegend=False,
                hovertemplate="Empfehlung: %{x}<br>Anzahl: %{y}<extra></extra>",
            )
        )
        for label in categories:
            fig_rec.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=10, color=color_map[label], symbol="square"),
                    name=label,
                    showlegend=True,
                    hoverinfo="skip",
                )
            )
        fig_rec.update_layout(
            title="Verteilung der Empfehlungen",
            xaxis_title="Empfehlung",
            yaxis_title="Anzahl",
            xaxis=dict(categoryorder="array", categoryarray=categories),
            legend_title_text="Empfehlung",
            barmode="group",
        )
        st.plotly_chart(fig_rec, width="stretch")

    with charts_col_2:
        fig_hist = px.histogram(
            cases_df,
            x="recommendation_score",
            nbins=16,
            title="Verteilung des Risiko-Scores",
            labels={"recommendation_score": "Risiko-Score", "count": "Anzahl"},
        )
        fig_hist.update_traces(marker_color="#2563eb")
        # Schwellen gemäß bestehender Empfehlungslogik:
        # score < 35 = niedrig, 35-69 = mittel, >= 70 = hoch
        score_min = float(cases_df["recommendation_score"].min()) if len(cases_df) else 0.0
        score_max = float(cases_df["recommendation_score"].max()) if len(cases_df) else 100.0
        fig_hist.add_vrect(
            x0=score_min,
            x1=min(35, score_max),
            fillcolor="#22c55e",
            opacity=0.12,
            line_width=0,
            layer="below",
        )
        fig_hist.add_vrect(
            x0=max(35, score_min),
            x1=min(70, score_max),
            fillcolor="#f59e0b",
            opacity=0.14,
            line_width=0,
            layer="below",
        )
        fig_hist.add_vrect(
            x0=max(70, score_min),
            x1=score_max,
            fillcolor="#ef4444",
            opacity=0.12,
            line_width=0,
            layer="below",
        )
        fig_hist.add_vline(x=35, line_dash="dash", line_color="#b45309", line_width=1.5)
        fig_hist.add_vline(x=70, line_dash="dot", line_color="#b91c1c", line_width=1.8)
        fig_hist.add_annotation(
            x=35,
            y=1.02,
            yref="paper",
            text="35",
            showarrow=False,
            font=dict(size=10, color="#92400e"),
        )
        fig_hist.add_annotation(
            x=70,
            y=1.02,
            yref="paper",
            text="70",
            showarrow=False,
            font=dict(size=10, color="#991b1b"),
        )
        st.plotly_chart(fig_hist, width="stretch")

    charts_col_3, charts_col_4 = st.columns(2)
    with charts_col_3:
        dept_view = cases_df.copy()
        dept_view["department"] = dept_view["department"].apply(lambda v: safe_text(v, "–"))
        dept_scores = (
            dept_view.groupby("department", dropna=False)["recommendation_score"]
            .mean()
            .reset_index(name="mean_score")
            .sort_values("mean_score", ascending=False)
        )
        dept_min = float(dept_scores["mean_score"].min()) if len(dept_scores) else 0.0
        dept_max = float(dept_scores["mean_score"].max()) if len(dept_scores) else 1.0
        dept_range = [dept_min, dept_max] if dept_max > dept_min else [dept_min, dept_min + 1.0]
        fig_dept = px.bar(
            dept_scores,
            x="mean_score",
            y="department",
            orientation="h",
            color="mean_score",
            title="Durchschnittlicher Score pro Abteilung",
            labels={"department": "Abteilung", "mean_score": "Durchschnittlicher Score"},
            color_continuous_scale=["#dbeafe", "#93c5fd", "#2563eb", "#1e3a8a"],
            range_color=dept_range,
        )
        fig_dept.update_layout(
            yaxis=dict(categoryorder="total ascending"),
            coloraxis_colorbar=dict(title="Ø Risiko-Score"),
            showlegend=False,
        )
        st.plotly_chart(fig_dept, width="stretch")

    with charts_col_4:
        app_view = cases_df.copy()
        app_view["application"] = app_view["application"].apply(lambda v: safe_text(v, "–"))
        app_scores = (
            app_view.groupby("application", dropna=False)["recommendation_score"]
            .mean()
            .reset_index(name="mean_score")
            .sort_values("mean_score", ascending=False)
        )
        app_min = float(app_scores["mean_score"].min()) if len(app_scores) else 0.0
        app_max = float(app_scores["mean_score"].max()) if len(app_scores) else 1.0
        app_range = [app_min, app_max] if app_max > app_min else [app_min, app_min + 1.0]
        fig_app = px.bar(
            app_scores,
            x="mean_score",
            y="application",
            orientation="h",
            color="mean_score",
            title="Durchschnittlicher Score pro Anwendung",
            labels={"application": "Anwendung", "mean_score": "Durchschnittlicher Score"},
            color_continuous_scale=["#dbeafe", "#93c5fd", "#2563eb", "#1e3a8a"],
            range_color=app_range,
        )
        fig_app.update_layout(
            yaxis=dict(categoryorder="total ascending"),
            coloraxis_colorbar=dict(title="Ø Risiko-Score"),
            showlegend=False,
        )
        st.plotly_chart(fig_app, width="stretch")

    st.markdown("### Rezertifizierungsfälle")
    st.caption(
        "Die Tabelle zeigt alle aktuell gefilterten Rezertifizierungsfälle. Die Auswahl und Bearbeitung einzelner Fälle erfolgt im Tab „Fallprüfung“."
    )
    worklist_df = build_worklist_table(cases_df)
    overview_table = worklist_df[
        [
            "case_id",
            "user_name",
            "department",
            "application",
            "entitlement",
            "recommendation_label",
            "recommendation_score",
            "case_status",
        ]
    ].rename(
        columns={
            "case_id": "Fall-ID",
            "user_name": "Benutzer",
            "department": "Abteilung",
            "application": "Anwendung",
            "entitlement": "Berechtigung",
            "recommendation_label": "Empfehlung",
            "recommendation_score": "Score",
            "case_status": "Fallstatus",
        }
    )
    overview_table["Empfehlung"] = overview_table["Empfehlung"].map(
        {
            "retain": "Beibehalten",
            "review": "Prüfen",
            "revoke": "Entziehen",
        }
    ).fillna("–")
    st.dataframe(
        style_recommendation_column(overview_table, ["Empfehlung"]),
        width="stretch",
        hide_index=True,
    )


def render_heatmap(cases_df: pd.DataFrame) -> None:
    st.subheader("Struktursicht (Heatmap)")
    if len(cases_df) == 0:
        st.warning("Keine Fälle entsprechen den aktuellen globalen Filtern.")
        clear_heatmap_selection_state()
        return

    heatmap_mode = st.radio(
        " ",
        options=["Abteilung × Anwendung", "Rolle × Berechtigung"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if heatmap_mode == "Abteilung × Anwendung":
        st.caption(
            "Diese Heatmap zeigt aggregierte Muster auf Gruppenebene und unterstützt die Priorisierung von Rezertifizierungsaktivitäten über Abteilungen und Anwendungen."
        )
        row_dim, col_dim = "department", "application"
        y_title, x_title = "Abteilung", "Anwendung"
        default_metric = "Anteil auffälliger Fälle"
    else:
        st.caption(
            "Diese Heatmap zeigt strukturelle Governance-Muster zwischen Rollen und Berechtigungen und macht auffällige oder kritische Kombinationen sichtbar."
        )
        row_dim, col_dim = "role", "entitlement"
        y_title, x_title = "Rolle", "Berechtigung"
        default_metric = "Anteil auffälliger Fälle"

    metric_options = [
        "Anteil auffälliger Fälle",
        "Durchschnittlicher Risiko-Score",
    ]
    metric_help_text = (
        "Metrik-Definitionen:\n"
        "- Anteil auffälliger Fälle = Anteil der Fälle in der Zelle, die vom System mit „Prüfen“ oder „Entziehen“ eingestuft wurden.\n"
        "- Durchschnittlicher Risiko-Score = Mittelwert des KI-Empfehlungs-Scores aller Fälle in der Zelle.\n"
        "Die Werte werden jeweils für die ausgewählte Kombination aus Zeilen- und Spaltenkategorie berechnet."
    )
    selected_metric = st.selectbox(
        "Metrik",
        options=metric_options,
        index=metric_options.index(default_metric),
        help=metric_help_text,
    )
    st.caption(
        "Wählen Sie unterhalb der Heatmap eine Zelle aus, um die zugrunde liegenden Fälle im Detail zu analysieren."
    )

    heatmap_df = cases_df.copy()
    heatmap_df[row_dim] = heatmap_df[row_dim].apply(lambda v: safe_text(v, "–"))
    heatmap_df[col_dim] = heatmap_df[col_dim].apply(lambda v: safe_text(v, "–"))

    agg_df = (
        heatmap_df.groupby([row_dim, col_dim], dropna=False)
        .agg(
            total_cases=("case_id", "count"),
            revoke_cases=("recommendation_label", lambda s: int((s == "revoke").sum())),
            review_cases=("recommendation_label", lambda s: int((s == "review").sum())),
            mean_score=("recommendation_score", "mean"),
            sod_cases=("toxic_combo", lambda s: int((s.astype(int) == 1).sum())),
        )
        .reset_index()
    )
    agg_df["critical_case_rate"] = (
        (agg_df["review_cases"] + agg_df["revoke_cases"]) / agg_df["total_cases"]
    )
    agg_df["sod_conflict_rate"] = agg_df["sod_cases"] / agg_df["total_cases"]

    metric_map = {
        "Anteil auffälliger Fälle": ("critical_case_rate", "Anteil auffälliger Fälle", ".1%"),
        "Durchschnittlicher Risiko-Score": (
            "mean_score",
            "Durchschnittlicher Risiko-Score",
            ".1f",
        ),
    }
    metric_col, color_title, z_format = metric_map[selected_metric]
    if selected_metric == "Anteil auffälliger Fälle":
        heatmap_colorscale = [
            [0.0, "#fff7ec"],
            [0.3, "#fee8c8"],
            [0.55, "#fdbb84"],
            [0.75, "#fc8d59"],
            [1.0, "#b30000"],
        ]
        heatmap_zmin = 0.0
        heatmap_zmax = 1.0
        heatmap_zmid = None
    else:
        heatmap_colorscale = [
            [0.0, "#4e79a7"],   # kühl, negative Werte
            [0.5, "#f5f5f5"],   # neutral um 0
            [1.0, "#e07a2d"],   # warm, positive Werte
        ]
        score_abs_max = float(agg_df["mean_score"].abs().max()) if len(agg_df) else 0.0
        score_abs_max = max(1.0, score_abs_max)
        heatmap_zmin = -score_abs_max
        heatmap_zmax = score_abs_max
        heatmap_zmid = 0.0

    z_matrix = agg_df.pivot(index=row_dim, columns=col_dim, values=metric_col)

    row_labels = z_matrix.index.astype(str).tolist()
    col_labels = z_matrix.columns.astype(str).tolist()
    valid_data_pairs = {
        (str(r), str(c))
        for r, c in agg_df[[row_dim, col_dim]].itertuples(index=False, name=None)
    }

    detail_idx = agg_df.set_index([row_dim, col_dim])
    customdata_data = []
    customdata_no_data = []
    text_data = []
    text_no_data = []
    for row_value in z_matrix.index:
        row_cells_data = []
        row_cells_no_data = []
        row_text_data = []
        row_text_no_data = []
        for col_value in z_matrix.columns:
            if (row_value, col_value) in detail_idx.index:
                d = detail_idx.loc[(row_value, col_value)]
                metric_value = float(d[metric_col])
                row_cells_data.append(
                    [
                        str(row_value),
                        str(col_value),
                        int(d["total_cases"]),
                        int(d["revoke_cases"]),
                        int(d["review_cases"]),
                        float(d["mean_score"]),
                        float(d["critical_case_rate"]),
                        int(d["sod_cases"]),
                    ]
                )
                row_cells_no_data.append([str(row_value), str(col_value)])
                if z_format == ".1%":
                    row_text_data.append(f"{metric_value:.1%}")
                else:
                    row_text_data.append(f"{metric_value:.1f}")
                row_text_no_data.append("")
            else:
                row_cells_data.append([str(row_value), str(col_value), 0, 0, 0, 0.0, 0.0, 0])
                row_cells_no_data.append([str(row_value), str(col_value)])
                row_text_data.append("")
                row_text_no_data.append("Keine Fälle")
        customdata_data.append(row_cells_data)
        customdata_no_data.append(row_cells_no_data)
        text_data.append(row_text_data)
        text_no_data.append(row_text_no_data)

    hovertemplate_data = (
        f"{y_title}: %{{y}}<br>"
        f"{x_title}: %{{x}}<br>"
        "Fälle gesamt: %{customdata[2]}<br>"
        "Davon entziehen empfohlen: %{customdata[3]}<br>"
        "Davon prüfen empfohlen: %{customdata[4]}<br>"
        "Durchschnittlicher Score: %{customdata[5]:.1f}<br>"
        "Anteil auffälliger Fälle: %{customdata[6]:.1%}<br>"
        "SoD-Konflikte: %{customdata[7]}<extra></extra>"
    )
    hovertemplate_no_data = (
        f"{y_title}: %{{customdata[0]}}<br>"
        f"{x_title}: %{{customdata[1]}}<br>"
        "Keine Fälle im aktuellen Filterkontext<extra></extra>"
    )

    no_data_mask = z_matrix.isna()
    z_no_data = no_data_mask.astype(float).where(no_data_mask, other=float("nan"))

    fig = go.Figure(
        data=go.Heatmap(
            z=z_matrix.values,
            x=col_labels,
            y=row_labels,
            text=text_data,
            colorscale=heatmap_colorscale,
            colorbar={"title": color_title},
            customdata=customdata_data,
            hovertemplate=hovertemplate_data,
            hoverongaps=False,
            zmin=heatmap_zmin,
            zmax=heatmap_zmax,
            zmid=heatmap_zmid,
        )
    )
    fig.add_trace(
        go.Heatmap(
            z=z_no_data.values,
            x=col_labels,
            y=row_labels,
            text=text_no_data,
            texttemplate="%{text}",
            textfont={"color": "#6b7280", "size": 11},
            colorscale=[[0, "#e5e7eb"], [1, "#e5e7eb"]],
            showscale=False,
            customdata=customdata_no_data,
            hovertemplate=hovertemplate_no_data,
            hoverongaps=False,
            zmin=0,
            zmax=1,
        )
    )
    fig.update_layout(
        title=f"{heatmap_mode} ({selected_metric})",
        xaxis_title=x_title,
        yaxis_title=y_title,
        clickmode="event+select",
        hoverlabel=dict(
            font=dict(size=20),
            align="left",
        ),
    )
    fig.data[0].update(texttemplate="%{text}")
    st.plotly_chart(
        fig,
        width="stretch",
        key=f"heatmap_view_{heatmap_mode}_{selected_metric}",
    )

    if heatmap_mode == "Abteilung × Anwendung":
        st.caption(
            "In produktiven Umgebungen mit sehr vielen Organisationseinheiten wären zusätzliche Aggregations- und Clustermechanismen erforderlich."
        )
    if selected_metric == "Anteil auffälliger Fälle":
        st.caption(
            "Die Farbintensität zeigt den Anteil auffälliger Fälle in der jeweiligen Zelle. Höhere Werte bedeuten, dass dort relativ mehr prüf- oder entziehungsrelevante Fälle auftreten."
        )
    if selected_metric == "Durchschnittlicher Risiko-Score":
        st.caption(
            "Die Farbskala ist auf 0 zentriert. Werte unter 0 liegen unter dem neutralen Referenzpunkt, Werte über 0 darüber."
        )

    st.markdown("**Zellauswahl für Drill-down**")
    select_col_1, select_col_2, select_col_3 = st.columns([1, 1, 1])
    active_selection = st.session_state.get("heatmap_selected_cell")

    row_default_idx = 0
    col_default_idx = 0
    if isinstance(active_selection, dict):
        if (
            active_selection.get("row_dim") == row_dim
            and active_selection.get("col_dim") == col_dim
            and active_selection.get("heatmap_mode") == heatmap_mode
        ):
            current_row = str(active_selection.get("row_value", ""))
            current_col = str(active_selection.get("col_value", ""))
            if current_row in row_labels:
                row_default_idx = row_labels.index(current_row)
            if current_col in col_labels:
                col_default_idx = col_labels.index(current_col)

    with select_col_1:
        selected_row = st.selectbox(
            y_title,
            options=row_labels,
            index=row_default_idx,
            key=f"heatmap_row_selector_{heatmap_mode}_{selected_metric}",
        )
    with select_col_2:
        selected_col = st.selectbox(
            x_title,
            options=col_labels,
            index=col_default_idx,
            key=f"heatmap_col_selector_{heatmap_mode}_{selected_metric}",
        )
    with select_col_3:
        st.markdown("<div style='height: 1.9rem;'></div>", unsafe_allow_html=True)
        open_detail = st.button(
            "Detailansicht öffnen",
            key=f"heatmap_open_detail_{heatmap_mode}_{selected_metric}",
            width="stretch",
        )

    chosen_cell = (str(selected_row), str(selected_col))
    if chosen_cell in detail_idx.index:
        d = detail_idx.loc[chosen_cell]
        metric_preview = (
            f"{float(d[metric_col]):.1%}" if z_format == ".1%" else f"{float(d[metric_col]):.1f}"
        )
        st.info(
            f"Ausgewählte Zelle: {y_title} = {safe_text(selected_row)}, {x_title} = {safe_text(selected_col)} | "
            f"{selected_metric}: {metric_preview} | Fälle: {int(d['total_cases'])}"
        )
    else:
        st.info(
            f"Ausgewählte Zelle: {y_title} = {safe_text(selected_row)}, {x_title} = {safe_text(selected_col)}"
        )

    if open_detail:
        if chosen_cell in valid_data_pairs:
            selected_context = (
                f"{heatmap_mode}|{selected_metric}|{chosen_cell[0]}|{chosen_cell[1]}"
            )
            st.session_state["heatmap_selected_cell"] = {
                "row_dim": row_dim,
                "col_dim": col_dim,
                "row_value": chosen_cell[0],
                "col_value": chosen_cell[1],
                "heatmap_mode": heatmap_mode,
                "metric": selected_metric,
                "context": selected_context,
            }
            st.session_state["heatmap_dialog_open"] = True
        else:
            clear_heatmap_selection_state()
            st.info("Für die ausgewählte Zelle liegen keine Fälle vor.")

    active_selection = st.session_state.get("heatmap_selected_cell")
    dialog_open = bool(st.session_state.get("heatmap_dialog_open", False))

    if not dialog_open or not isinstance(active_selection, dict):
        return

    if (
        active_selection.get("row_dim") != row_dim
        or active_selection.get("col_dim") != col_dim
        or active_selection.get("heatmap_mode") != heatmap_mode
        or active_selection.get("metric") != selected_metric
    ):
        clear_heatmap_selection_state()
        return

    row_value = str(active_selection.get("row_value", ""))
    col_value = str(active_selection.get("col_value", ""))
    expected_context = f"{heatmap_mode}|{selected_metric}|{chosen_cell[0]}|{chosen_cell[1]}"
    if str(active_selection.get("context", "")) != expected_context:
        clear_heatmap_selection_state()
        return

    if row_value == "" or col_value == "":
        clear_heatmap_selection_state()
        return

    if (row_value, col_value) not in valid_data_pairs:
        clear_heatmap_selection_state()
        return

    show_heatmap_cell_dialog(
        cases_df=cases_df,
        row_dim=row_dim,
        col_dim=col_dim,
        row_value=row_value,
        col_value=col_value,
        heatmap_mode=heatmap_mode,
        metric_name=str(active_selection.get("metric", selected_metric)),
    )


def render_case_view(cases_df: pd.DataFrame, decisions_df: pd.DataFrame) -> None:
    st.subheader("Fallprüfung")
    st.caption(
        "Wählen Sie einen Fall aus der Tabelle und öffnen Sie diesen zur Prüfung. Nach dem Speichern können Sie Ihre Entscheidung direkt im Tab „Audit Log“ prüfen."
    )
    st.caption(
        "Ein Fall entspricht einer konkreten Nutzer-Berechtigungs-Zuweisung, die im Rahmen der Rezertifizierung geprüft wird."
    )

    if len(cases_df) == 0:
        st.warning("Keine Fälle im aktuellen Filterkontext.")
        st.session_state.pop("worklist_selected_case_id", None)
        st.session_state.pop("edit_case_id", None)
        st.session_state.pop("pending_case_decision", None)
        return

    feedback = st.session_state.pop("decision_feedback", None)
    if isinstance(feedback, dict):
        if feedback.get("type") == "warning":
            st.warning(str(feedback.get("text", "")))
        else:
            st.success(str(feedback.get("text", "")))

    worklist_df = build_worklist_table(cases_df)
    worklist_display_df = worklist_df.rename(
        columns={
            "case_id": "Fall-ID",
            "user_name": "Benutzer",
            "department": "Abteilung",
            "role": "Rolle",
            "application": "Anwendung",
            "entitlement": "Berechtigung",
            "recommendation_label": "Empfehlung",
            "recommendation_score": "Score",
            "confidence": "KI-Empfehlungssicherheit",
            "case_status": "Fallstatus",
        }
    ).copy()
    worklist_display_df["Empfehlung"] = worklist_display_df["Empfehlung"].map(
        {
            "retain": "Beibehalten",
            "review": "Prüfen",
            "revoke": "Entziehen",
        }
    ).fillna("–")
    worklist_display_df["KI-Empfehlungssicherheit"] = worklist_display_df[
        "KI-Empfehlungssicherheit"
    ].apply(format_confidence_display)
    selection_event = st.dataframe(
        style_recommendation_column(worklist_display_df, ["Empfehlung"]),
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="case_review_worklist_table",
    )
    selected_case_id = extract_single_selected_case_id(
        selection_event=selection_event,
        table_df=worklist_df,
        widget_key="case_review_worklist_table",
    )
    valid_case_ids = set(cases_df["case_id"].astype(str))

    has_selection = selected_case_id is not None
    button_clicked = st.button(
        "Fall bearbeiten",
        type="primary" if has_selection else "secondary",
        key="case_view_edit_case_btn",
    )

    if not has_selection:
        st.session_state.pop("worklist_selected_case_id", None)
        if str(st.session_state.get("edit_case_id", "")) not in valid_case_ids:
            st.session_state.pop("edit_case_id", None)
        if button_clicked:
            st.info("Bitte treffen Sie eine Auswahl.")
        return

    st.session_state["worklist_selected_case_id"] = str(selected_case_id)
    if button_clicked:
        st.session_state["active_main_tab"] = "Fallprüfung"
        st.session_state["edit_case_id"] = str(selected_case_id)


def render_audit_log(
    decisions_df: pd.DataFrame, cases_df: pd.DataFrame, filtered_case_ids: set[str]
) -> None:
    st.subheader("Audit Log")
    st.caption(
        "Hier werden KI-Empfehlung, Aktion und finale Entscheidung dokumentiert."
    )
    filtered_log = decisions_df[decisions_df["case_id"].astype(str).isin(filtered_case_ids)].copy()
    if len(filtered_log) == 0:
        st.info("Für den aktuellen Filterkontext liegen noch keine Entscheidungen vor.")
        return

    case_cols = [
        "case_id",
        "user_id",
        "department",
        "application",
        "entitlement",
        "recommendation_label",
        "case_status",
    ]
    case_view = cases_df[case_cols].copy().rename(columns={"user_id": "user_name"})
    merged_log = filtered_log.merge(case_view, on="case_id", how="left")
    merged_log["reviewer_decision"] = merged_log["reviewer_decision"].apply(
        lambda v: safe_text(v, "–")
    )
    merged_log["comment"] = merged_log["comment"].apply(lambda v: safe_text(v, "–"))
    merged_log["timestamp"] = merged_log["timestamp"].apply(lambda v: safe_text(v, "–"))
    merged_log["user_name"] = merged_log["user_name"].apply(lambda v: safe_text(v, "–"))
    merged_log["department"] = merged_log["department"].apply(lambda v: safe_text(v, "–"))
    merged_log["application"] = merged_log["application"].apply(lambda v: safe_text(v, "–"))
    merged_log["entitlement"] = merged_log["entitlement"].apply(lambda v: safe_text(v, "–"))
    merged_log["recommendation_label"] = merged_log["recommendation_label"].apply(
        lambda v: safe_text(v, "–")
    )

    total_decisions = len(merged_log)
    override_mask = merged_log["reviewer_decision"].str.startswith("override_")
    override_count = int(override_mask.sum())
    override_rate = (override_count / total_decisions) if total_decisions else 0.0
    escalations_count = int((merged_log["reviewer_decision"] == "escalate").sum())

    kpi_cols = st.columns(3)
    with kpi_cols[0]:
        kpi_value("Entscheidungen gesamt", total_decisions)
    with kpi_cols[1]:
        st.metric("Override-Rate", f"{override_rate:.1%}")
    with kpi_cols[2]:
        kpi_value("Eskalationen", escalations_count)
    st.caption(
        "Die Override-Rate kann einen indirekten Hinweis darauf liefern, in welchem Maße die KI-Empfehlungen als hilfreich eingestuft werden. Die Interpretation ist kontextabhängig."
    )

    merged_log = merged_log[
        [
            "case_id",
            "user_name",
            "department",
            "application",
            "entitlement",
            "recommendation_label",
            "case_status",
            "reviewer_decision",
            "final_decision",
            "comment",
            "timestamp",
        ]
    ].sort_values("timestamp", ascending=False)
    merged_log["fallstatus"] = merged_log["case_status"].apply(format_case_status_label)
    merged_log["ki_empfehlung"] = merged_log["recommendation_label"].apply(
        lambda v: format_system_recommendation(safe_text(v, "–"))
    )
    merged_log["reviewer_entscheidung"] = merged_log["reviewer_decision"].apply(
        lambda v: format_reviewer_decision(safe_text(v, "–"))
    )
    merged_log["finale_entscheidung"] = merged_log["final_decision"].apply(
        lambda v: format_final_decision(safe_text(v, "–"))
    )
    merged_log["hinweis"] = merged_log["reviewer_decision"].str.startswith("override_").map(
        lambda x: "Abweichung" if x else ""
    )
    display_log = merged_log.rename(
        columns={
            "case_id": "Fall-ID",
            "user_name": "Mitarbeiter",
            "department": "Abteilung",
            "application": "Anwendung",
            "entitlement": "Berechtigung",
            "fallstatus": "Fallstatus",
            "hinweis": "Hinweis",
            "comment": "Kommentar",
            "timestamp": "Zeitstempel",
        }
    )[
        [
            "Fall-ID",
            "Mitarbeiter",
            "Abteilung",
            "Anwendung",
            "Berechtigung",
            "Fallstatus",
            "ki_empfehlung",
            "reviewer_entscheidung",
            "finale_entscheidung",
            "Hinweis",
            "Kommentar",
            "Zeitstempel",
        ]
    ].rename(
        columns={
            "ki_empfehlung": "KI-Empfehlung",
            "reviewer_entscheidung": "Aktion",
            "finale_entscheidung": "Finale Entscheidung",
        }
    )
    st.dataframe(
        display_log,
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        "Audit Log als CSV exportieren",
        data=display_log.to_csv(index=False).encode("utf-8"),
        file_name="audit_log_filtered.csv",
        mime="text/csv",
        key="download_audit_log",
    )


def render_evaluation_mode() -> None:
    st.subheader("Evaluation")
    st.caption("Kompakte Evaluation mit 3 Aufgaben und 5 Likert-Fragen (1-5).")

    reset_feedback = st.session_state.pop("reset_feedback", None)
    if reset_feedback:
        st.success(str(reset_feedback))

    decisions_df = load_decisions(DECISIONS_PATH)
    eval_df = load_evaluations(EVALUATION_PATH)
    status_col_1, status_col_2 = st.columns(2)
    with status_col_1:
        st.metric("Gespeicherte Entscheidungen", len(decisions_df))
    with status_col_2:
        st.metric("Evaluationseinträge", len(eval_df))

    with st.expander("Administration (vor einem Walkthrough)", expanded=False):
        st.caption(
            "Setzt nur Entscheidungen und Evaluationseinträge zurück. Der Falldatensatz bleibt unverändert."
        )
        if st.button(
            "Demo-/Evaluationszustand zurücksetzen",
            key="reset_demo_evaluation_state_btn",
            type="secondary",
        ):
            reset_result = reset_demo_and_evaluation_state()
            clear_session_state_after_reset()
            st.session_state["reset_feedback"] = (
                "Reset abgeschlossen: "
                f"{reset_result['decisions_removed']} Entscheidungen und "
                f"{reset_result['evaluations_removed']} Evaluationseinträge entfernt."
            )
            st.rerun()

    participant_id = st.text_input("Teilnehmer-ID", value="P-001")

    st.write("**Aufgaben (erledigt?)**")
    task_1 = st.checkbox(
        "Identifizieren Sie in der Übersicht und Heatmap die Abteilung mit den meisten kritischen Fällen und filtern Sie darauf."
    )
    task_2 = st.checkbox(
        "Öffnen Sie einen Fall mit hohem Risiko-Score, prüfen Sie die KI-Erklärung und treffen Sie eine begründete Entscheidung mit Kommentar."
    )
    task_3 = st.checkbox(
        "Prüfen Sie im Audit Log, ob Ihre Entscheidung korrekt protokolliert wurde."
    )

    st.write("**Likert-Fragen (1 = stimme gar nicht zu, 5 = stimme voll zu)**")
    q1 = st.radio(
        "1) Die KI-Empfehlungen waren nachvollziehbar.",
        options=[1, 2, 3, 4, 5],
        horizontal=True,
        key="eval_q1",
    )
    q2 = st.radio(
        "2) Die visuelle Erklärung der Risikofaktoren (Faktordiagramm) war hilfreich für meine Entscheidung.",
        options=[1, 2, 3, 4, 5],
        horizontal=True,
        key="eval_q2",
    )
    q3 = st.radio(
        "3) Die Übersicht und KPIs haben mir geholfen, kritische Bereiche schnell zu erkennen.",
        options=[1, 2, 3, 4, 5],
        horizontal=True,
        key="eval_q3",
    )
    q4 = st.radio(
        "4) Der Vergleich mit der Peer-Group und die Berechtigungshistorie lieferten nützlichen Kontext.",
        options=[1, 2, 3, 4, 5],
        horizontal=True,
        key="eval_q4",
    )
    q5 = st.radio(
        "5) Ich würde ein solches Dashboard für Zuweisungsprüfungen im Arbeitsalltag nutzen.",
        options=[1, 2, 3, 4, 5],
        horizontal=True,
        key="eval_q5",
    )

    eval_comment = st.text_area("Optionaler Kommentar", max_chars=500, key="eval_comment")

    if st.button("Evaluation speichern", key="save_evaluation_btn"):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "participant_id": participant_id.strip() or "unknown",
            "task_1_completed": int(task_1),
            "task_2_completed": int(task_2),
            "task_3_completed": int(task_3),
            "likert_q1": int(q1),
            "likert_q2": int(q2),
            "likert_q3": int(q3),
            "likert_q4": int(q4),
            "likert_q5": int(q5),
            "comment": eval_comment.strip(),
        }
        save_evaluation(entry)
        st.success("Evaluation gespeichert.")


def inject_dialog_width_css() -> None:
    return


def inject_accessibility_css() -> None:
    st.markdown(
        """
        <style>
        button:focus-visible,
        [role="button"]:focus-visible,
        textarea:focus-visible {
            outline: 3px solid #1d4ed8 !important;
            outline-offset: 2px !important;
            box-shadow: 0 0 0 2px rgba(29, 78, 216, 0.25) !important;
            border-radius: 8px;
        }
        /* Dropdowns auf neutrales Standardverhalten zurücksetzen (ohne blaue Ummantelung). */
        [data-baseweb="select"] > div,
        [data-baseweb="select"] > div:focus-within,
        [data-baseweb="select"] *:focus,
        [data-baseweb="select"] *:focus-visible {
            outline: none !important;
            box-shadow: none !important;
            border-color: #d1d5db !important;
        }
        [data-testid="stCaptionContainer"] p {
            color: #374151;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Mini-IGA Zuweisungsrezertifizierung", layout="wide")
    inject_dialog_width_css()
    inject_accessibility_css()
    st.title("VA-IGA-System für Berechtigungsrezertifizierung (Demo)")
    st.info(
        "Starten Sie mit der Übersicht, grenzen Sie die Anzahl an Rezertifizierungsfällen mit Filtern ein und führen Sie danach die Fallprüfung durch."
    )

    if not CASES_PATH.exists():
        st.error("Keine Fälledaten gefunden. Bitte zuerst `python generate_data.py` ausführen.")
        return

    cases_df = load_cases(CASES_PATH)
    decisions_df = load_decisions(DECISIONS_PATH)
    analysis_df = build_case_scope(cases_df, decisions_df)
    active_filters = render_global_filters(analysis_df)
    filtered_df = apply_global_filters(analysis_df, active_filters)
    filtered_case_ids = set(filtered_df["case_id"].astype(str).tolist())
    filtered_decisions_df = decisions_df[
        decisions_df["case_id"].astype(str).isin(filtered_case_ids)
    ]

    tab_labels = MAIN_TAB_LABELS
    desired_tab = st.session_state.get("active_main_tab", tab_labels[0])
    if not isinstance(desired_tab, str) or desired_tab not in tab_labels:
        desired_tab = tab_labels[0]
    if st.session_state.get("confirm_case_id") or st.session_state.get("edit_case_id"):
        desired_tab = "Fallprüfung"
    st.session_state["active_main_tab"] = desired_tab

    tab_containers = st.tabs(
        tab_labels,
        default=desired_tab,
        key="main_tabs",
        on_change=sync_main_tab_state,
    )
    tabs_by_label = dict(zip(tab_labels, tab_containers))

    with tabs_by_label["Übersicht"]:
        render_overview(filtered_df, filtered_decisions_df)
    with tabs_by_label["Struktur"]:
        render_heatmap(filtered_df)
    with tabs_by_label["Fallprüfung"]:
        render_case_view(filtered_df, decisions_df)
    with tabs_by_label["Audit Log"]:
        render_audit_log(decisions_df, filtered_df, filtered_case_ids)
    with tabs_by_label["Evaluation"]:
        render_evaluation_mode()

    confirm_case_id = str(st.session_state.get("confirm_case_id", "")).strip()
    edit_case_id = str(st.session_state.get("edit_case_id", "")).strip()
    if confirm_case_id:
        render_decision_confirm_dialog()
    elif edit_case_id:
        render_case_edit_dialog(filtered_df, analysis_df, decisions_df, edit_case_id)

if __name__ == "__main__":
    main()
