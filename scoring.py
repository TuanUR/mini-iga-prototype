from __future__ import annotations

from typing import Any, Mapping
import math

RETAIN_THRESHOLD = 35
REVOKE_THRESHOLD = 70


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        text = str(value).strip().lower()
        if text in {"", "none", "nan", "nat"}:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    text = str(value).strip()
    return text if text and text.lower() not in {"nan", "nat", "none"} else default


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    text = str(value).strip().lower()
    return text not in {"", "nan", "nat", "none"}


def classify_recommendation(score: int) -> str:
    if score >= REVOKE_THRESHOLD:
        return "revoke"
    if score >= RETAIN_THRESHOLD:
        return "review"
    return "retain"


def compute_confidence(score: int, recommendation: str) -> float:
    if recommendation == "revoke":
        margin = max(score - REVOKE_THRESHOLD, 0)
    elif recommendation == "review":
        margin = max(min(score - RETAIN_THRESHOLD, REVOKE_THRESHOLD - score), 0)
    else:
        margin = max(RETAIN_THRESHOLD - score, 0)
    return round(min(99.0, 50.0 + (margin / RETAIN_THRESHOLD) * 49.0), 1)


def evaluate_case(row: Mapping[str, Any]) -> dict[str, Any]:
    contributions: list[dict[str, Any]] = []

    def add(factor: str, points: int, reason_code: str) -> None:
        contributions.append(
            {"factor": factor, "points": int(points), "reason_code": reason_code}
        )

    last_login_days = _to_int(row.get("last_login_days"))
    stale_access_days = _to_int(row.get("stale_access_days"))
    privilege_level = _to_str(row.get("privilege_level")).lower()
    toxic_combo = _to_int(row.get("toxic_combo"))
    role = _to_str(row.get("role"))
    user_status = _to_str(row.get("user_status"), default="active").lower()

    if last_login_days > 120:
        add("Inactive login > 120 days", 35, "inactive_login>120d")
    elif last_login_days > 60:
        add("Inactive login > 60 days", 20, "inactive_login>60d")
    elif last_login_days > 30:
        add("Inactive login > 30 days", 10, "inactive_login>30d")
    elif last_login_days <= 7:
        add("Very recent login (<=7 days)", -15, "very_recent_login<=7d")
    else:
        add("Recent login (<=30 days)", -10, "recent_login<=30d")

    if stale_access_days > 365:
        add("Stale access > 365 days", 30, "stale_access>365d")
    elif stale_access_days > 180:
        add("Stale access > 180 days", 18, "stale_access>180d")
    elif stale_access_days > 90:
        add("Stale access > 90 days", 8, "stale_access>90d")
    elif stale_access_days <= 30:
        add("Access recently used (<=30 days)", -12, "access_recently_used<=30d")
    else:
        add("Access used in last quarter", -8, "access_used_last_quarter")

    if privilege_level == "high":
        add("High privilege", 25, "high_privilege")
    elif privilege_level == "medium":
        add("Medium privilege", 10, "medium_privilege")
    else:
        add("Low privilege", -10, "low_privilege")

    if toxic_combo == 1:
        add("SoD conflict (toxic combination)", 40, "toxic_combination")
    else:
        add("No SoD conflict detected", -15, "no_toxic_combination")

    if role in {"Contractor", "Admin"}:
        add("Sensitive role", 10, "sensitive_role")

    if _has_value(row.get("department_change_date")):
        add(
            "Department change (privilege creep risk)",
            15,
            "dept_change_privilege_creep",
        )
    else:
        add("No department change", -5, "no_department_change")

    if user_status == "terminated":
        add("Terminated user (orphan account)", 30, "terminated_user_orphan")
    elif user_status == "inactive":
        add("Inactive user", 15, "inactive_user")
    else:
        add("Active user", -5, "active_user")

    score = int(sum(int(c["points"]) for c in contributions))
    recommendation = classify_recommendation(score)
    confidence = compute_confidence(score, recommendation)
    reason_text = (
        "; ".join(f"{c['reason_code']} ({int(c['points']):+d})" for c in contributions)
        if contributions
        else "baseline_risk"
    )

    # Show the strongest explanatory drivers independent of sign.
    top = sorted(contributions, key=lambda c: abs(int(c["points"])), reverse=True)[:3]
    if top:
        top_reasons = [f"{c['factor']} ({int(c['points']):+d})" for c in top]
    else:
        top_reasons = ["No major risk driver; low-risk baseline (+0)"]

    return {
        "score": score,
        "recommendation": recommendation,
        "confidence": confidence,
        "reason_text": reason_text,
        "top_reasons": top_reasons,
        "contributions": contributions,
    }


def weighted_recommendation(row: Mapping[str, Any]) -> tuple[int, str, str]:
    result = evaluate_case(row)
    return int(result["score"]), str(result["recommendation"]), str(result["reason_text"])
