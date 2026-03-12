"""Generate synthetic IGA access-review data with realistic lifecycle scenarios.

Enhancements over the original version
---------------------------------------
* Temporal fields: hire_date, department_change_date, previous_department,
  last_recertification_date, entitlement_grant_date, user_status
* Explicit Mover / Leaver / SoD anomaly scenarios woven into the data
* Confidence score derived from distance-to-threshold logic
"""

from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path

SEED = 42
N_CASES = 240

DEPARTMENTS = ["Finance", "HR", "IT", "Sales", "Operations", "Legal"]
APPLICATIONS = ["ERP", "CRM", "IAM", "Payroll", "Ticketing", "BI"]
ROLES = ["Employee", "Manager", "Analyst", "Admin", "Contractor"]
ENTITLEMENTS = [
    "read_reports",
    "approve_payments",
    "manage_identities",
    "export_data",
    "admin_console",
    "user_support",
]

TODAY = date(2026, 3, 12)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def weighted_recommendation(row: dict) -> tuple[int, str, str]:
    """Transparent, rule-based scoring for IGA review recommendations."""
    score = 0
    reasons: list[str] = []

    # --- inactivity ---
    if row["last_login_days"] > 120:
        score += 35
        reasons.append("inactive_login>120d (+35)")
    elif row["last_login_days"] > 60:
        score += 20
        reasons.append("inactive_login>60d (+20)")
    elif row["last_login_days"] > 30:
        score += 10
        reasons.append("inactive_login>30d (+10)")

    # --- stale access ---
    if row["stale_access_days"] > 365:
        score += 30
        reasons.append("stale_access>365d (+30)")
    elif row["stale_access_days"] > 180:
        score += 18
        reasons.append("stale_access>180d (+18)")
    elif row["stale_access_days"] > 90:
        score += 8
        reasons.append("stale_access>90d (+8)")

    # --- privilege level ---
    if row["privilege_level"] == "high":
        score += 25
        reasons.append("high_privilege (+25)")
    elif row["privilege_level"] == "medium":
        score += 10
        reasons.append("medium_privilege (+10)")

    # --- SoD / toxic combo ---
    if row["toxic_combo"] == 1:
        score += 40
        reasons.append("toxic_combination (+40)")

    # --- sensitive role ---
    if row["role"] in {"Contractor", "Admin"}:
        score += 10
        reasons.append("sensitive_role (+10)")

    # --- department change (Mover / Privilege Creep) ---
    if row.get("department_change_date"):
        score += 15
        reasons.append("dept_change_privilege_creep (+15)")

    # --- terminated / inactive user (Leaver / Orphan Account) ---
    if row.get("user_status") == "terminated":
        score += 30
        reasons.append("terminated_user_orphan (+30)")
    elif row.get("user_status") == "inactive":
        score += 15
        reasons.append("inactive_user (+15)")

    # --- classify ---
    if score >= 70:
        recommendation = "revoke"
    elif score >= 35:
        recommendation = "review"
    else:
        recommendation = "retain"

    reason_text = "; ".join(reasons) if reasons else "baseline_risk"
    return score, recommendation, reason_text


def compute_confidence(score: int, recommendation: str) -> float:
    """Confidence = distance of score from the nearest decision threshold."""
    if recommendation == "revoke":
        margin = max(score - 70, 0)
    elif recommendation == "review":
        margin = max(min(score - 35, 70 - score), 0)
    else:
        margin = max(35 - score, 0)
    return round(min(99.0, 50.0 + (margin / 35.0) * 49.0), 1)


# ---------------------------------------------------------------------------
# Data generation helpers
# ---------------------------------------------------------------------------

def _random_date(start: date, end: date) -> date:
    delta = (end - start).days
    if delta <= 0:
        return start
    return start + timedelta(days=random.randint(0, delta))


def _make_history_events(row: dict) -> str:
    """Build a semicolon-separated string of lifecycle events for the timeline."""
    events: list[str] = []
    events.append(f"{row['hire_date']}|Joiner|Eintritt ins Unternehmen")
    events.append(f"{row['entitlement_grant_date']}|Grant|Berechtigung '{row['entitlement']}' zugewiesen")
    if row["department_change_date"]:
        events.append(
            f"{row['department_change_date']}|Mover|"
            f"Abteilungswechsel von {row['previous_department']} nach {row['department']}"
        )
    if row["last_recertification_date"]:
        events.append(f"{row['last_recertification_date']}|Recert|Letzte Rezertifizierung (bestätigt)")
    if row["user_status"] == "terminated":
        leave_date = _random_date(TODAY - timedelta(days=60), TODAY - timedelta(days=5))
        events.append(f"{leave_date.isoformat()}|Leaver|Austritt aus dem Unternehmen")
    last_login_date = (TODAY - timedelta(days=row["last_login_days"])).isoformat()
    events.append(f"{last_login_date}|Login|Letzter Login")
    events.sort(key=lambda e: e.split("|")[0])
    return ";".join(events)


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_cases(n_cases: int = N_CASES, seed: int = SEED) -> list[dict]:
    random.seed(seed)

    rows: list[dict] = []
    for i in range(1, n_cases + 1):
        role = random.choices(ROLES, weights=[45, 18, 20, 7, 10], k=1)[0]

        # --- temporal lifecycle fields ---
        hire_date = _random_date(TODAY - timedelta(days=2500), TODAY - timedelta(days=60))
        entitlement_grant_date = _random_date(
            hire_date + timedelta(days=1),
            min(hire_date + timedelta(days=365), TODAY - timedelta(days=10)),
        )

        # ~20 % have a department change (Mover scenario)
        if random.random() < 0.20:
            previous_department = random.choice(DEPARTMENTS)
            department = random.choice([d for d in DEPARTMENTS if d != previous_department])
            dept_change_date = _random_date(
                entitlement_grant_date + timedelta(days=30),
                TODAY - timedelta(days=10),
            )
            department_change_date = dept_change_date.isoformat()
        else:
            department = random.choice(DEPARTMENTS)
            previous_department = ""
            department_change_date = ""

        # ~8 % are terminated (Leaver / Orphan Account scenario)
        if random.random() < 0.08:
            user_status = "terminated"
        elif random.random() < 0.06:
            user_status = "inactive"
        else:
            user_status = "active"

        # last recertification: ~70 % have one
        if random.random() < 0.70:
            last_recertification_date = _random_date(
                TODAY - timedelta(days=365), TODAY - timedelta(days=30)
            ).isoformat()
        else:
            last_recertification_date = ""

        row: dict = {
            "case_id": f"CASE-{i:04d}",
            "user_id": f"USR-{random.randint(1000, 9999)}",
            "department": department,
            "previous_department": previous_department,
            "application": random.choice(APPLICATIONS),
            "role": role,
            "entitlement": random.choice(ENTITLEMENTS),
            "hire_date": hire_date.isoformat(),
            "entitlement_grant_date": entitlement_grant_date.isoformat(),
            "department_change_date": department_change_date,
            "last_recertification_date": last_recertification_date,
            "user_status": user_status,
            "last_login_days": random.randint(0, 200),
            "stale_access_days": random.randint(0, 500),
            "privilege_level": random.choices(
                ["low", "medium", "high"], weights=[55, 30, 15], k=1
            )[0],
            "toxic_combo": random.choices([0, 1], weights=[88, 12], k=1)[0],
        }

        score, recommendation, reason_text = weighted_recommendation(row)
        row["risk_score"] = score
        row["recommendation"] = recommendation
        row["rule_explanation"] = reason_text
        row["confidence"] = compute_confidence(score, recommendation)
        row["history_events"] = _make_history_events(row)

        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def init_decisions_file(path: Path) -> None:
    columns = [
        "timestamp",
        "case_id",
        "reviewer",
        "recommended",
        "final_decision",
        "action_type",
        "comment",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()


def write_cases_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No rows generated")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    cases_rows = generate_cases()
    write_cases_csv(data_dir / "review_cases.csv", cases_rows)
    init_decisions_file(data_dir / "decisions.csv")

    # quick stats
    statuses = {}
    recs = {}
    movers = 0
    for r in cases_rows:
        statuses[r["user_status"]] = statuses.get(r["user_status"], 0) + 1
        recs[r["recommendation"]] = recs.get(r["recommendation"], 0) + 1
        if r["department_change_date"]:
            movers += 1

    print(f"Generated {len(cases_rows)} review cases → data/review_cases.csv")
    print(f"  Recommendations : {recs}")
    print(f"  User statuses   : {statuses}")
    print(f"  Mover scenarios : {movers}")
    print(f"Initialized decisions log → data/decisions.csv")


if __name__ == "__main__":
    main()
