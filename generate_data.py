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

from scoring import evaluate_case

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
MANAGERS_BY_DEPARTMENT = {
    "Finance": ["Nina Weber", "Lars Hoffmann"],
    "HR": ["Mara Klein", "Tobias Franke"],
    "IT": ["Sven Berger", "Julia Neumann"],
    "Sales": ["Pia Walter", "David Brandt"],
    "Operations": ["Lea Richter", "Tim Behrens"],
    "Legal": ["Sara Vogel", "Markus Hahn"],
}
ENTITLEMENT_OWNERS = {
    "read_reports": "Data Governance Team",
    "approve_payments": "Finance Access Owner",
    "manage_identities": "IAM Platform Owner",
    "export_data": "Data Compliance Owner",
    "admin_console": "Security Operations Owner",
    "user_support": "Service Desk Owner",
}

TODAY = date(2026, 3, 12)
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


def _business_need_text(role: str, application: str, entitlement: str, assignment_type: str) -> str:
    if assignment_type == "Role-derived":
        return (
            f"Berechtigung ist für die Rolle '{role}' in der Anwendung '{application}' "
            "innerhalb der Standard-Rollenmatrix erforderlich."
        )
    return (
        f"Direkte Zuweisung für eine operative Tätigkeit in '{application}' "
        f"(Berechtigung: '{entitlement}')."
    )


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
        manager_name = random.choice(MANAGERS_BY_DEPARTMENT.get(department, ["Nicht angegeben"]))
        assignment_type = random.choices(
            ["Direct", "Role-derived"],
            weights=[60, 40] if role in {"Admin", "Contractor"} else [30, 70],
            k=1,
        )[0]
        source_role = f"{department}_{role}_Base" if assignment_type == "Role-derived" else ""
        effective_permission = not (user_status == "terminated" and random.random() < 0.75)
        row["business_need"] = _business_need_text(
            role=role,
            application=str(row["application"]),
            entitlement=str(row["entitlement"]),
            assignment_type=assignment_type,
        )
        row["entitlement_owner"] = ENTITLEMENT_OWNERS.get(str(row["entitlement"]), "Access Owner")
        row["manager_name"] = manager_name
        row["assignment_type"] = assignment_type
        row["source_role"] = source_role
        row["effective_permission"] = bool(effective_permission)

        evaluation = evaluate_case(row)
        row["risk_score"] = int(evaluation["score"])
        row["recommendation"] = str(evaluation["recommendation"])
        row["rule_explanation"] = str(evaluation["reason_text"])
        row["confidence"] = float(evaluation["confidence"])
        row["history_events"] = _make_history_events(row)

        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def init_decisions_file(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DECISION_COLUMNS)
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
