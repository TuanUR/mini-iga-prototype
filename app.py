from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA_DIR = Path("data")
CASES_PATH = DATA_DIR / "review_cases.csv"
DECISIONS_PATH = DATA_DIR / "decisions.csv"
EVALUATION_PATH = DATA_DIR / "evaluation_log.csv"
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


@st.cache_data
def load_cases(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_decisions(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=DECISION_COLUMNS)

    decisions_df = pd.read_csv(path)
    source_columns = set(decisions_df.columns)
    for col in DECISION_COLUMNS:
        if col not in decisions_df.columns:
            decisions_df[col] = ""
    if "reviewer_decision" not in source_columns and "final_decision" in source_columns:
        decisions_df["reviewer_decision"] = decisions_df["final_decision"]
    if "final_decision" not in source_columns and "reviewer_decision" in source_columns:
        decisions_df["final_decision"] = decisions_df["reviewer_decision"]
    decisions_df = decisions_df[DECISION_COLUMNS].copy()
    decisions_df["recommended"] = decisions_df["recommended"].fillna("").astype(str)
    decisions_df["reviewer_decision"] = decisions_df["reviewer_decision"].fillna("").astype(str)

    def _normalize_reviewer_decision(row: pd.Series) -> str:
        rd = str(row["reviewer_decision"]).strip()
        recommended = str(row["recommended"]).strip()
        if rd in {"confirm", "escalate"} or rd.startswith("override_"):
            return rd
        if rd in {"retain", "review", "revoke"}:
            return "confirm" if rd == recommended else f"override_{rd}"
        return rd

    decisions_df["reviewer_decision"] = decisions_df.apply(_normalize_reviewer_decision, axis=1)

    if len(decisions_df) == 0:
        return decisions_df

    normalized = (
        decisions_df.sort_values("timestamp")
        .drop_duplicates(subset=["case_id"], keep="last")
        .reset_index(drop=True)
    )
    if len(normalized) != len(decisions_df):
        normalized.to_csv(path, index=False)
    return normalized


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
    if action == "confirm":
        return "confirm", recommendation_label, "confirm"
    if action == "override_retain":
        return "override_retain", "retain", "override_retain"
    if action == "override_revoke":
        return "override_revoke", "revoke", "override_revoke"
    return "escalate", "escalate", "escalated"


@st.dialog("Fall bearbeiten", width="large")
def render_case_edit_dialog(
    cases_df: pd.DataFrame, decisions_df: pd.DataFrame, case_id: str
) -> None:
    selected = cases_df.loc[cases_df["case_id"].astype(str) == str(case_id)]
    if len(selected) == 0:
        st.warning("Der ausgewählte Fall ist im aktuellen Filterkontext nicht mehr verfügbar.")
        if st.button("Schließen", key="close_case_dialog_missing"):
            st.session_state.pop("edit_case_id", None)
            st.session_state.pop("pending_case_decision", None)
            st.rerun()
        return

    case_row = selected.iloc[0]
    recommendation_label, recommendation_score, confidence, top_reasons, contrib_df = explain_case(
        case_row
    )
    current_decision = get_current_decision(decisions_df, str(case_row["case_id"]))
    conf_text, conf_icon = confidence_label(confidence)
    criticality = (
        "hoch"
        if int(recommendation_score) >= 70
        else "mittel"
        if int(recommendation_score) >= 35
        else "niedrig"
    )

    left_col, right_col = st.columns([1, 1.6], gap="large")

    with left_col:
        with st.expander("Abschnitt A: Empfehlung im Überblick", expanded=True):
            overview_df = pd.DataFrame(
                [
                    {"Merkmal": "Fall-ID", "Wert": str(case_row["case_id"])},
                    {"Merkmal": "Benutzer", "Wert": str(case_row["user_id"])},
                    {"Merkmal": "Department", "Wert": str(case_row["department"])},
                    {"Merkmal": "Rolle", "Wert": str(case_row["role"])},
                    {"Merkmal": "Anwendung", "Wert": str(case_row["application"])},
                    {"Merkmal": "Berechtigung", "Wert": str(case_row["entitlement"])},
                    {"Merkmal": "Kritikalität", "Wert": criticality},
                    {"Merkmal": "Aktuelle Empfehlung", "Wert": recommendation_label},
                    {"Merkmal": "Empfehlungs-Score", "Wert": str(int(recommendation_score))},
                    {"Merkmal": "Konfidenz", "Wert": f"{confidence:.1f}% ({conf_icon} {conf_text})"},
                ]
            )
            st.dataframe(overview_df, width="stretch", hide_index=True)

        with st.expander("Abschnitt B: Warum diese Empfehlung?", expanded=True):
            reason_cols = [c for c in ["reason_1", "reason_2", "reason_3"] if c in case_row.index]
            reasons = [str(case_row[c]).strip() for c in reason_cols if str(case_row[c]).strip()]
            if not reasons:
                reasons = top_reasons
            for idx, reason in enumerate(reasons, start=1):
                st.write(f"{idx}. {reason}")

        with st.expander("Abschnitt F: Review-Entscheidung", expanded=True):
            if current_decision is None:
                st.write("Noch keine Entscheidung gespeichert")
            else:
                current_comment = str(current_decision["comment"]).strip()
                st.write("Aktuelle Entscheidung")
                st.write(f"- Status: `{current_decision['reviewer_decision']}`")
                st.write(f"- Gespeichert am: `{current_decision['timestamp']}`")
                st.write(f"- Kommentar: `{current_comment if current_comment else '-'}`")

            action_options = ["confirm", "override_retain", "override_revoke", "escalate"]
            action_labels = {
                "confirm": "Bestätigen",
                "override_retain": "Auf beibehalten ändern",
                "override_revoke": "Auf entziehen ändern",
                "escalate": "Eskalieren",
            }
            default_action = "confirm"
            default_comment = ""
            reviewer = "reviewer_1"
            if current_decision is not None:
                current_action = str(current_decision["reviewer_decision"])
                default_action = current_action if current_action in action_options else "confirm"
                default_comment = str(current_decision["comment"])
                if str(current_decision["reviewer"]).strip():
                    reviewer = str(current_decision["reviewer"])

            action_key = f"dialog_action_{case_id}"
            comment_key = f"dialog_comment_{case_id}"
            if action_key not in st.session_state:
                st.session_state[action_key] = default_action
            if comment_key not in st.session_state:
                st.session_state[comment_key] = default_comment

            st.selectbox(
                "Aktion",
                options=action_options,
                format_func=lambda x: action_labels[x],
                key=action_key,
            )
            st.text_area("Kommentar", key=comment_key, max_chars=500)

            selected_action = str(st.session_state[action_key])
            normalized_comment = str(st.session_state[comment_key]).strip()
            reviewer_decision, final_decision, action_type = map_action_to_decision(
                selected_action, recommendation_label
            )
            button_label = (
                "Entscheidung speichern"
                if current_decision is None
                else "Entscheidung aktualisieren"
            )

            if st.button(button_label, type="primary", key=f"prepare_save_{case_id}"):
                if current_decision is not None:
                    old_decision = str(current_decision["reviewer_decision"]).strip()
                    old_comment = str(current_decision["comment"]).strip()
                    if old_decision == reviewer_decision and old_comment == normalized_comment:
                        st.warning("Keine Änderungen zum Speichern vorhanden.")
                        return
                st.session_state["pending_case_decision"] = {
                    "case_id": str(case_row["case_id"]),
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
                st.rerun()

            pending = st.session_state.get("pending_case_decision")
            if isinstance(pending, dict) and pending.get("case_id") == str(case_row["case_id"]):
                pending_entry = pending["entry"]
                st.markdown("---")
                st.write("**Bestätigung**")
                st.write("Möchten Sie diese Entscheidung wirklich speichern?")
                st.write(f"- Fall-ID: `{case_row['case_id']}`")
                st.write(f"- Benutzer: `{case_row['user_id']}`")
                st.write(f"- Anwendung: `{case_row['application']}`")
                st.write(f"- Berechtigung: `{case_row['entitlement']}`")
                st.write(f"- Aktuelle Empfehlung: `{recommendation_label}`")
                st.write(f"- Neue Reviewer-Entscheidung: `{pending_entry['reviewer_decision']}`")
                st.write(
                    f"- Kommentar: `{pending_entry['comment'] if pending_entry['comment'] else '-'}`"
                )
                confirm_col, cancel_col = st.columns(2)
                with confirm_col:
                    if st.button("Fortfahren", type="primary", key=f"confirm_save_{case_id}"):
                        save_result = save_decision(pending_entry)
                        if save_result == "unchanged":
                            st.warning("Keine Änderungen zum Speichern vorhanden.")
                            st.session_state.pop("pending_case_decision", None)
                            return
                        if pending_entry["reviewer_decision"] == "escalate":
                            st.session_state["decision_feedback"] = {
                                "type": "warning",
                                "text": "Der Fall wurde zur weiteren Prüfung eskaliert.",
                            }
                        elif save_result == "created":
                            st.session_state["decision_feedback"] = {
                                "type": "success",
                                "text": "Entscheidung erfolgreich gespeichert.",
                            }
                        else:
                            st.session_state["decision_feedback"] = {
                                "type": "success",
                                "text": "Entscheidung erfolgreich aktualisiert.",
                            }
                        st.session_state.pop("pending_case_decision", None)
                        st.session_state.pop("edit_case_id", None)
                        st.session_state.pop(action_key, None)
                        st.session_state.pop(comment_key, None)
                        st.rerun()
                with cancel_col:
                    if st.button("Abbrechen", key=f"cancel_save_{case_id}"):
                        st.session_state.pop("pending_case_decision", None)
                        st.rerun()

    with right_col:
        with st.expander("Abschnitt C: Beitrag einzelner Faktoren", expanded=True):
            contrib_plot_df = contrib_df.copy()
            contrib_plot_df["direction"] = contrib_plot_df["direction"].map(
                {"Risk-up": "Risikosteigernd", "Risk-down": "Risikosenkend"}
            )
            fig_contrib = px.bar(
                contrib_plot_df,
                x="points",
                y="factor",
                orientation="h",
                color="direction",
                color_discrete_map={"Risikosteigernd": "#d62728", "Risikosenkend": "#2ca02c"},
                labels={"points": "Beitragspunkte", "factor": "Faktor", "direction": "Beitragsart"},
                title="Faktorbeiträge für die Empfehlung",
                text="points",
            )
            fig_contrib.add_vline(x=0, line_width=1, line_dash="dash", line_color="#666666")
            st.plotly_chart(fig_contrib, width="stretch")

        with st.expander("Abschnitt D: Vergleich mit der Peer Group", expanded=False):
            render_peer_group_comparison(cases_df, case_row)

        with st.expander("Abschnitt E: Ähnliche Fälle", expanded=False):
            similar_df = build_similar_cases(cases_df, case_row, top_n=5)
            if len(similar_df) == 0:
                st.info("Keine ähnlichen Fälle im aktuellen Filterkontext.")
            else:
                st.dataframe(
                    similar_df[
                        [
                            "case_id",
                            "role",
                            "entitlement",
                            "application",
                            "recommendation_label",
                            "recommendation_score",
                            "confidence",
                            "similarity_reason",
                        ]
                    ],
                    hide_index=True,
                    width="stretch",
                )

        with st.expander("Abschnitt G: Zeitleiste / Zeitstrahl", expanded=False):
            render_timeline(case_row)


def load_evaluations(path: Path) -> pd.DataFrame:
    columns = [
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
    if not path.exists():
        return pd.DataFrame(columns=columns)
    return pd.read_csv(path)


def save_evaluation(entry: dict) -> None:
    eval_df = load_evaluations(EVALUATION_PATH)
    eval_df = pd.concat([eval_df, pd.DataFrame([entry])], ignore_index=True)
    eval_df.to_csv(EVALUATION_PATH, index=False)


def kpi_value(label: str, value: int) -> None:
    st.metric(label=label, value=f"{value:,}")


def explain_case(case_row: pd.Series) -> tuple[str, int, float, list[str], pd.DataFrame]:
    """Local explanation with score-based recommendation and reviewer-friendly factors."""
    positive: list[tuple[str, int]] = []
    negative: list[tuple[str, int]] = []

    last_login_days = int(case_row["last_login_days"])
    stale_access_days = int(case_row["stale_access_days"])
    privilege_level = str(case_row["privilege_level"])
    toxic_combo = int(case_row["toxic_combo"])
    role = str(case_row["role"])
    recommendation_label = str(case_row["recommendation"])
    recommendation_score = int(case_row["risk_score"])

    if last_login_days > 120:
        positive.append(("Inactive login > 120 days", 35))
    elif last_login_days > 60:
        positive.append(("Inactive login > 60 days", 20))
    elif last_login_days > 30:
        positive.append(("Inactive login > 30 days", 10))
    elif last_login_days <= 7:
        negative.append(("Very recent login (<=7 days)", -15))
    else:
        negative.append(("Recent login (<=30 days)", -10))

    if stale_access_days > 365:
        positive.append(("Stale access > 365 days", 30))
    elif stale_access_days > 180:
        positive.append(("Stale access > 180 days", 18))
    elif stale_access_days > 90:
        positive.append(("Stale access > 90 days", 8))
    elif stale_access_days <= 30:
        negative.append(("Access recently used (<=30 days)", -12))
    else:
        negative.append(("Access used in last quarter", -8))

    if privilege_level == "high":
        positive.append(("High privilege", 25))
    elif privilege_level == "medium":
        positive.append(("Medium privilege", 10))
    else:
        negative.append(("Low privilege", -10))

    if toxic_combo == 1:
        positive.append(("SoD conflict (toxic combination)", 40))
    else:
        negative.append(("No SoD conflict detected", -15))

    if role in {"Contractor", "Admin"}:
        positive.append(("Sensitive role", 10))
    else:
        negative.append(("Standard role profile", -5))

    dept_change = str(case_row.get("department_change_date", "")).strip()
    if dept_change:
        positive.append(("Department change (privilege creep risk)", 15))
    else:
        negative.append(("No department change", -5))

    user_status = str(case_row.get("user_status", "active")).strip()
    if user_status == "terminated":
        positive.append(("Terminated user (orphan account)", 30))
    elif user_status == "inactive":
        positive.append(("Inactive user", 15))
    else:
        negative.append(("Active user", -5))

    top_reason_items = sorted(positive, key=lambda x: x[1], reverse=True)[:3]
    if not top_reason_items:
        top_reason_items = [("No major risk driver; low-risk baseline", 0)]
    top_reasons = [f"{reason} ({points:+d})" for reason, points in top_reason_items]

    confidence = compute_confidence(recommendation_score, recommendation_label)

    contrib_rows = []
    for factor, points in positive + negative:
        contrib_rows.append(
            {
                "factor": factor,
                "points": points,
                "direction": "Risk-up" if points >= 0 else "Risk-down",
            }
        )
    contrib_df = pd.DataFrame(contrib_rows).sort_values("points", ascending=True)
    return recommendation_label, recommendation_score, confidence, top_reasons, contrib_df


def compute_confidence(score: int, recommendation: str) -> float:
    if recommendation == "revoke":
        margin = max(score - 70, 0)
    elif recommendation == "review":
        margin = max(min(score - 35, 70 - score), 0)
    else:
        margin = max(35 - score, 0)
    return min(99.0, 50.0 + (margin / 35.0) * 49.0)


def confidence_label(confidence: float) -> tuple[str, str]:
    """Return (label, color) for a confidence value."""
    if confidence >= 80.0:
        return "Hoch", "🟢"
    elif confidence >= 60.0:
        return "Mittel", "🟡"
    else:
        return "Unsicher – manuelle Prüfung empfohlen", "🔴"


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

    type_colors = {
        "Joiner": "#2ca02c",
        "Grant": "#1f77b4",
        "Mover": "#ff7f0e",
        "Recert": "#9467bd",
        "Leaver": "#d62728",
        "Login": "#8c564b",
    }
    events_df["color"] = events_df["type"].map(type_colors).fillna("#7f7f7f")
    events_df["y"] = 0

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=events_df["date"],
            y=events_df["y"],
            mode="markers+text",
            marker=dict(
                size=14,
                color=events_df["color"].tolist(),
                symbol="diamond",
                line=dict(width=1, color="#333"),
            ),
            text=events_df["type"],
            textposition="top center",
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Datum: %{x|%d.%m.%Y}<br>"
                "%{customdata[1]}<extra></extra>"
            ),
            customdata=list(zip(events_df["type"], events_df["description"])),
        )
    )
    fig.update_layout(
        title="Berechtigungshistorie (Zeitstrahl)",
        xaxis_title="Zeitachse",
        yaxis=dict(visible=False, range=[-0.5, 0.5]),
        height=200,
        margin=dict(l=40, r=40, t=50, b=40),
        showlegend=False,
    )
    fig.add_hline(y=0, line_width=1, line_color="#cccccc", line_dash="dot")
    st.plotly_chart(fig, use_container_width=True)


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
    st.sidebar.header("Global Filters")
    st.sidebar.caption("2) Narrow down with filters")
    st.sidebar.info(
        "Die Filter wirken auf alle Hauptansichten und helfen dabei, den Governance-Raum schrittweise einzugrenzen."
    )

    departments = sorted(scope_df["department"].dropna().astype(str).unique().tolist())
    applications = sorted(scope_df["application"].dropna().astype(str).unique().tolist())
    labels = ["retain", "review", "revoke"]

    min_score = int(scope_df["recommendation_score"].min()) if len(scope_df) else 0
    max_score = int(scope_df["recommendation_score"].max()) if len(scope_df) else 100
    min_conf = float(scope_df["confidence"].min()) if len(scope_df) else 50.0
    max_conf = float(scope_df["confidence"].max()) if len(scope_df) else 99.0

    return {
        "department": st.sidebar.multiselect("Department", departments, default=departments),
        "application": st.sidebar.multiselect("Anwendung", applications, default=applications),
        "recommendation_label": st.sidebar.multiselect(
            "Empfehlung", labels, default=labels
        ),
        "recommendation_score_range": st.sidebar.slider(
            "Empfehlungs-Score (Bereich)",
            min_value=min_score,
            max_value=max_score,
            value=(min_score, max_score),
        ),
        "confidence_range": st.sidebar.slider(
            "Konfidenz (Bereich)",
            min_value=float(min_conf),
            max_value=float(max_conf),
            value=(float(min_conf), float(max_conf)),
            step=0.1,
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


def extract_heatmap_selection(event: object) -> tuple[str, str] | None:
    selection = _safe_get(event, "selection")
    points = _safe_get(selection, "points")
    if not isinstance(points, (list, tuple)) or len(points) == 0:
        return None

    point = points[0]
    x_val = _safe_get(point, "x")
    y_val = _safe_get(point, "y")
    if x_val is not None and y_val is not None:
        return str(y_val), str(x_val)

    customdata = _safe_get(point, "customdata")
    if isinstance(customdata, (list, tuple)) and len(customdata) >= 2:
        row_from_cd, col_from_cd = customdata[0], customdata[1]
        if row_from_cd is not None and col_from_cd is not None:
            return str(row_from_cd), str(col_from_cd)
    return None


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


@st.dialog("Heatmap-Detailansicht")
def show_heatmap_cell_dialog(
    cases_df: pd.DataFrame,
    row_dim: str,
    col_dim: str,
    row_value: str,
    col_value: str,
    heatmap_mode: str,
    metric_name: str,
) -> None:
    cell_df = cases_df[
        (cases_df[row_dim].astype(str) == str(row_value))
        & (cases_df[col_dim].astype(str) == str(col_value))
    ].copy()

    if len(cell_df) == 0:
        st.warning("Für die ausgewählte Zelle sind im aktuellen Filterkontext keine Fälle vorhanden.")
        if st.button("Schließen", key="close_heatmap_dialog_empty"):
            clear_heatmap_selection_state()
            st.rerun()
        return

    metric_texts = {
        "Fallzahl": "Diese Zelle zeigt die Anzahl der Fälle in der ausgewählten Kombination.",
        "Durchschnittlicher Empfehlungs-Score": "Diese Zelle zeigt den durchschnittlichen Empfehlungs-Score aller Fälle in der ausgewählten Kombination.",
        "Kritische Fallrate": "Diese Zelle zeigt den Anteil der Fälle mit den Empfehlungen 'prüfen' oder 'entziehen' in der ausgewählten Kombination.",
        "SoD-Konfliktrate": "Diese Zelle zeigt den Anteil der Fälle mit einem Segregation-of-Duties-Konflikt in der ausgewählten Kombination.",
    }
    st.info(metric_texts.get(metric_name, ""))

    if heatmap_mode == "Department × Anwendung":
        st.caption(
            "Diese Detailansicht unterstützt die Priorisierung von Review-Aktivitäten auf Gruppenebene."
        )
        row_label, col_label = "Department", "Anwendung"
    else:
        st.caption(
            "Diese Detailansicht unterstützt die strukturelle Governance-Analyse von Rollen-Berechtigungs-Kombinationen."
        )
        row_label, col_label = "Rolle", "Berechtigung"

    st.write(f"**{row_label}:** `{row_value}`")
    st.write(f"**{col_label}:** `{col_value}`")

    metrics = compute_cell_metrics(cell_df)
    m_col_1, m_col_2, m_col_3 = st.columns(3)
    with m_col_1:
        st.metric("Fälle gesamt", metrics["total_cases"])
        st.metric("Anzahl retain", metrics["retain_cases"])
        st.metric("Anzahl review", metrics["review_cases"])
    with m_col_2:
        st.metric("Anzahl revoke", metrics["revoke_cases"])
        st.metric("Kritische Fallrate", f"{metrics['critical_rate']:.1%}")
        st.metric("Durchschnittlicher Empfehlungs-Score", f"{metrics['mean_score']:.1f}")
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
    detail_table["toxic_combo"] = detail_table["toxic_combo"].astype(int)
    detail_table = detail_table.sort_values("recommendation_score", ascending=False).rename(
        columns={
            "case_id": "fall_id",
            "user_id": "benutzer",
            "department": "department",
            "application": "anwendung",
            "role": "rolle",
            "entitlement": "berechtigung",
            "recommendation_label": "empfehlung",
            "recommendation_score": "empfehlungs_score",
            "confidence": "konfidenz",
            "toxic_combo": "sod_konflikt",
            "case_status": "fallstatus",
        }
    )
    detail_table["fallstatus"] = detail_table["fallstatus"].map(
        {"open": "offen", "decided": "entschieden", "escalated": "eskaliert"}
    )
    st.dataframe(detail_table, width="stretch", hide_index=True)

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

    candidates = candidates.sort_values(
        by=[
            "same_role",
            "same_entitlement",
            "same_application",
            "score_distance",
            "case_id",
        ],
        ascending=[False, False, False, True, True],
    )

    top = candidates.head(top_n).copy()

    def similarity_reason(row: pd.Series) -> str:
        tags = []
        if int(row["same_role"]) == 1:
            tags.append("same role")
        if int(row["same_entitlement"]) == 1:
            tags.append("same entitlement")
        if int(row["same_application"]) == 1:
            tags.append("same application")
        if not tags:
            tags.append("closest score profile")
        return f"{', '.join(tags)}; score distance={int(row['score_distance'])}"

    top["similarity_reason"] = top.apply(similarity_reason, axis=1)
    return top


def render_peer_group_comparison(cases_df: pd.DataFrame, case_row: pd.Series) -> None:
    peer_df = cases_df[
        (cases_df["case_id"] != case_row["case_id"])
        & (cases_df["role"] == case_row["role"])
        & (cases_df["application"] == case_row["application"])
    ].copy()
    if len(peer_df) == 0:
        peer_df = cases_df[
            (cases_df["case_id"] != case_row["case_id"]) & (cases_df["role"] == case_row["role"])
        ].copy()

    if len(peer_df) < 3:
        st.info("Für diese Kombination steht keine ausreichend große Peer Group zur Verfügung.")
        return

    selected_conf = compute_confidence(
        int(case_row["recommendation_score"]), str(case_row["recommendation_label"])
    )
    peer_conf = peer_df.apply(
        lambda r: compute_confidence(int(r["recommendation_score"]), str(r["recommendation_label"])),
        axis=1,
    ).mean()

    chart_df = pd.DataFrame(
        [
            {"metric": "Recommendation score", "group": "Selected case", "value": int(case_row["recommendation_score"])},
            {"metric": "Recommendation score", "group": "Peer average", "value": float(peer_df["recommendation_score"].mean())},
            {"metric": "Confidence", "group": "Selected case", "value": float(selected_conf)},
            {"metric": "Confidence", "group": "Peer average", "value": float(peer_conf)},
            {"metric": "Last login days", "group": "Selected case", "value": int(case_row["last_login_days"])},
            {"metric": "Last login days", "group": "Peer average", "value": float(peer_df["last_login_days"].mean())},
            {"metric": "Stale access days", "group": "Selected case", "value": int(case_row["stale_access_days"])},
            {"metric": "Stale access days", "group": "Peer average", "value": float(peer_df["stale_access_days"].mean())},
        ]
    )

    fig = px.bar(
        chart_df,
        x="value",
        y="metric",
        color="group",
        barmode="group",
        orientation="h",
        title="Selected Case vs Peer Group Average",
        labels={"value": "Value", "metric": "Metric"},
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"Peer-Group-Größe: {len(peer_df)} Fälle. "
        "Vergleichskriterien: gleiche funktionale Rolle + gleiche Anwendung "
        "(Fallback bei zu kleiner Gruppe: gleiche Rolle über alle Anwendungen)."
    )


def render_overview(cases_df: pd.DataFrame, decisions_df: pd.DataFrame) -> None:
    st.subheader("Globale Übersicht und Priorisierung")
    st.info(
        "Beginnen Sie mit der Übersicht, grenzen Sie die Daten mit Filtern ein und prüfen Sie anschließend einzelne Fälle im Detail."
    )
    st.caption(
        "Diese Ansicht zeigt aggregierte Muster über alle aktuell gefilterten Fälle. Sie unterstützt die Priorisierung von Review-Aktivitäten und hilft dabei, auffällige Bereiche frühzeitig zu erkennen."
    )

    if len(cases_df) == 0:
        st.warning("No cases match the current global filters.")
        return

    total_cases = len(cases_df)
    rec_counts = cases_df["recommendation_label"].value_counts()
    retain_count = int(rec_counts.get("retain", 0))
    review_count = int(rec_counts.get("review", 0))
    revoke_count = int(rec_counts.get("revoke", 0))
    sod_conflicts = int(cases_df["toxic_combo"].astype(int).sum())

    kpi_row = st.columns(5)
    with kpi_row[0]:
        kpi_value("Fälle gesamt", total_cases)
    with kpi_row[1]:
        kpi_value("Entziehen empfohlen", revoke_count)
    with kpi_row[2]:
        kpi_value("Prüfung empfohlen", review_count)
    with kpi_row[3]:
        kpi_value("Beibehalten empfohlen", retain_count)
    with kpi_row[4]:
        kpi_value("SoD-Konflikte", sod_conflicts)

    st.caption(
        "Nutzen Sie diese Kennzahlen und Verteilungen, um globale Muster zu erkennen, Prioritäten festzulegen und anschließend gezielt in die Fallprüfung zu wechseln."
    )

    overview_table = cases_df[
        [
            "case_id",
            "user_id",
            "department",
            "application",
            "entitlement",
            "recommendation_label",
            "recommendation_score",
            "case_status",
        ]
    ].rename(
        columns={
            "case_id": "fall_id",
            "user_id": "benutzer",
            "department": "abteilung",
            "application": "anwendung",
            "entitlement": "berechtigung",
            "recommendation_label": "empfehlung",
            "recommendation_score": "empfehlungs_score",
            "case_status": "fallstatus",
        }
    )
    overview_table["fallstatus"] = overview_table["fallstatus"].map(
        {"open": "offen", "decided": "entschieden", "escalated": "eskaliert"}
    )

    charts_col_1, charts_col_2 = st.columns(2)

    with charts_col_1:
        rec_dist = (
            cases_df["recommendation_label"]
            .value_counts()
            .reindex(["retain", "review", "revoke"], fill_value=0)
            .rename_axis("recommendation")
            .reset_index(name="count")
        )
        fig_rec = px.bar(
            rec_dist,
            x="recommendation",
            y="count",
            color="recommendation",
            title="Verteilung der Empfehlungen",
            labels={"recommendation": "Empfehlung", "count": "Anzahl Fälle"},
        )
        st.plotly_chart(fig_rec, width="stretch")

    with charts_col_2:
        fig_hist = px.histogram(
            cases_df,
            x="recommendation_score",
            nbins=16,
            title="Verteilung des Empfehlungs-Scores",
            labels={"recommendation_score": "Empfehlungs-Score", "count": "Anzahl Fälle"},
        )
        st.plotly_chart(fig_hist, width="stretch")

    charts_col_3, charts_col_4 = st.columns(2)
    with charts_col_3:
        dept_scores = (
            cases_df.groupby("department", dropna=False)["recommendation_score"]
            .mean()
            .reset_index(name="mean_score")
            .sort_values("mean_score", ascending=False)
        )
        fig_dept = px.bar(
            dept_scores,
            x="department",
            y="mean_score",
            title="Durchschnittlicher Score pro Department",
            labels={"department": "Department", "mean_score": "Durchschnittlicher Score"},
        )
        st.plotly_chart(fig_dept, width="stretch")

    with charts_col_4:
        app_scores = (
            cases_df.groupby("application", dropna=False)["recommendation_score"]
            .mean()
            .reset_index(name="mean_score")
            .sort_values("mean_score", ascending=False)
        )
        fig_app = px.bar(
            app_scores,
            x="application",
            y="mean_score",
            title="Durchschnittlicher Score pro Anwendung",
            labels={"application": "Anwendung", "mean_score": "Durchschnittlicher Score"},
        )
        st.plotly_chart(fig_app, width="stretch")

    st.markdown("### Review-Fälle im aktuellen Filterkontext")
    st.caption(
        "Die Tabelle zeigt alle aktuell gefilterten Fälle. Nutzen Sie die Übersicht zur Priorisierung und wechseln Sie anschließend in die Fallansicht, um einzelne Fälle im Detail zu prüfen."
    )
    st.dataframe(overview_table, width="stretch", hide_index=True)


def render_heatmap(cases_df: pd.DataFrame) -> None:
    st.subheader("Struktur")
    if len(cases_df) == 0:
        st.warning("No cases match the current global filters.")
        clear_heatmap_selection_state()
        return

    heatmap_mode = st.radio(
        "Heatmap-Typ",
        options=["Department × Anwendung", "Rolle × Berechtigung"],
        horizontal=True,
    )
    if heatmap_mode == "Department × Anwendung":
        st.caption(
            "Diese Heatmap zeigt aggregierte Muster auf Gruppenebene und unterstützt die Priorisierung von Review-Aktivitäten über Departments und Anwendungen."
        )
        row_dim, col_dim = "department", "application"
        y_title, x_title = "Department", "Anwendung"
        default_metric = "Kritische Fallrate"
    else:
        st.caption(
            "Diese Heatmap zeigt strukturelle Governance-Muster zwischen Rollen und Berechtigungen und macht auffällige oder kritische Kombinationen sichtbar."
        )
        row_dim, col_dim = "role", "entitlement"
        y_title, x_title = "Rolle", "Berechtigung"
        default_metric = "Durchschnittlicher Empfehlungs-Score"

    metric_options = [
        "Fallzahl",
        "Durchschnittlicher Empfehlungs-Score",
        "Kritische Fallrate",
        "SoD-Konfliktrate",
    ]
    selected_metric = st.selectbox(
        "Metrik",
        options=metric_options,
        index=metric_options.index(default_metric),
    )
    st.caption("Klicken Sie auf eine Zelle, um die Zusammensetzung im Detail zu sehen.")

    agg_df = (
        cases_df.groupby([row_dim, col_dim], dropna=False)
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
        "Fallzahl": ("total_cases", "Fallzahl", ".0f"),
        "Durchschnittlicher Empfehlungs-Score": (
            "mean_score",
            "Durchschnittlicher Empfehlungs-Score",
            ".1f",
        ),
        "Kritische Fallrate": ("critical_case_rate", "Kritische Fallrate", ".1%"),
        "SoD-Konfliktrate": ("sod_conflict_rate", "SoD-Konfliktrate", ".1%"),
    }
    metric_col, color_title, z_format = metric_map[selected_metric]

    z_matrix = agg_df.pivot(index=row_dim, columns=col_dim, values=metric_col).fillna(0.0)

    row_labels = z_matrix.index.astype(str).tolist()
    col_labels = z_matrix.columns.astype(str).tolist()

    detail_idx = agg_df.set_index([row_dim, col_dim])
    customdata = []
    for row_value in z_matrix.index:
        row_cells = []
        for col_value in z_matrix.columns:
            if (row_value, col_value) in detail_idx.index:
                d = detail_idx.loc[(row_value, col_value)]
                row_cells.append(
                    [
                        str(row_value),
                        str(col_value),
                        int(d["total_cases"]),
                        int(d["revoke_cases"]),
                        int(d["review_cases"]),
                        float(d["mean_score"]),
                        float(d["critical_case_rate"]),
                        float(d["sod_conflict_rate"]),
                    ]
                )
            else:
                row_cells.append(
                    [str(row_value), str(col_value), 0, 0, 0, 0.0, 0.0, 0.0]
                )
        customdata.append(row_cells)

    hovertemplate = (
        f"{y_title}: %{{y}}<br>"
        f"{x_title}: %{{x}}<br>"
        "Fälle gesamt: %{customdata[2]}<br>"
        "Anzahl revoke: %{customdata[3]}<br>"
        "Anzahl review: %{customdata[4]}<br>"
        "Durchschnittlicher Empfehlungs-Score: %{customdata[5]:.1f}<br>"
        "Kritische Fallrate: %{customdata[6]:.1%}<br>"
        "SoD-Konfliktrate: %{customdata[7]:.1%}<extra></extra>"
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=z_matrix.values,
            x=col_labels,
            y=row_labels,
            text=z_matrix.values,
            colorscale="Blues",
            colorbar={"title": color_title},
            customdata=customdata,
            hovertemplate=hovertemplate,
            zmin=0,
        )
    )
    fig.update_layout(
        title=f"{heatmap_mode} ({selected_metric})",
        xaxis_title=x_title,
        yaxis_title=y_title,
    )
    fig.update_traces(texttemplate=f"%{{z:{z_format}}}")
    selection_event = st.plotly_chart(
        fig,
        width="stretch",
        on_select="rerun",
        key=f"heatmap_{heatmap_mode}_{selected_metric}",
    )

    selected_cell = extract_heatmap_selection(selection_event)
    if selected_cell is not None:
        selected_row, selected_col = selected_cell
        st.session_state["heatmap_selected_cell"] = {
            "row_dim": row_dim,
            "col_dim": col_dim,
            "row_value": selected_row,
            "col_value": selected_col,
            "heatmap_mode": heatmap_mode,
            "metric": selected_metric,
        }
        st.session_state["heatmap_dialog_open"] = True

    active_selection = st.session_state.get("heatmap_selected_cell")
    dialog_open = bool(st.session_state.get("heatmap_dialog_open", False))

    if not dialog_open or not isinstance(active_selection, dict):
        return

    if (
        active_selection.get("row_dim") != row_dim
        or active_selection.get("col_dim") != col_dim
        or active_selection.get("heatmap_mode") != heatmap_mode
    ):
        clear_heatmap_selection_state()
        return

    row_value = str(active_selection.get("row_value", ""))
    col_value = str(active_selection.get("col_value", ""))
    if row_value == "" or col_value == "":
        clear_heatmap_selection_state()
        return

    valid_rows = set(cases_df[row_dim].astype(str).unique().tolist())
    valid_cols = set(cases_df[col_dim].astype(str).unique().tolist())
    if row_value not in valid_rows or col_value not in valid_cols:
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
        "Details on demand: Wählen Sie einen Fall aus der Tabelle und öffnen Sie danach den Bearbeitungsdialog."
    )

    if len(cases_df) == 0:
        st.warning("Keine Fälle im aktuellen Filterkontext.")
        st.session_state.pop("edit_case_id", None)
        st.session_state.pop("pending_case_decision", None)
        return

    feedback = st.session_state.pop("decision_feedback", None)
    if isinstance(feedback, dict):
        if feedback.get("type") == "warning":
            st.warning(str(feedback.get("text", "")))
        else:
            st.success(str(feedback.get("text", "")))

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
    table_df["case_status"] = table_df["case_status"].map(
        {"open": "offen", "decided": "entschieden", "escalated": "eskaliert"}
    )
    table_df = table_df.sort_values("recommendation_score", ascending=False).reset_index(drop=True)

    selection_event = st.dataframe(
        table_df,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="case_review_table",
    )

    selected_case_id = None
    selection = _safe_get(selection_event, "selection")
    selected_rows = _safe_get(selection, "rows")
    if isinstance(selected_rows, (list, tuple)) and len(selected_rows) == 1:
        row_idx = selected_rows[0]
        if isinstance(row_idx, int) and 0 <= row_idx < len(table_df):
            selected_case_id = str(table_df.iloc[row_idx]["case_id"])
    elif isinstance(selected_rows, (list, tuple)) and len(selected_rows) > 1:
        st.warning("Bitte wählen Sie genau eine Zeile aus.")
        st.session_state.pop("selected_case_id", None)
        return

    if selected_case_id is None:
        st.info("Bitte wählen Sie einen Fall aus der Tabelle aus.")
        st.session_state.pop("selected_case_id", None)
        st.session_state.pop("edit_case_id", None)
        return

    st.session_state["selected_case_id"] = selected_case_id
    selected_case = cases_df.loc[cases_df["case_id"].astype(str) == selected_case_id].iloc[0]

    st.markdown("#### Ausgewählter Fall")
    st.write(f"- Fall-ID: `{selected_case['case_id']}`")
    st.write(f"- Benutzer: `{selected_case['user_id']}`")
    st.write(f"- Anwendung: `{selected_case['application']}`")
    st.write(f"- Berechtigung: `{selected_case['entitlement']}`")
    st.write(f"- Empfehlung: `{selected_case['recommendation_label']}`")

    if st.button("Fall bearbeiten", type="primary", key=f"edit_case_btn_{selected_case_id}"):
        st.session_state["edit_case_id"] = selected_case_id

    if st.session_state.get("edit_case_id") == selected_case_id:
        render_case_edit_dialog(cases_df, decisions_df, selected_case_id)


def render_audit_log(
    decisions_df: pd.DataFrame, cases_df: pd.DataFrame, filtered_case_ids: set[str]
) -> None:
    st.subheader("Audit Log")
    st.caption("Decision history for the currently filtered case scope.")
    filtered_log = decisions_df[decisions_df["case_id"].astype(str).isin(filtered_case_ids)].copy()
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
            "comment",
            "timestamp",
        ]
    ].sort_values("timestamp", ascending=False)
    st.dataframe(merged_log, width="stretch", hide_index=True)


def render_evaluation_mode() -> None:
    st.subheader("Evaluation Mode")
    st.caption("Mini-Usability-Evaluation mit 3 Aufgaben und 5 Likert-Fragen (1-5).")

    eval_df = load_evaluations(EVALUATION_PATH)
    st.write(f"Bisherige Evaluationseinträge: **{len(eval_df)}**")

    participant_id = st.text_input("Participant ID", value="P-001")

    st.write("**Aufgaben (erledigt?)**")
    task_1 = st.checkbox("Aufgabe 1: Öffne einen Fall und prüfe die lokale Erklärung.")
    task_2 = st.checkbox("Aufgabe 2: Treffe eine Entscheidung (Confirm oder Override) mit Kommentar.")
    task_3 = st.checkbox("Aufgabe 3: Nutze die Heatmap, um ein Muster zu identifizieren.")

    st.write("**Likert-Fragen (1 = stimme gar nicht zu, 5 = stimme voll zu)**")
    q1 = st.radio(
        "1) Die Empfehlungen waren nachvollziehbar.",
        options=[1, 2, 3, 4, 5],
        horizontal=True,
        key="eval_q1",
    )
    q2 = st.radio(
        "2) Die lokale Erklärung war hilfreich für die Entscheidung.",
        options=[1, 2, 3, 4, 5],
        horizontal=True,
        key="eval_q2",
    )
    q3 = st.radio(
        "3) Die Übersicht/KPIs waren klar verständlich.",
        options=[1, 2, 3, 4, 5],
        horizontal=True,
        key="eval_q3",
    )
    q4 = st.radio(
        "4) Die Bearbeitung eines Falls war effizient.",
        options=[1, 2, 3, 4, 5],
        horizontal=True,
        key="eval_q4",
    )
    q5 = st.radio(
        "5) Ich würde den Prototypen für Access Reviews nutzen.",
        options=[1, 2, 3, 4, 5],
        horizontal=True,
        key="eval_q5",
    )

    eval_comment = st.text_area("Optionaler Kommentar", max_chars=500, key="eval_comment")

    if st.button("Save Evaluation", key="save_evaluation_btn"):
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


def main() -> None:
    st.set_page_config(page_title="Mini IGA Access Review", layout="wide")
    st.markdown(
        """
        <style>
        div[data-testid="stDialog"] div[role="dialog"],
        section[data-testid="stDialog"] div[role="dialog"],
        div[role="dialog"][aria-modal="true"] {
            width: min(96vw, 1800px) !important;
            max-width: 1800px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Mini IGA Access Review Prototype")
    st.info(
        "Start with the overview, narrow down with filters, inspect a selected case in detail."
    )

    if not CASES_PATH.exists():
        st.error("No review data found. Run `python generate_data.py` first.")
        return

    cases_df = load_cases(CASES_PATH)
    decisions_df = load_decisions(DECISIONS_PATH)
    scope_df = build_case_scope(cases_df, decisions_df)
    active_filters = render_global_filters(scope_df)
    filtered_scope_df = apply_global_filters(scope_df, active_filters)
    filtered_case_ids = set(filtered_scope_df["case_id"].astype(str).tolist())
    filtered_decisions_df = decisions_df[
        decisions_df["case_id"].astype(str).isin(filtered_case_ids)
    ]

    tab_overview, tab_structure, tab_case_review, tab_audit = st.tabs(
        ["Übersicht", "Struktur", "Fallprüfung", "Audit Log"]
    )
    with tab_overview:
        render_overview(filtered_scope_df, filtered_decisions_df)
    with tab_structure:
        render_heatmap(filtered_scope_df)
    with tab_case_review:
        render_case_view(filtered_scope_df, decisions_df)
    with tab_audit:
        render_audit_log(decisions_df, filtered_scope_df, filtered_case_ids)

    st.divider()
    render_evaluation_mode()


if __name__ == "__main__":
    main()
