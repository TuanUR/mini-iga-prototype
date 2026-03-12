# Mini IGA Prototype (Streamlit)

Minimaler Masterarbeits-Prototyp für **Identity Governance and Administration (IGA)** mit Fokus auf **Access Review / Rezertifizierung** und rein lokalen, synthetischen Daten.

## Features

- Generierung synthetischer Review-Fälle
- Transparente, gewichtete Regel-Logik für Empfehlungen:
  - `retain`
  - `review`
  - `revoke`
- KPI-Overview und Verteilung der Empfehlungen
- Heatmap:
  - `Department x Application` oder
  - `Role x Entitlement`
- Fallansicht mit lokaler Erklärung (Score + Regelbeiträge)
- Entscheidungs-Workflow: Confirm / Override / Comment
- Audit Log als CSV (`data/decisions.csv`)

## Projektstruktur

- `requirements.txt`
- `generate_data.py`
- `app.py`
- `data/review_cases.csv`
- `data/decisions.csv`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Daten generieren

```bash
python generate_data.py
```

## App starten

```bash
streamlit run app.py
```

Danach im Browser die angezeigte lokale URL öffnen.

## Regel-Logik (kurz)

Die Empfehlung entsteht über additive Gewichte, u. a. für:

- Inaktivität (`last_login_days`)
- veraltete Berechtigung (`stale_access_days`)
- Privilegstufe (`privilege_level`)
- toxische Kombination (`toxic_combo`)
- sensible Rollen (`Admin`, `Contractor`)

Schwellenwerte:

- `score < 35` -> `retain`
- `35 <= score < 70` -> `review`
- `score >= 70` -> `revoke`

## Hinweise

- Keine externe Datenbank
- Keine API-Anbindung
- Alles lokal lauffähig
