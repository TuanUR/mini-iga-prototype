# VA-IGA: Visual Analytics für Berechtigungsrezertifizierung

Prototyp zur Masterarbeit *„Visual Analytics für Identity and Access Management im Zeitalter von KI – State of the Art und Weiterentwicklung des Status Quo"*

---

## Übersicht

Dieses Repository enthält einen Streamlit-basierten Prototyp, der Visual-Analytics-Prinzipien auf den Use Case der **Zugangsrezertifizierung** (Access Certification) im Identity and Access Management (IAM) anwendet. Ein regelbasiertes Scoring-Modell generiert KI-Empfehlungen für 240 synthetische Rezertifizierungsfälle; ein interaktives Dashboard unterstützt Reviewer bei der Prüfung, Erklärung und Dokumentation ihrer Entscheidungen.

### Vier-Sichten-Architektur

| Sicht | Tab | Funktion |
|-------|-----|----------|
| **Sicht 1** | Übersicht | KPIs, Verteilungsdiagramme, filterbare Falltabelle |
| **Sicht 2** | Struktur | Heatmap (Abteilung × Anwendung / Rolle × Berechtigung) mit Drill-Down |
| **Sicht 3** | Fallprüfung | XAI-Faktordiagramm, Peer-Group-Vergleich, Berechtigungshistorie, Entscheidungsworkflow |
| **Sicht 4** | Audit Log | Vollständige Entscheidungshistorie mit Versioning, Override-Rate, CSV-Export |

Ein fünfter Tab (**Evaluation**) enthält einen integrierten Fragebogen mit Reflexionsfragen und Likert-Items für die formative Evaluation.

---

## Projektstruktur

```
├── app.py                  # Hauptanwendung (Streamlit)
├── scoring.py              # Regelbasiertes Scoring-Modell (7 Faktorkategorien)
├── generate_data.py        # Synthetische Datengenerierung (240 Fälle)
├── data/
│   ├── review_cases.csv    # Falldatensatz (generiert durch generate_data.py)
│   ├── decisions.csv       # Entscheidungsprotokoll (Audit Trail)
│   └── evaluation_log.csv  # Evaluationsergebnisse
├── requirements.txt        # Python-Abhängigkeiten
└── README.md
```

---

## Lokale Installation

### Voraussetzungen

- Python 3.10+
- pip

### Setup

```bash
# Repository klonen
git clone https://github.com/<user>/<repo>.git
cd <repo>

# Abhängigkeiten installieren
pip install -r requirements.txt

# Synthetische Daten generieren
python generate_data.py

# Prototyp starten
streamlit run app.py
```

Die Anwendung öffnet sich unter `http://localhost:8501`.

---

## Deployment (Streamlit Community Cloud)

Die Anwendung ist für das Deployment auf [Streamlit Community Cloud](https://share.streamlit.io/) konfiguriert.

### Google Sheets Persistenz

Auf Streamlit Community Cloud ist das Dateisystem ephemer. Entscheidungen und Evaluationen werden daher in einem Google Sheet persistiert. Für lokale Nutzung ist dies optional – die App fällt automatisch auf lokale CSV-Dateien zurück.

**Einrichtung:**

1. Google Cloud Projekt mit aktivierter **Sheets API** und **Drive API** erstellen
2. Service Account mit JSON-Key erstellen
3. Google Sheet anlegen und mit der Service-Account-E-Mail als Editor teilen
4. In Streamlit Cloud → App Settings → Secrets eintragen:

```toml
[gcp_service_account]
type = "service_account"
project_id = "..."
private_key_id = "..."
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "...@....iam.gserviceaccount.com"
client_id = "..."
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"

[google_sheets]
spreadsheet_url = "https://docs.google.com/spreadsheets/d/SHEET_ID/edit"
```

Die Tabellenblätter `decisions` und `evaluation_log` werden automatisch erstellt.

---

## Scoring-Modell

Das regelbasierte Scoring-Modell (`scoring.py`) berechnet einen Risiko-Score aus sieben Faktorkategorien:

| Kategorie | Risikoerhöhend | Risikosenkend |
|-----------|---------------|---------------|
| Login-Aktivität | Inaktiv > 120 Tage: +35 | Aktiv ≤ 7 Tage: −15 |
| Zuweisungsaktualität | Unbenutzt > 365 Tage: +30 | Kürzlich genutzt: −12 |
| Privilegstufe | Hoch: +25 | Niedrig: −10 |
| SoD-Konflikt | Toxic Combination: +40 | Kein Konflikt: −15 |
| Rolle | Admin/Contractor: +10 | — |
| Abteilungswechsel | Privilege Creep: +15 | Kein Wechsel: −5 |
| Benutzerstatus | Terminated: +30 | Aktiv: −5 |

**Empfehlungsschwellen:** Score ≥ 70 → Entziehen · 35–69 → Prüfen · < 35 → Beibehalten

---

## Evaluationskonzept

Die Evaluation folgt dem FEDS-Framework (Venable et al., 2016) als **formative, artifizielle** Evaluations-Episode:

- **Methodik:** Asynchroner Self-Guided Expert Walkthrough
- **Teilnehmer:** 2–3 Domänenexperten (Convenience Sampling)
- **Instrumente:** 3 zielorientierte Aufgaben, strukturierte Reflexionsfragen, 5 Likert-Items, offene Kommentare
- **Persistenz:** Alle Eingaben werden pro Teilnehmer-ID im Prototyp gespeichert

### Teilnehmer-ID

Beim Start muss eine Teilnehmer-ID im Format `P-XXX` (z. B. P-001) eingegeben und bestätigt werden. Die ID wird für die gesamte Sitzung gesperrt und dient als Zuordnungsschlüssel für Entscheidungen und Evaluationsdaten.

---

## Audit Trail

Jede Entscheidung wird als neue Zeile protokolliert – auch Revisionen. Das Audit Log zeigt die vollständige Versionshistorie mit:

- **Version** (v1, v2, ...) mit Status (aktuell / revidiert)
- **Reviewer** (Teilnehmer-ID)
- **Zeitstempel**, KI-Empfehlung, Aktion, finale Entscheidung, Kommentar
- **Override-Markierung** bei Abweichung von der KI-Empfehlung

---

## Design Requirements

| DR | Beschreibung | Umsetzung |
|----|-------------|-----------|
| DR1 | Shneiderman-konformer Informationszugang | Übersicht → Heatmap → Fallprüfung + globale Filter |
| DR2 | Integration von ML-Empfehlungen | KI-Empfehlung mit Score, Konfidenz und Entscheidungsworkflow |
| DR3 | Post-hoc-Erklärbarkeit | Divergierendes Faktordiagramm (SHAP-artig) |
| DR4 | Kontextualisierung | Peer-Group-Vergleich + Berechtigungshistorie (Zeitstrahl) |
| DR5 | Auditierbare Entscheidungsdokumentation | Audit Trail mit Versioning und CSV-Export |

---

## Technologie-Stack

- **Frontend:** [Streamlit](https://streamlit.io/)
- **Visualisierung:** [Plotly](https://plotly.com/python/) (Express + Graph Objects)
- **Datenverarbeitung:** [pandas](https://pandas.pydata.org/)
- **Persistenz:** CSV (lokal) / [Google Sheets](https://developers.google.com/sheets/api) (Cloud)
- **Authentifizierung:** [gspread](https://docs.gspread.org/) + Google Service Account

---

## Lizenz

Dieses Projekt wurde im Rahmen einer Masterarbeit an der Universität Regensburg erstellt und dient ausschließlich akademischen Zwecken.
