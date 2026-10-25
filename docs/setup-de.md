# BatteryML

Workflow update and dependency behavior is documented in
[`docs/WORKFLOW_FRESHNESS.md`](docs/WORKFLOW_FRESHNESS.md).

The guided capacity and BMS SOH investigation, its evidence limits, artifacts,
and operator workflow are documented in
[`docs/CAPACITY_SOH_INVESTIGATION.md`](docs/CAPACITY_SOH_INVESTIGATION.md).

Phase 4A single-device SOH-gap diagnosis, candidate reconciliation, uncertainty,
and the dedicated frontend workflow are documented in
[`docs/PHASE4A_SOH_GAP_DIAGNOSIS.md`](docs/PHASE4A_SOH_GAP_DIAGNOSIS.md).

The offline, cached scientific data gate and its purpose-specific readiness
rules are documented in [`docs/DATA_VALIDATION.md`](docs/DATA_VALIDATION.md).

BatteryML ist eine lokale Plattform zur Analyse von Batteriespeichern. Sie
besteht aus zwei zusammengehörigen Projektordnern:

- `BatteryML`: FastAPI-Backend, React-Frontend, Workflow-Steuerung und Katalog
- `Veera_SOH`: Datenextraktion, Vorverarbeitung, Merkmalsbildung und Modelle

## Übergabe auf einen anderen Rechner

Die vollständige deutsche Installations- und Übergabeanleitung steht in:

[docs/architecture.md](docs/architecture.md)

Die englische Version steht in:

[docs/architecture.md](docs/architecture.md)

Wichtig: Nach dem Entpacken müssen die alten virtuellen Umgebungen und lokalen
Abhängigkeiten gelöscht und auf dem neuen Rechner neu erstellt werden:

```text
BatteryML/.venv/
BatteryML/frontend/node_modules/
BatteryML/frontend/dist/
Veera_SOH/venv/
```

Der Ordner `Veera_SOH/data/` darf nicht gelöscht werden. Er enthält die Roh- und
Analyseergebnisse.

Die ältere ausführliche technische Dokumentation ist in
[README_TECHNISCH.md](README_TECHNISCH.md) erhalten.
