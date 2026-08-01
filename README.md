# Incremental HILS Anomaly Detection

**Incremental Learning Based Anomaly Detection for Hardware-in-the-Loop Simulation (HILS)**

A fully offline system that detects anomalies in multi-rate HILS data and identifies the most likely root-cause subsystem. Designed for air-gapped environments and technical demonstration.

---

## Key Features

- **Multi-rate data alignment** – Handles subsystems operating at different sampling frequencies
- **Incremental / Online learning** – Continuously updates the model with every new sample
- **Hybrid anomaly detection** – Combines Half-Space Trees with statistical residual analysis
- **Adaptive thresholding** – Threshold adjusts automatically based on recent data
- **Subsystem-level root-cause analysis** – Maps anomalies back to the originating subsystem
- **Mission-control style dashboard** – Professional Streamlit GUI with RCI branding
- **Streaming console demo** – Real-time style demonstration of online detection
- **Fully offline** – No external API calls or internet dependency

---

## Current Performance (Synthetic Data)

| Metric                    | Result     |
|--------------------------|------------|
| Detection Rate           | High       |
| Root-Cause Identification| Working    |
| Anomaly Rate (typical)   | ~0.5–0.8%  |
| Operating Mode           | Online / Incremental |

> Note: Results are based on synthetic multi-rate HILS data. Performance on real HILS data may vary.

---

## System Architecture (Simplified)
Multi-rate HILS Streams
↓
Alignment (→ 10 ms)
↓
Incremental Detector (Hybrid)
↓
Adaptive Threshold
↓
Root-Cause Analyzer → Subsystem
↓
Dashboard + Reports


**Subsystems currently modelled:**
- IMU
- Actuator
- Flight Motion Simulator (FMS)
- Guidance
- Telemetry

---

## Project Structure
incremental/
├── assets/                  # Logos and static assets
├── configs/                 # Configuration files
├── data/
│   └── sample/              # Synthetic multi-rate data
├── docs/
├── scripts/
│   ├── generate_sample_data.py
│   ├── evaluate_detection.py
│   ├── stream_demo.py
│   └── test_autoencoder.py
├── src/
│   └── incremental_hils/
│       ├── anomaly_detection/
│       ├── root_cause/
│       ├── synchronization/
│       ├── self_supervised/
│       └── visualization/
│           └── app.py       # Streamlit Mission Control Dashboard
├── outputs/
├── pyproject.toml
├── requirements.txt
└── README.md

## Quick Start

### 1. Create and activate virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
2. Install dependencies
Bashpip install -r requirements.txt
pip install plotly          # Required for the dashboard
3. Generate sample data (if not already present)
Bashpython scripts/generate_sample_data.py
4. Launch the Mission Control Dashboard
Bashstreamlit run src/incremental_hils/visualization/app.py
5. Run the Streaming Console Demo
Bashpython scripts/stream_demo.py

Dashboard Features

Overall system status (System Healthy / Anomaly Detected)
Anomaly score + adaptive threshold visualization
Root-cause distribution chart
Per-subsystem health indicators
Detailed event log with CSV download
Confidence distribution
RCI branding


Configuration Options (Dashboard Sidebar)

























ParameterDescriptionDefaultWarm-up samplesSamples used only for model adaptation800Threshold percentileControls detection sensitivity99.0Min confidenceMinimum confidence to display an anomaly1.8

Current Status
MVP / Working Baseline (August 2026)
Completed

Multi-rate alignment
Incremental hybrid detector
Adaptive thresholding
Subsystem root-cause analysis
Streamlit Mission Control GUI
Streaming console demonstration
Synthetic data pipeline

Planned / Future Work

Integration with real HILS data
Improved continual learning strategies
Formal evaluation metrics on real fault cases
Production hardening and packaging
Optional early-warning / predictive capability


Design Principles

Fully offline / air-gapped capable
Low external dependencies
Transparent and auditable decisions
Suitable for technical demonstration and further research


License
MIT

Research Centre Imarat (RCI)
Incremental Learning Based Anomaly Detection – HILS
text---

After updating the file, you can view it with:

```bash
cat README.md

Architecture
+-----------------------------------------------------------------------+
|                     HILS Anomaly Detection System                     |
|                     (Offline / Air-gapped Capable)                     |
+-----------------------------------------------------------------------+
|                                                                       |
|   +---------------------+                                             |
|   |  Multi-rate HILS    |                                             |
|   |  Data Streams       |                                             |
|   |                     |                                             |
|   |  - IMU              |                                             |
|   |  - Actuator         |                                             |
|   |  - FMS              |                                             |
|   |  - Guidance         |                                             |
|   |  - Telemetry        |                                             |
|   +----------+----------+                                             |
|              |                                                        |
|              v                                                        |
|   +---------------------+                                             |
|   |  Synchronization    |                                             |
|   |  & Alignment Layer  |  (Resample to common 10 ms timeline)        |
|   +----------+----------+                                             |
|              |                                                        |
|              v                                                        |
|   +---------------------------------------------------+               |
|   |           Incremental Detection Engine            |               |
|   |                                                   |               |
|   |  +--------------------+   +--------------------+  |               |
|   |  | Half-Space Trees   | + | Statistical        |  |               |
|   |  | (Online Learning)  |   | Residuals (Z-score)|  |               |
|   |  +--------------------+   +--------------------+  |               |
|   |                     |                             |               |
|   |                     v                             |               |
|   |            Combined Anomaly Score                 |               |
|   +----------------------+----------------------------+               |
|                          |                                            |
|                          v                                            |
|   +---------------------+                                             |
|   | Adaptive Threshold  |  (Rolling percentile based)                 |
|   +----------+----------+                                             |
|              |                                                        |
|              | (if score > threshold)                                 |
|              v                                                        |
|   +---------------------+                                             |
|   | Root-Cause Analyzer |                                             |
|   |                     |                                             |
|   | Feature → Subsystem |                                             |
|   | Aggregation +       |                                             |
|   | Confidence Scoring   |                                             |
|   +----------+----------+                                             |
|              |                                                        |
|              v                                                        |
|   +---------------------------------------------------+               |
|   |              Presentation Layer                   |               |
|   |                                                   |               |
|   |  +-------------------+   +---------------------+  |               |
|   |  | Streamlit Mission |   | Streaming Console   |  |               |
|   |  | Control Dashboard |   | Demo                |  |               |
|   |  +-------------------+   +---------------------+  |               |
|   |                                                   |               |
|   |  - Status Banner                                  |               |
|   |  - Score + Threshold Graph                        |               |
|   |  - Root-Cause Distribution                        |               |
|   |  - Subsystem Health Gauges                        |               |
|   |  - Event Log + CSV Export                         |               |
|   +---------------------------------------------------+               |
|                                                                       |
+-----------------------------------------------------------------------+