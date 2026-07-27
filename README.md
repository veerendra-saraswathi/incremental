# Incremental HILS Anomaly Detection

**Incremental Learning Based Anomaly Detection for Hardware-in-the-Loop Simulation (HILS)**

This project detects anomalies in multi-rate HILS data and identifies the most likely root-cause subsystem. It supports both offline evaluation and online/streaming style processing.

---

## Key Features

- Multi-rate data alignment (different subsystem frequencies)
- Incremental / online anomaly detection
- Hybrid detector (statistical residuals + Half-Space Trees)
- Self-supervised Autoencoder (baseline added)
- Root-cause analysis at subsystem level
- Ground-truth evaluation
- Streamlit GUI for interactive demo
- Automatic saving of results and summary reports

---

## Current Performance (Synthetic Data)

| Metric                  | Result |
|-------------------------|--------|
| Detection Rate          | 100%   |
| Root-Cause Accuracy     | 100%   |
| Ground-truth intervals  | 4/4    |

---

## Project Structure

---

## Quick Start

```bash
# 1. Create and activate virtual environment (Python 3.11 recommended)
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate sample data
python scripts/generate_sample_data.py

# 4. Run evaluation (detection + root-cause + save results)
python scripts/evaluate_detection.py

# 5. Launch GUI
streamlit run src/incremental_hils/visualization/app.py
Main Scripts





























ScriptPurposegenerate_sample_data.pyCreate synthetic multi-rate HILS dataevaluate_detection.pyFull evaluation + save reportsstream_demo.pySimple streaming demotest_autoencoder.pyTest self-supervised autoencodervisualization/app.pyStreamlit GUI

Output Files
After running evaluation, results are saved in:
textoutputs/reports/
├── hils_anomalies_YYYYMMDD_HHMMSS.csv
├── hils_anomalies_YYYYMMDD_HHMMSS.json
└── hils_summary_YYYYMMDD_HHMMSS.json

Requirements

Python 3.11 or 3.12 (recommended)
See requirements.txt


Status
MVP / Working Baseline (July 2026)

 Multi-rate alignment
 Incremental anomaly detection
 Root-cause identification
 Ground-truth evaluation
 Result reporting
 Streamlit GUI
 Self-supervised autoencoder (baseline)
 Real HILS data integration
 More advanced continual learning methods
 Production hardening


License
MIT
text---

After you replace the README, the project is in good shape.

