# SANNIF – Semantic AI Network Intrusion Framework

SANNIF is a premium, Apple-inspired network intrusion detection and threat intelligence dashboard powered by a production-trained XGBoost model. It features a clean, minimal, and highly functional interface designed for cybersecurity analysts and SOC (Security Operations Center) demonstrations.

---

## Key Features

- **Apple Human Interface Design**: High-contrast, light-themed premium cards, layout grid, subtle borders, and smooth slide-up fade-in transitions.
- **Robust Schema Validation**: Automatically extracts expected features from the loaded model metadata and verifies uploaded CSV headers to prevent runtime crashes.
- **System Health Monitor**: Sidebar-mounted live check reporting model deserialization, encoder loading, engine mode, and required features verification state.
- **Detailed Threat Analytics**: Interactive Plotly diagrams representing threat classification shares, category frequencies, confidence distribution histograms, and top flagged vectors.
- **Auto-Archiving Report Engine**: Exports predictions and automatically archives unified records (inputs + outputs) on the server in the `reports/` folder with timestamped filenames.
- **SOC Diagnostics Console**: Diagnostic tab capturing environment parameters (memory footprint, library versions, model features matrix, backend execution latency) for rapid debugging.
- **Demonstration Mode Fallback**: If the production XGBoost DLL is blocked by Windows AppLocker/WDAC policies, SANNIF automatically activates a simulated inference engine to keep the full UI functional.

---

## Folder Structure

```
PBL SANNIF/
│
├── .streamlit/
│   └── config.toml              # Streamlit light theme config
│
├── models/
│   ├── xgboost_model.json       # Production XGBoost model (JSON format)
│   ├── attack_encoder.pkl       # Serialized LabelEncoder
│   └── encoded_dataset.csv      # UNSW-NB15 encoded dataset
│
├── reports/                     # Auto-saved prediction reports (created on startup)
│
├── app.py                       # Main Streamlit interface and page router
├── model_loader.py              # Model deserialization and predict_attacks function
├── requirements.txt             # Production dependencies
└── README.md                    # Documentation
```

---

## Getting Started

### 1. Prerequisites
Ensure you have Python 3.9+ installed on your system.

### 2. Install Dependencies
Initialize your virtual environment and install the required libraries:

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Model Files Setup
Before running the application, place your pre-trained model and label encoder files inside the `models/` directory:
- `models/xgboost_model.json` — Trained XGBoost classifier (JSON format, loaded via `model.load_model()`)
- `models/attack_encoder.pkl` — Label encoder mapping integer predictions to attack category strings

### 4. Run the Application
Start the Streamlit application:

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Model & Dataset Specifications

- **Underlying Classifier**: XGBoost Baseline
- **Accuracy**: 82.01%
- **Benchmark Dataset**: UNSW-NB15
- **Features Analyzed**: 34 features (e.g. `dur`, `proto`, `service`, `state`, `spkts`, `dpkts`, etc.)
- **Output Classes**: 10 (9 Intrusions: Generic, Exploits, Fuzzers, DoS, Reconnaissance, Analysis, Backdoor, Shellcode, Worms + 1 Normal category)

---

## Diagnostics Information

If the **System Health** panel shows a red indicator:
1. Navigate to the **Diagnostics** page.
2. Verify the exact path error shown in the console logs.
3. If the engine mode shows **Simulated Fallback** (amber indicator), the XGBoost DLL is blocked by Windows security policies — the app remains fully functional using simulated inference.

---

## Pages

| Page | Purpose |
|------|---------|
| 🏠 Dashboard | Live KPI cards, model profile, detection metrics |
| 📂 Upload Dataset | CSV drag-and-drop with schema validation |
| 🔍 Detection Results | Run XGBoost inference, view paginated threat logs |
| 📊 Analytics | Plotly pie/bar/histogram charts |
| ⬇ Export Report | Download CSV, view server archives |
| 🩺 Diagnostics | System environment, backend status, feature list |
| ℹ About | Project description, model specifications |