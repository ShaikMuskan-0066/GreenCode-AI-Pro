# GreenCode AI Pro

**GreenCode AI Pro** is a **Streamlit** dashboard for **AI sustainability**: upload a Python training script, get **AST-based** inefficiency checks, **CodeCarbon**-backed CO₂ estimates (India grid mix), **live CPU/RAM/GPU** charts (**Plotly**), and exportable **reports**.

![Python](https://img.shields.io/badge/python-3.10+-3776AB?style=flat&logo=python&logoColor=white)

## Features

- **Dark UI** (`.streamlit/config.toml` + custom CSS) with header, metrics, and cards.
- **Upload** `.py` training scripts or load the built-in **`sample_train.py`**.
- **Detects**: large batch sizes, `num_workers = 0`, missing/disabled mixed precision, full fine-tuning patterns.
- **Real-time monitoring** (~2s refresh via `st.fragment`): CPU %, RAM %, optional **NVIDIA GPU** % (`nvidia-ml-py` / `pynvml`).
- **Carbon**: heuristic kWh from issues + **CodeCarbon** `Emissions` for **IND**; INR cost at ~8.5/kWh (teaching default).
- **Live CO₂ line**: load-scaled display from the baseline estimate (demo visualization, not a power meter).
- **Plotly**: gauges + multi-series history chart.
- **Reports**: auto-save to `reports/report.txt`, preview, **download**, re-save button.

## Architecture

```text
┌─────────────┐     ┌────────────┐     ┌────────────────┐
│  Streamlit  │────▶│ analyzer.py│────▶│ AnalysisResult │
│   app.py    │     └────────────┘     └───────┬────────┘
└──────┬──────┘                               │
       │                                      ▼
       │         ┌────────────────┐     ┌──────────────┐
       ├────────▶│carbon_tracker  │────▶│CarbonEstimate│
       │         └────────────────┘     └──────────────┘
       │                                      │
       │         ┌────────────┐               │
       ├────────▶│ monitor.py │               │
       │         │ (psutil +  │               │
       │         │  pynvml)   │               │
       │         └────────────┘               │
       │                                      ▼
       │         ┌──────────────┐     ┌─────────────┐
       └────────▶│suggestions.py│     │  utils.py   │
                 └──────────────┘     │ save report │
                                       └─────────────┘
```

## Installation

```powershell
cd GreenCodeAI-Pro
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Optional GPU metrics** (NVIDIA):

```powershell
pip install nvidia-ml-py
```

## Usage

```powershell
streamlit run app.py
```

Then:

1. Click **Load built-in sample** in the sidebar, **or** upload a `.py` file and click **Run analysis on uploaded file**.
2. Adjust **Training duration (hours)** to rescale energy and CO₂.
3. Watch **live** gauges and charts; scroll to **issues**, **suggestions**, and **reports**.

## Screenshots

After launching the app, capture:

- The **header** and **live metrics** row.
- **Gauge charts** and the **history** plot.
- **Carbon tracking** metrics and **issue** expanders.

Save images (for example `docs/dashboard.png`) and embed them in your fork:

```markdown
![Dashboard](docs/dashboard.png)
```

## Project layout

```text
GreenCodeAI-Pro/
├── app.py
├── analyzer.py
├── carbon_tracker.py
├── monitor.py
├── suggestions.py
├── utils.py
├── sample_train.py
├── requirements.txt
├── README.md
├── .streamlit/config.toml
└── reports/
    └── report.txt
```

## Future scope

- Multi-file / project-level analysis and `.ipynb` support.
- Hugging Face `Trainer` argument parsing.
- User-configurable grid region and tariff.
- Authentication and hosted deployment (Streamlit Community Cloud, Docker).

## Disclaimer

Estimates are **educational heuristics**. For rigorous carbon accounting, instrument real training runs (e.g. CodeCarbon `EmissionsTracker` around actual workloads) and use provider-specific emission factors.
