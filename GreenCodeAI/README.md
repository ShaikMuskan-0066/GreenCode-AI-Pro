# GreenCode AI – Sustainable ML Code Optimizer

**GreenCode AI** is a beginner-friendly Python command-line tool for your final-year project. It reads an ML training script, spots common energy-wasting patterns, estimates electricity use and CO₂ using [CodeCarbon](https://github.com/mlco2/codecarbon), and prints a colorful report with [Rich](https://github.com/Textualize/rich).

## Project overview

Training large models can use a lot of electricity. This project helps you **think about greener defaults**: mixed precision, sensible batch sizes, DataLoader workers, and parameter-efficient fine-tuning. The numbers are **teaching estimates** (heuristic kWh × India grid mix from CodeCarbon), not lab-grade measurements.

## Features

- **Static analysis** of a `.py` file with the `ast` module (regex fallback if syntax is invalid).
- **Detects** patterns such as:
  - `num_workers = 0`
  - Large `batch_size` (greater than 128)
  - Mixed precision disabled or missing
  - Full fine-tuning signals (`train_full_model = True`, optimizer on `model.parameters()`)
  - Weak or missing DataLoader tuning
- **Estimates** energy (kWh), CO₂ (kg), and rough electricity cost in **₹** (India retail heuristic).
- **Suggestions** mapped from issues (LoRA, quantization, workers, AMP, batch size).
- **Rich terminal UI**: tables, panels, progress animation, ASCII banner.
- **Report file**: results are saved to `reports/report.txt` on each run.

## Installation

Use Python 3.10 or newer. From the `GreenCodeAI` folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> **Note:** `sample_train.py` imports PyTorch for realism. You do **not** need PyTorch to run `greencheck.py` (analysis only reads the file). Install PyTorch separately if you want to execute `sample_train.py`.

Optional (for your own experiments, not required by this CLI):

```powershell
pip install bitsandbytes peft
```

## Usage

```powershell
cd GreenCodeAI
python greencheck.py sample_train.py
```

Optional arguments:

```powershell
python greencheck.py sample_train.py --hours 2 --report reports\my_report.txt
```

- `--hours` scales the heuristic energy and CO₂ (default `1.0`).
- `--report` sets the plain-text output path.

## Screenshots

Rich output depends on your terminal font and theme. After installing, run the command above and take a screenshot of:

- The green ASCII banner and “GreenCode AI Report” panel.
- The “Issues Found” and “Optimization Suggestions” tables.

Add your screenshot image files to your repo (for example `docs/screenshot.png`) and embed them in your fork of this README if your platform supports image previews.

## Project structure

```text
GreenCodeAI/
├── greencheck.py      # CLI entry
├── analyzer.py        # AST / regex analysis
├── carbon_tracker.py  # Heuristic kWh + CodeCarbon CO₂
├── suggestions.py     # Issue → fix mapping
├── sample_train.py    # Intentionally inefficient demo script
├── requirements.txt
├── README.md
└── reports/
    └── report.txt     # Overwritten on each run
```

## Future scope

- Deeper PyTorch / Hugging Face AST patterns (e.g. detect `Trainer` args).
- GPU model name → TDP-based energy models.
- HTML / PDF export and historical comparison across runs.
- Integration with real `EmissionsTracker` during an actual training job.

## License

Use and modify freely for educational purposes.
