"""
greencheck.py — CLI entry point for GreenCode AI: Sustainable ML Code Optimizer.

Usage:
    python greencheck.py sample_train.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.text import Text

from analyzer import analyze_training_script
from carbon_tracker import estimate_carbon_footprint
from suggestions import build_suggestions, issues_to_display_lines


def _configure_windows_console() -> None:
    """
    Prefer UTF-8 on Windows so Rich can print symbols like checkmarks safely.
    """
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            reconf = getattr(stream, "reconfigure", None)
            if callable(reconf):
                try:
                    reconf(encoding="utf-8")
                except Exception:
                    pass


def _ascii_logo() -> str:
    """
    Return a simple ASCII banner for the CLI (plain ASCII for Windows consoles).
    """
    return r"""
 __   __     _    ____                ____    _
 \ \ / /__ _| |_ / ___|___  ___  __ _|  _ \  (_)___
  \ V / _` | __| |  _/ _ \/ __|/ _` | | | | | / __|
   | | (_| | |_| |_| | (_) \__ \ (_| | |_| |_| \__ \
   |_|\__,_|\__|\____\___/|___/\__,_|____/(_) |___/
                                          |_|
          Sustainable ML Code Optimizer
"""


def _check_optional_dependencies(console: Console) -> None:
    """
    Print a friendly warning if optional libraries are missing.

    Raises:
        SystemExit: If a required dependency is missing (Rich is essential here).
    """
    missing: list[str] = []
    try:
        import rich  # noqa: F401, PLC0415
    except ImportError:
        missing.append("rich")
    try:
        import codecarbon  # noqa: F401, PLC0415
    except ImportError:
        missing.append("codecarbon")
    if missing:
        console.print(
            "[bold red]Missing required packages:[/bold red] "
            + ", ".join(missing)
            + "\nInstall with: [cyan]pip install -r requirements.txt[/cyan]"
        )
        raise SystemExit(1)


def _save_report_text(
    report_path: Path,
    issues_lines: list[str],
    suggestions_lines: list[str],
    energy: float,
    co2: float,
    cost_inr: float,
    method: str,
    notes: str,
) -> None:
    """
    Write a plain-text copy of the analysis to reports/report.txt.
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "=" * 48,
        "GreenCode AI - Sustainable ML Code Optimizer",
        "=" * 48,
        "",
        "Issues Found:",
        *[f"  - {line}" for line in issues_lines],
        "",
        f"Estimated Energy Usage: {energy} kWh",
        f"Estimated CO₂ Emission: {co2} kg",
        f"Estimated Electricity Cost: ₹{cost_inr}",
        "",
        f"Estimation method: {method}",
        f"Notes: {notes}",
        "",
        "Optimization Suggestions:",
        *[f"  - {line}" for line in suggestions_lines],
        "",
        "=" * 48,
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_cli(script_path: str, duration_hours: float, report_path: Path) -> int:
    """
    Run the full GreenCode pipeline for one Python file.

    Args:
        script_path: Path to the training script to analyze.
        duration_hours: Hours assumed for carbon scaling.
        report_path: Where to save the text report.

    Returns:
        Process exit code (0 = success).
    """
    _configure_windows_console()
    console = Console()
    _check_optional_dependencies(console)

    console.print(Text(_ascii_logo(), style="bold green"))

    try:
        target = Path(script_path).expanduser()
    except (OSError, ValueError) as exc:
        console.print(f"[red]Invalid path:[/red] {exc}")
        return 1

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=True,
    ) as progress:
        task_id = progress.add_task("Analyzing script...", total=3)
        try:
            analysis = analyze_training_script(target)
        except FileNotFoundError:
            console.print(f"[bold red]File not found:[/bold red] {target}")
            return 1
        except ValueError as exc:
            console.print(f"[bold red]Invalid file:[/bold red] {exc}")
            return 1
        except OSError as exc:
            console.print(f"[bold red]Could not read file:[/bold red] {exc}")
            return 1
        progress.update(task_id, advance=1)
        time.sleep(0.15)

        try:
            carbon = estimate_carbon_footprint(analysis, duration_hours=duration_hours)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Carbon estimation failed:[/red] {exc}")
            return 1
        progress.update(task_id, advance=1)
        time.sleep(0.15)

        suggestions = build_suggestions(analysis.issues)
        issue_titles = issues_to_display_lines(analysis.issues)
        progress.update(task_id, advance=1)

    # --- Rich layout ---
    title = Panel.fit(
        Text("GreenCode AI Report", style="bold white on green"),
        border_style="green",
        box=box.DOUBLE,
    )
    console.print(title)
    console.print()

    issues_table = Table(title="Issues Found", box=box.ROUNDED, show_header=True, header_style="bold magenta")
    issues_table.add_column("Status", justify="center", width=6)
    issues_table.add_column("Issue", style="white")
    issues_table.add_column("Detail", style="dim")
    if analysis.issues:
        for issue in analysis.issues:
            issues_table.add_row("[green]*[/green]", issue.title, issue.detail)
    else:
        issues_table.add_row("-", "No major issues detected", "Great job keeping this script efficient.")

    console.print(issues_table)
    console.print()

    metrics = Table(title="Estimated Impact (teaching heuristic)", box=box.HEAVY_EDGE)
    metrics.add_column("Metric", style="cyan", no_wrap=True)
    metrics.add_column("Value", style="bold yellow")
    metrics.add_row("Energy usage", f"{carbon.energy_kwh} kWh")
    metrics.add_row("CO₂ emissions", f"{carbon.co2_kg} kg")
    metrics.add_row("Electricity cost (India, approx.)", f"₹{carbon.cost_inr}")
    metrics.add_row("CO₂ calculation", carbon.method)
    console.print(metrics)
    if carbon.notes:
        console.print(Panel(carbon.notes, title="Notes", border_style="blue"))
    console.print()

    sug_table = Table(title="Optimization Suggestions", box=box.SIMPLE_HEAD)
    sug_table.add_column("Tip", style="green")
    for s in suggestions:
        sug_table.add_row(f"[green]*[/green] {s}")
    console.print(sug_table)

    console.print()
    console.rule("[bold green]End of report[/bold green]")

    _save_report_text(
        report_path,
        issue_titles,
        suggestions,
        carbon.energy_kwh,
        carbon.co2_kg,
        carbon.cost_inr,
        carbon.method,
        carbon.notes,
    )
    console.print(f"\n[dim]Report saved to:[/dim] {report_path.resolve()}")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """
    Parse command-line arguments for greencheck.
    """
    parser = argparse.ArgumentParser(
        description="GreenCode AI — analyze ML training scripts for greener patterns.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "script",
        help="Path to the Python training script (e.g. sample_train.py)",
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=1.0,
        help="Assumed training duration in hours (scales energy and CO₂).",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="reports/report.txt",
        help="Output path for the plain-text report (relative to CWD).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    Program entry point used by `python greencheck.py`.
    """
    args = _parse_args(argv)
    report_path = Path(args.report)
    return run_cli(args.script, duration_hours=args.hours, report_path=report_path)


if __name__ == "__main__":
    sys.exit(main())
