"""
report_generator.py — PDF sustainability reports using ReportLab.
"""

from __future__ import annotations

from pathlib import Path

from analyzer import AnalysisResult
from carbon_tracker import CarbonEstimate
from code_metrics import CodeMetrics
from sustainability_score import score_status_label


def default_pdf_path() -> Path:
    """
    Default output path for generated PDF reports.

    Returns:
        Path to reports/report.pdf.
    """
    root = Path(__file__).resolve().parent / "reports"
    root.mkdir(parents=True, exist_ok=True)
    return root / "report.pdf"


def generate_pdf_report(
    analysis: AnalysisResult,
    carbon: CarbonEstimate,
    suggestions: list[str],
    language: str,
    green_score: int,
    metrics: CodeMetrics | None = None,
    output_path: Path | None = None,
) -> Path:
    """
    Build a PDF report with analysis summary and sustainability metrics.

    Args:
        analysis: Code analysis result.
        carbon: Carbon / energy estimate.
        suggestions: Optimization suggestions.
        language: Detected programming language.
        green_score: 0–100 sustainability score.
        metrics: Optional code metrics.
        output_path: Destination PDF path.

    Returns:
        Path to the written PDF file.

    Raises:
        ImportError: If reportlab is not installed.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError as exc:
        raise ImportError("reportlab is required for PDF export. pip install reportlab") from exc

    out = output_path or default_pdf_path()
    out.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(str(out), pagesize=A4, rightMargin=2 * cm, leftMargin=2 * cm)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("GreenCode AI Pro — Sustainability Report", styles["Title"]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(f"<b>File analyzed:</b> {analysis.file_path}", styles["Normal"]))
    story.append(Paragraph(f"<b>Language:</b> {language}", styles["Normal"]))
    story.append(Spacer(1, 0.3 * cm))

    status = score_status_label(green_score)
    story.append(Paragraph(f"<b>Green Score:</b> {green_score}/100 ({status})", styles["Heading2"]))
    story.append(Spacer(1, 0.2 * cm))

    carbon_rows = [
        ["Metric", "Value"],
        ["Energy (est.)", f"{carbon.energy_kwh} kWh"],
        ["CO₂ (est.)", f"{carbon.co2_kg} kg"],
        ["Cost (est.)", f"₹{carbon.cost_inr}"],
    ]
    ct = Table(carbon_rows, colWidths=[8 * cm, 8 * cm])
    ct.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#00d4aa")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ]
        )
    )
    story.append(ct)
    story.append(Spacer(1, 0.4 * cm))

    if metrics is not None:
        story.append(Paragraph("<b>Code metrics</b>", styles["Heading3"]))
        mrows = [
            ["Total lines", str(metrics.total_lines)],
            ["Functions", str(metrics.functions)],
            ["Classes", str(metrics.classes)],
            ["Imports", str(metrics.imports)],
        ]
        mt = Table(mrows, colWidths=[8 * cm, 8 * cm])
        mt.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey)]))
        story.append(mt)
        story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("<b>Detected issues</b>", styles["Heading3"]))
    if analysis.issues:
        for issue in analysis.issues:
            story.append(Paragraph(f"• {issue.title}: {issue.detail}", styles["Normal"]))
    else:
        story.append(Paragraph("No major issues detected.", styles["Normal"]))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("<b>Optimization suggestions</b>", styles["Heading3"]))
    for s in suggestions:
        story.append(Paragraph(f"• {s}", styles["Normal"]))

    doc.build(story)
    return out
