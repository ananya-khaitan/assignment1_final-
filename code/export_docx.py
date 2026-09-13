"""Export only current causal-run CSV artifacts to a DOCX review bundle."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Inches, Pt

from config import OUT_DATA, OUT_FIG, OUT_ROOT

REPORT_DIR = Path(OUT_ROOT) / "reports"
DOCX_OUT = REPORT_DIR / "Critical_Mineral_Causal_Run.docx"
TABLES = [
    ("walk_forward_metrics", "Forecast accuracy on final one-step test observations"),
    ("interval_summary", "Prediction-interval quality on final test observations"),
    ("dm_test_results", "Diebold--Mariano comparisons against the proposed model"),
    ("trading_results", "Causally aligned trading evaluation"),
    ("feature_stability", "Selected-ARX feature stability"),
]


def _add_table(document: Document, frame: pd.DataFrame) -> None:
    if frame.empty:
        document.add_paragraph("No rows generated.")
        return
    table = document.add_table(rows=1, cols=len(frame.columns))
    table.style = "Light Grid Accent 1"
    for index, column in enumerate(frame.columns):
        table.rows[0].cells[index].text = str(column)
    for row in frame.itertuples(index=False):
        cells = table.add_row().cells
        for index, value in enumerate(row):
            cells[index].text = str(value)
    for row in table.rows:
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(7)


def main() -> None:
    registry = Path(OUT_DATA) / "experiment_registry.json"
    if not registry.exists():
        raise FileNotFoundError("No causal experiment registry found. Run run_all.py before exporting a report.")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    document = Document()
    for section in document.sections:
        section.top_margin = section.bottom_margin = Cm(1.2)
        section.left_margin = section.right_margin = Cm(1.2)
    title = document.add_heading("Critical Mineral Causal Forecasting Run", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    document.add_paragraph(
        "This bundle contains only artifacts produced by the current causal one-step run. "
        "Legacy tables and figures are deliberately excluded."
    )
    for stem, caption in TABLES:
        path = Path(OUT_DATA) / f"{stem}.csv"
        if not path.exists():
            document.add_heading(caption, level=1)
            document.add_paragraph(f"Missing required output: {path.name}")
            continue
        document.add_heading(caption, level=1)
        _add_table(document, pd.read_csv(path))
    # Only figures created by a current run may be included; the runner does
    # not currently generate report plots, so legacy PNGs are never embedded.
    current_figure_manifest = Path(OUT_FIG) / "causal_figure_manifest.txt"
    if current_figure_manifest.exists():
        for relative in current_figure_manifest.read_text(encoding="utf-8").splitlines():
            path = Path(OUT_FIG) / relative.strip()
            if path.exists():
                document.add_picture(str(path), width=Inches(6.5))
    document.save(DOCX_OUT)
    print(f"Saved: {DOCX_OUT}")


if __name__ == "__main__":
    main()
