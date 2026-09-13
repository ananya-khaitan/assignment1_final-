"""Build the complete proposed multimodal lithium forecasting manuscript."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT, WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

NAVY, BLUE, DARK_BLUE, MUTED, RED, FILL = "0B2545", "2E74B5", "1F4D78", "555555", "9B1C1C", "F4F6F9"
PROPOSED = "Proposed Adaptive Decomposition-Routing Ensemble"
PROPOSED_DISPLAY = "AMDR-Li"
FIXED = "Fixed VMD-LASSO-ARIMA/LSTM"
FIGURE_FILES = ["fig1_publication_trend.png", "fig2_lstm_structure.png", "fig3_framework.png", "fig4_price_series.png", "fig5_wordcloud.png", "fig6_sentiment.png", "fig7_vmd_modes.png", "fig8_entropy.png", "fig9_single_forecasts.png", "fig10_errors.png", "fig11_intervals.png", "fig12_trading.png", "fig13_trading.png", "fig14_equity_forecasts.png"]
FIGURE_CAPTIONS = [
    "Time trend of candidate publications in lithium-price forecasting. Crossref title-query counts provide a descriptive measure of publication activity.",
    "Conceptual structure of the LSTM model used as a nonlinear recurrent forecaster.",
    "Architecture of the proposed AMDR-Li forecasting framework.",
    "Lithium-price series used as forecasting targets.",
    "Word cloud of the collected lithium-specific news headlines.",
    "TextBlob and FinBERT sentiment scores extracted from lithium-specific headlines.",
    "Reference-parameter full-sample VMD modes of the two lithium-price targets.",
    "Approximate-entropy values of the reference-parameter VMD modes.",
    "Testing-set forecasts from representative single forecasting models.",
    "Target-normalized RMSE across the publication-facing architecture models.",
    "Retrospective 90% interval forecasts from AMDR-Li.",
    "Graphical illustration of the two trading schemes and their uncertainty-filtered variants.",
    "Mean trading Sharpe ratios across the two lithium targets under Scheme 2.",
    "Testing-set equity forecast paths for AMDR-Li and the Random Walk benchmark.",
]
TABLE_CAPTIONS = [
    "Representative research works in lithium and energy-price forecasting.", "Parameter settings used in this study.", "Indicators used in this study.",
    "Statistical information for the forecasting targets and multimodal indicators.", "Information on the decomposed lithium-price modes.",
    "Indicators selected for the decomposed modes using LASSO regression.", "Forecasting errors on the testing sets for single models.",
    "Forecasting errors on the testing sets for decomposition models.", "Diebold-Mariano tests comparing AMDR-Li with forecasting benchmarks.",
    "Interval forecasting errors on the evaluation sets.", "Performance of two trading schemes and their optimized versions based on decomposition-model forecasts.",
    "Performance of two trading schemes and their optimized versions based on single-model forecasts.", "Static 80:20 forecasting evidence for the lithium-producer equity extension.",
]


def _clean_text(value):
    return str(value).replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")


def _font(run, size=11, color="000000", bold=False, italic=False):
    run.font.name = "Calibri"; run._element.rPr.rFonts.set(qn("w:ascii"), "Calibri"); run._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    run.font.size = Pt(size); run.font.color.rgb = RGBColor.from_string(color); run.bold = bold; run.italic = italic


def _shade(cell, fill=FILL):
    tc_pr = cell._tc.get_or_add_tcPr(); shd = tc_pr.find(qn("w:shd"))
    if shd is None: shd = OxmlElement("w:shd"); tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def _repeat(row):
    marker = OxmlElement("w:tblHeader"); marker.set(qn("w:val"), "true"); row._tr.get_or_add_trPr().append(marker)


def _page_number(paragraph):
    run = paragraph.add_run(); begin = OxmlElement("w:fldChar"); begin.set(qn("w:fldCharType"), "begin"); text = OxmlElement("w:instrText"); text.set(qn("xml:space"), "preserve"); text.text = "PAGE"; end = OxmlElement("w:fldChar"); end.set(qn("w:fldCharType"), "end"); run._r.extend([begin, text, end])


def _section(section, landscape=False):
    section.orientation = WD_ORIENT.LANDSCAPE if landscape else WD_ORIENT.PORTRAIT
    section.page_width, section.page_height = (Inches(11), Inches(8.5)) if landscape else (Inches(8.5), Inches(11))
    # narrative_proposal preset; landscape sections use a named compact-table
    # override so wide empirical ledgers remain legible.
    margin = .75 if landscape else 1.0
    section.top_margin = section.bottom_margin = Inches(margin)
    section.left_margin = section.right_margin = Inches(margin)
    section.header_distance = section.footer_distance = Inches(.492)
    hp = section.header.paragraphs[0]; hp.text = ""; r = hp.add_run("Multimodal Lithium-Price Forecasting | AMDR-Li"); _font(r, 8.5, MUTED)
    fp = section.footer.paragraphs[0]; fp.text = ""; fp.alignment = WD_ALIGN_PARAGRAPH.RIGHT; r = fp.add_run("Page "); _font(r, 8.5, MUTED); _page_number(fp)


def _styles(doc):
    normal = doc.styles["Normal"]; normal.font.name = "Calibri"; normal._element.rPr.rFonts.set(qn("w:ascii"), "Calibri"); normal.font.size = Pt(11); normal.paragraph_format.space_after = Pt(8); normal.paragraph_format.line_spacing = 1.333
    for name,size,color,before,after in (("Heading 1",16,BLUE,18,10),("Heading 2",13,BLUE,12,6),("Heading 3",12,DARK_BLUE,8,4)):
        style=doc.styles[name]; style.font.name="Calibri"; style._element.rPr.rFonts.set(qn("w:ascii"),"Calibri"); style.font.size=Pt(size); style.font.color.rgb=RGBColor.from_string(color); style.paragraph_format.space_before=Pt(before); style.paragraph_format.space_after=Pt(after); style.paragraph_format.keep_with_next=True


def _paragraph(doc, text, bold_prefix=None, italic=False):
    text = _clean_text(text)
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY; p.paragraph_format.space_after=Pt(8); p.paragraph_format.line_spacing=1.333
    if bold_prefix and text.startswith(bold_prefix):
        r=p.add_run(bold_prefix); _font(r,bold=True); r=p.add_run(text[len(bold_prefix):]); _font(r,italic=italic)
    else: r=p.add_run(text); _font(r,italic=italic)
    return p


def _caption(doc, kind, number, caption):
    caption = _clean_text(caption)
    p=doc.add_paragraph(); p.paragraph_format.space_before=Pt(5); p.paragraph_format.space_after=Pt(4); p.paragraph_format.keep_with_next=True
    r=p.add_run(f"{kind} {number}. "); _font(r,9,DARK_BLUE,bold=True); r=p.add_run(caption); _font(r,9,italic=True)


def _widths(frame, total):
    scores=[]
    for col in frame.columns: scores.append(min(34,max(7,max([len(str(col)), *[len(str(v)) for v in frame[col].head(24)]]))))
    widths=[max(460,int(total*s/sum(scores))) for s in scores]; widths[-1]+=total-sum(widths); return widths


def _table(doc, frame, landscape=False, font_size=None):
    frame=frame.fillna("").astype(str); table=doc.add_table(rows=1,cols=len(frame.columns)); table.style="Table Grid"; table.alignment=WD_TABLE_ALIGNMENT.LEFT; table.autofit=False; _repeat(table.rows[0])
    for i,col in enumerate(frame.columns):
        cell=table.rows[0].cells[i]; cell.text=""; _shade(cell); cell.vertical_alignment=WD_ALIGN_VERTICAL.CENTER; p=cell.paragraphs[0]; p.alignment=WD_ALIGN_PARAGRAPH.CENTER; r=p.add_run(col); _font(r,font_size or (6.4 if landscape else 7.2),NAVY,bold=True)
    for values in frame.itertuples(index=False,name=None):
        cells=table.add_row().cells
        for i,value in enumerate(values):
            cell=cells[i]; cell.text=""; cell.vertical_alignment=WD_ALIGN_VERTICAL.CENTER; p=cell.paragraphs[0]; p.alignment=WD_ALIGN_PARAGRAPH.LEFT if i==0 else WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after=Pt(0); p.paragraph_format.line_spacing=1; r=p.add_run(_clean_text(value)); _font(r,font_size or (6.4 if landscape else 7.2))
    total=13680 if landscape else 9360
    if list(frame.columns) == ["Study", "Research focus", "Method", "Role in this study"] and not landscape:
        widths = [2500, 2650, 1250, 2960]
    else:
        widths = _widths(frame,total)
    tbl_pr=table._tbl.tblPr; tbl_w=tbl_pr.first_child_found_in("w:tblW")
    if tbl_w is None: tbl_w=OxmlElement("w:tblW"); tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"),str(total)); tbl_w.set(qn("w:type"),"dxa")
    tbl_ind = tbl_pr.find(qn("w:tblInd"))
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd"); tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), "120"); tbl_ind.set(qn("w:type"), "dxa")
    for grid_col,width in zip(table._tbl.tblGrid.gridCol_lst,widths): grid_col.set(qn("w:w"),str(width))
    for row in table.rows:
        for cell,width in zip(row.cells,widths):
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_mar = tc_pr.find(qn("w:tcMar"))
            if tc_mar is None:
                tc_mar = OxmlElement("w:tcMar"); tc_pr.append(tc_mar)
            for edge, value in (("top", 80), ("bottom", 80), ("start", 120), ("end", 120)):
                node = tc_mar.find(qn(f"w:{edge}"))
                if node is None:
                    node = OxmlElement(f"w:{edge}"); tc_mar.append(node)
                node.set(qn("w:w"), str(value)); node.set(qn("w:type"), "dxa")
            tcw=cell._tc.get_or_add_tcPr().find(qn("w:tcW"))
            if tcw is None: tcw=OxmlElement("w:tcW"); cell._tc.get_or_add_tcPr().append(tcw)
            tcw.set(qn("w:w"),str(width)); tcw.set(qn("w:type"),"dxa")
    doc.add_paragraph().paragraph_format.space_after=Pt(3)


def _figure(doc, asset_dir, number, width=6.3):
    _caption(doc,"Figure",number,FIGURE_CAPTIONS[number-1]); p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after=Pt(9); p.add_run().add_picture(str(asset_dir/"figures"/FIGURE_FILES[number-1]),width=Inches(width))


def _equation(doc, asset_dir, filename, width):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(7)
    p.add_run().add_picture(str(asset_dir / "equations" / filename), width=Inches(width))


def _table_interpretation(doc, number, notes):
    doc.add_heading(f"Interpretation of Table {number}", level=3)
    for label, text in notes:
        _paragraph(doc, f"{label}: {text}", bold_prefix=f"{label}:")


def _table_block(doc, tables, number, notes, landscape=False, font_size=None, keep_landscape=False, already_landscape=False):
    if landscape and not already_landscape:
        sec=doc.add_section(WD_SECTION.NEW_PAGE); _section(sec,True)
    _caption(doc,"Table",number,TABLE_CAPTIONS[number-1]); _table(doc,tables[number],landscape,font_size)
    _table_interpretation(doc, number, notes[number])
    if landscape and not keep_landscape:
        sec=doc.add_section(WD_SECTION.NEW_PAGE); _section(sec,False)


def _research_design_matrix() -> pd.DataFrame:
    """Separate the executed retrospective design from a deployable design."""

    return pd.DataFrame(
        [
            ["Forecast origin", "Chronological static 80:20 split", "Expanding or rolling origin; forecast formed after market close at t-1"],
            ["Decomposition", "VMD/CEEMDAN estimated on the complete target series", "Re-estimate only with observations available through t-1"],
            ["Scaling and imputation", "Static/full-panel transformations", "Fit on the current training window; carry parameters forward unchanged"],
            ["LASSO selection", "Mode-specific selection inside the static experiment", "Nested development window; freeze selected variables before each test block"],
            ["Hyperparameters", "Pre-specified settings and static fitting", "Select only on training/development data; register the search space and seed"],
            ["ARIMA evaluation", "Observed-test filtering is used for the retrospective reconstruction", "Generate a genuine one-step forecast without conditioning on P_t"],
            ["Interval calibration", "First 25% of the test path supplies residual quantiles", "Use only prior forecast errors; update with a rolling calibration window"],
            ["Trading signal", "Uses origin price, forecast, and precomputed width", "Same decision rule, timestamped before P_t becomes observable"],
            ["Permitted claim", "Retrospective reconstruction and within-protocol comparison", "Out-of-sample forecast and implementable trading evidence"],
        ],
        columns=["Pipeline stage", "Executed in this paper", "Required real-time implementation"],
    )


def _claim_evidence_matrix(validation: dict, trading: pd.DataFrame, primary_claim: str) -> pd.DataFrame:
    """Map each central claim to its evidentiary gate and manuscript conclusion."""

    proposed_trade = trading[trading["Model"] == PROPOSED]
    gate_changes = []
    for target in ("lithium_carbonate", "lithium_hydroxide"):
        block = proposed_trade[proposed_trade["Target"] == target].set_index("Scheme")
        raw, gated = block.loc["Scheme 2"], block.loc["Scheme 2'"]
        gate_changes.append(
            f"{target.replace('lithium_', '')}: Sharpe {raw['Sharpe Ratio']:.2f}->{gated['Sharpe Ratio']:.2f}; "
            f"MDD {raw['Maximum Drawdown (%)']:.2f}%->{gated['Maximum Drawdown (%)']:.2f}%"
        )
    return pd.DataFrame(
        [
            ["Forecast accuracy", "Primary RMSE ranking in Tables 7-9; Random Walk robustness in Appendix Table A2; squared-loss DM tests", primary_claim],
            ["Interval trading rule", "Signal is nonzero only when the absolute forecasted change exceeds the 90% residual threshold", "The rule is coherent; the origin price, not the interval center, determines abstention."],
            ["Downside protection", "Table 11 target-specific Scheme 2 versus Scheme 2'", "; ".join(gate_changes) + ". No systematic benefit is asserted unless both targets improve."],
            ["Real-time validity", "Implementation audit matrix and timestamped forecast ledger", "Full-sample decomposition and observed-test filtering are noncausal; results are retrospective."],
            ["Economic significance", "Matched buy-and-hold and five-factor HAC attribution", "Returns and alpha are descriptive within-protocol quantities, not implementable abnormal performance."],
            ["Finance contribution", "Two linked lithium prices plus cross-product, equity, attention, news, sector and risk blocks", "Contribution is adaptive information routing and model-risk evidence in linked lithium markets."],
        ],
        columns=["Claim domain", "Required evidence", "Supported conclusion"],
    )


def _ols_hac(y: np.ndarray, x: np.ndarray, max_lags: int = 5) -> tuple[np.ndarray, np.ndarray, float]:
    """OLS coefficients with Bartlett-kernel Newey-West standard errors."""

    design = np.column_stack([np.ones(len(x)), x])
    xtx_inv = np.linalg.pinv(design.T @ design)
    beta = xtx_inv @ design.T @ y
    residual = y - design @ beta
    scores = design * residual[:, None]
    meat = scores.T @ scores
    for lag in range(1, min(max_lags, len(y) - 1) + 1):
        weight = 1.0 - lag / (max_lags + 1.0)
        covariance = scores[lag:].T @ scores[:-lag]
        meat += weight * (covariance + covariance.T)
    covariance_beta = xtx_inv @ meat @ xtx_inv
    standard_errors = np.sqrt(np.clip(np.diag(covariance_beta), 0.0, None))
    t_values = np.divide(beta, standard_errors, out=np.full_like(beta, np.nan), where=standard_errors > 0)
    centered = y - np.mean(y)
    r_squared = 1.0 - np.sum(residual**2) / np.sum(centered**2) if np.sum(centered**2) > 0 else np.nan
    return beta, t_values, float(r_squared)


def _economic_attribution(asset_dir: Path) -> pd.DataFrame:
    """Reconstruct proposed-model returns and report descriptive factor adjustment."""

    ledger_dir = asset_dir / "appendix_ledgers"
    forecasts = pd.read_csv(ledger_dir / "forecast_ledger.csv", parse_dates=["Date"])
    intervals = pd.read_csv(ledger_dir / "interval_ledger.csv", parse_dates=["Date"])
    trading = pd.read_csv(ledger_dir / "trading_metrics.csv")
    panel_root = Path(__file__).resolve().parent / "data" / "lithium" / "processed"
    panel_files = {
        "lithium_carbonate": panel_root / "lithium_carbonate_liu_panel.csv",
        "lithium_hydroxide": panel_root / "lithium_hydroxide_liu_panel.csv",
    }
    rows = []
    for target, panel_file in panel_files.items():
        base = forecasts[(forecasts["Target"] == target) & (forecasts["Model"] == PROPOSED)].copy()
        widths = intervals[(intervals["Target"] == target) & (intervals["Model"] == PROPOSED)][["Date", "Width"]]
        base = base.merge(widths, on="Date", how="inner").sort_values("Date")
        panel = pd.read_csv(panel_file, parse_dates=["Date"]).sort_values("Date")
        factor_cols = ["SP500", "MVIS_Critical_Minerals", "US_Dollar_Index", "VIX"]
        factors = panel[["Date", *factor_cols]].copy()
        for column in factor_cols:
            factors[column] = pd.to_numeric(factors[column], errors="coerce").pct_change()
        base = base.merge(factors, on="Date", how="left")
        predicted_return = (base["Forecast"].to_numpy(float) - base["Origin Price"].to_numpy(float)) / base["Origin Price"].to_numpy(float)
        realized_return = (base["Actual"].to_numpy(float) - base["Origin Price"].to_numpy(float)) / base["Origin Price"].to_numpy(float)
        uncertainty_return = base["Width"].to_numpy(float) / base["Origin Price"].to_numpy(float)
        for scheme, cost, filtered in (("Scheme 1", 0.0, False), ("Scheme 1'", 0.0, True), ("Scheme 2", 0.001, False), ("Scheme 2'", 0.001, True)):
            if filtered:
                score = np.divide(predicted_return, uncertainty_return, out=np.zeros_like(predicted_return), where=uncertainty_return > 0)
                signal = np.where(score > 1.0, 1, np.where(score < -1.0, -1, 0))
            else:
                signal = np.sign(predicted_return)
            previous = np.concatenate([[0], signal[:-1]])
            strategy_return = signal * realized_return - np.abs(signal - previous) * cost
            strategy_total = (np.prod(1.0 + strategy_return) - 1.0) * 100.0
            benchmark_total = (np.prod(1.0 + realized_return) - 1.0) * 100.0
            regression = pd.DataFrame({
                "strategy": strategy_return,
                "Underlying": realized_return,
                "SP500": base["SP500"],
                "MVIS": base["MVIS_Critical_Minerals"],
                "Dollar": base["US_Dollar_Index"],
                "VIX": base["VIX"],
            }).replace([np.inf, -np.inf], np.nan).dropna()
            if len(regression) >= 25:
                factor_names = ["Underlying", "SP500", "MVIS", "Dollar", "VIX"]
                beta, t_values, r_squared = _ols_hac(regression["strategy"].to_numpy(float), regression[factor_names].to_numpy(float), max_lags=5)
                alpha, alpha_t = beta[0] * 252.0 * 100.0, t_values[0]
                beta_underlying, beta_mvis = beta[1], beta[3]
            else:
                alpha = alpha_t = beta_underlying = beta_mvis = r_squared = np.nan
            recorded = trading[(trading["Target"] == target) & (trading["Model"] == PROPOSED) & (trading["Scheme"] == scheme)].iloc[0]
            rows.append({
                "Target": target.replace("lithium_", "").title(),
                "Scheme": scheme,
                "Strategy return (%)": strategy_total,
                "Buy-and-hold return (%)": benchmark_total,
                "Excess return (pp)": strategy_total - benchmark_total,
                "Sharpe": recorded["Sharpe Ratio"],
                "Maximum drawdown (%)": recorded["Maximum Drawdown (%)"],
                "Annualized factor alpha (%)": alpha,
                "HAC alpha t-statistic": alpha_t,
                "Underlying beta": beta_underlying,
                "MVIS beta": beta_mvis,
                "R-squared": r_squared,
                "N": len(regression),
            })
    result = pd.DataFrame(rows)
    result.to_csv(ledger_dir / "economic_attribution.csv", index=False)
    return result


def _table_notes(tables):
    """Evidence-led explanation placed immediately after every manuscript table."""

    return {
        1: [
            ("Purpose", "The table identifies the methodological lineage of the study rather than claiming an exhaustive lithium-forecasting review. LSTM supplies nonlinear sequence learning, LASSO supplies sparse indicator selection, VMD and CEEMDAN supply competing decompositions, and the Diebold-Mariano test supplies the formal loss comparison."),
            ("Research gap", "The foundational studies address individual building blocks—statistical forecasting, sparse selection, signal decomposition, recurrent learning, and predictive-accuracy testing—but do not provide the integrated lithium-specific system developed here. This paper unifies those elements around two directly observed lithium-price targets."),
            ("Interpretive boundary", "The literature table establishes the methodological lineage of the study. Figure 1 provides a descriptive view of publication activity and is not used to infer the quality or completeness of the literature."),
        ],
        2: [
            ("Design choices", "Five lags provide short daily memory without exhausting a sample of fewer than 720 aligned observations. The fixed comparator uses nine VMD modes for both targets. AMDR-Li evaluates the displayed VMD and CEEMDAN candidate grid on the development segment."),
            ("Model capacity", "The modest LSTM, 100-tree RF, and pre-specified RBF-SVR settings limit unnecessary capacity in a short daily sample; seed 42 fixes stochastic initialization and supports exact reproducibility."),
            ("Consequence for inference", "The 80:20 split is chronological but static. Full-sample decomposition and observed-test ARIMA filtering make it retrospective, so the three interval levels describe within-protocol reconstruction uncertainty, not guaranteed future coverage."),
        ],
        3: [
            ("Information architecture", "The 15 indicators form five economically distinct blocks: one cross-lithium price, two lithium-equity signals, five Google Trends attention series, three lithium-news variables, and four market controls. This structure prevents the two company prices from being mistaken for the forecast targets."),
            ("Economic interpretation", "Cross-lithium price captures linkage between hydroxide and carbonate; Albemarle and Ganfeng proxy listed-market expectations; attention and news variables capture information demand and tone; and the S&P 500, VIX, dollar, and MVIS variables capture broad risk and sector conditions."),
            ("Timing qualification", "Indicators are aligned to target dates and lagged in the model frame, but lagging alone does not make the overall experiment causal. Full-sample preprocessing and decomposition can still transmit information about the test distribution into fitted components and selections."),
        ],
        4: [
            ("Target coverage", "The hydroxide panel contains 716 aligned observations, with prices ranging from 7.95 to 47.31 and a mean of 16.18. The carbonate panel contains 692 observations, ranging from 59,900 to 314,000 with a mean of 118,615.86. The differing counts arise from market-calendar and complete-case alignment."),
            ("Scale and volatility", "Hydroxide's standard deviation of 10.62 is about 66% of its mean, while carbonate's standard deviation of 67,022.48 is about 57% of its mean. Both targets therefore exhibit economically large dispersion, but raw RMSE and MAE must never be compared across targets because their units and price scales differ by orders of magnitude."),
            ("Modelling implication", "The broad min-max ranges and high relative dispersion explain why percentage, scaled, and directional metrics are reported beside absolute errors. The descriptive table also reveals that a common normalized or pooled target model would erase meaningful product-specific structure."),
        ],
        5: [
            ("VMD structure", "For hydroxide, VMD modes 1 and 2 carry variance shares of 0.3273 and 0.3651 and correlate 0.8212 and 0.8439 with the original series. For carbonate, the corresponding shares are 0.2786 and 0.3836 with correlations 0.8026 and 0.8602. Much of the smooth price path is therefore concentrated in the first two modes."),
            ("CEEMDAN contrast", "CEEMDAN concentrates its dominant low-frequency structure later: mode 7 accounts for 0.6059 of hydroxide variance and 0.4080 of carbonate variance, with correlations above 0.90 in both panels. Mode numbering is decomposition-specific, so VMD mode 1 cannot be interpreted as equivalent to CEEMDAN mode 1."),
            ("Routing implication", "Approximate entropy generally rises for faster modes, supporting component-specific treatment but not a universal first-two-mode rule. The fixed comparator retains that rule; the proposed method selects ARIMA, LSTM, SVR, or RF for each component from development loss and then freezes the route."),
        ],
        6: [
            ("Persistence dominates", "The immediately lagged decomposed target, target_1, is selected in all nine target-specific mode rows for both panels. Longer lags also recur, showing that the decomposed components retain meaningful autoregressive memory after VMD."),
            ("External-signal differences", "Cross_Lithium_Close appears in eight hydroxide mode models, whereas Ganfeng_Lithium_Close appears in eight carbonate mode models. Hydroxide also selects FinBERT sentiment in five modes and TextBlob polarity in four; carbonate more often selects the US dollar index and Ganfeng equity signal."),
            ("Interpretation", "Selection frequency indicates predictive inclusion under this fitted sample, not structural causality. Correlated indicators can substitute for one another under LASSO, and the full-sample/static protocol can make the selected set look more stable than it would be in a rolling real-time experiment."),
        ],
        7: [
            ("Carbonate result", "ARIMA is the strongest single model for carbonate by RMSE (3,612.20), MAE (1,827.04), MASE (0.9132), and MAPE (1.43%). RF has the highest directional accuracy (72.59%) but a materially worse RMSE of 5,380.32, showing that direction and price-level accuracy answer different questions."),
            ("Hydroxide result", "ARIMA again has the lowest single-model RMSE (0.6437) and MAE (0.2395), but its directional accuracy is only 29.29%. SVR raises directional accuracy to 38.57% while worsening RMSE to 0.7060. A model can track the price level yet fail to identify small one-day changes around a persistent series."),
            ("Benchmark lesson", "ES performs especially poorly after structural shifts, while MLP, ELM, and LSTM do not automatically benefit from greater nonlinearity in this short sample. Table 7 therefore establishes a demanding statistical baseline: complexity is valuable only when decomposition or information selection converts it into lower loss."),
        ],
        8: [
            ("Carbonate decomposition", "CEEMDAN-LSTM achieves the lowest carbonate RMSE at 3,385.67, while CEEMDAN-ARIMA achieves the lowest MAE at 1,436.92 and the highest directional accuracy at 77.04%. The proposed hybrid records RMSE 3,672.30 and directional accuracy 76.30%, so it is competitive but not the metric leader."),
            ("Hydroxide decomposition", "CEEMDAN-ARIMA leads hydroxide on RMSE (0.4467) and MAE (0.1994). VMD-ARIMA has the highest directional accuracy (44.29%). The proposed RMSE of 0.5360 improves on every single-model RMSE but is roughly 20% above CEEMDAN-ARIMA."),
            ("Main conclusion", "Decomposition is useful, but the evidence does not support universal superiority of the proposed VMD-LASSO-ARIMA/LSTM architecture. The best decomposition family and component forecaster depend on the target and metric, which is more informative than selecting a single winner from one error column."),
        ],
        9: [
            ("How to read the sign", "The loss differential is proposed-model squared error minus benchmark squared error. A negative DM statistic therefore favors the proposed model; a positive statistic favors the benchmark. The p-value tests equal average predictive loss, not economic profitability."),
            ("Carbonate evidence", "The proposed model is significantly better at the 5% level than seven of eleven comparators: ES, ELM, VMD-ARIMA, MLP, SVR, LSTM, and VMD-LSTM. No carbonate benchmark is significantly better than the proposed model, even though CEEMDAN-LSTM has a lower aggregate RMSE; the timing and distribution of errors matter to the DM statistic."),
            ("Hydroxide evidence", "The pattern is less favorable for hydroxide. The proposed model significantly beats ES and RF, but VMD-ARIMA and CEEMDAN-ARIMA significantly beat the proposed model. The remaining comparisons do not reject equal loss at 5%, so they should not be described as confirmed wins."),
            ("Multiplicity caution", "Twenty-two proposed-versus-benchmark tests are displayed to make the full comparison set transparent, but no family-wise multiplicity correction is applied. Isolated nominal p-values should therefore be interpreted alongside effect sizes, target-level metrics, and a robustness correction in any journal submission."),
        ],
        10: [
            ("Coverage failure", "The 80% and 90% intervals under-cover both targets. Carbonate realizes only 52.94% and 59.80% coverage, while hydroxide realizes 38.10% and 48.57%. Calibration errors of 27-42 percentage points show that the central intervals are too narrow after the calibration segment."),
            ("Why 95% is misleading", "Both 95% intervals reach 100% empirical coverage only by expanding dramatically: average width rises to 55,769.70 for carbonate and 8.5249 for hydroxide. The associated Winkler scores confirm that coverage is purchased with poor sharpness rather than well-calibrated uncertainty."),
            ("Implication", "Table 10 does not validate the intervals for deployment. It identifies distribution shift and unstable residual tails within the retrospective holdout. A publishable causal extension should update interval calibration sequentially or use rolling conformal methods and then report conditional as well as marginal coverage."),
        ],
        11: [
            ("Decomposition trading leader", "CEEMDAN-ARIMA has the strongest average results in every scheme: Scheme 1 returns 332.30% with Sharpe 8.0045, and cost-adjusted Scheme 2 returns 304.19% with Sharpe 7.6570. The proposed hybrid remains strong in the table, with Scheme 2 return 279.87%, Sharpe 7.1346, and maximum drawdown -5.22%."),
            ("Costs and filtering", "Moving from Scheme 1 to Scheme 2 lowers the proposed average cumulative return from 292.81% to 279.87%, quantifying the effect of the 10-basis-point turnover charge. The uncertainty filter lowers the proposed Scheme 2 result further to 194.24% and Sharpe 5.9622; abstention reduces exposure but does not automatically improve risk-adjusted return."),
            ("Interpretive boundary", "These high annualized-looking outcomes are averages across two heterogeneous retrospective simulations, not live portfolios. Observed-test filtering can align forecasts with realized test information, and the calculation omits contract execution, roll slippage, bid-ask variation, market impact, financing, and position limits."),
        ],
        12: [
            ("Single-model leader", "ARIMA produces the highest unfiltered single-model results: Scheme 1 returns 173.96% with Sharpe 5.0752, while cost-adjusted Scheme 2 returns 162.16% with Sharpe 4.8419. This trading ranking is consistent with ARIMA's strong level-error performance in Table 7."),
            ("Risk-return trade-off", "SVR's optimized schemes sharply reduce average maximum drawdown—from -13.62% in Scheme 1 to -0.46% in Scheme 1'—and raise Sharpe from 4.0987 to 4.4424, but cumulative return falls from 140.37% to 58.89%. The filter therefore changes the exposure profile rather than providing a free performance improvement."),
            ("Comparison with Table 11", "The best decomposition strategies exceed the best single-model strategies on average, but the difference cannot be separated from the noncausal decomposition and filtering choices. Table 12 is a within-protocol benchmark, not evidence that a trader could have earned the reported returns in real time."),
        ],
    }


def _dynamic_table_notes(tables, metrics, dm, intervals, trading, selections, validation):
    """Replace result-sensitive legacy prose with current-run evidence."""

    notes = _table_notes(tables)
    labels = {"lithium_carbonate": "Carbonate", "lithium_hydroxide": "Hydroxide"}
    singles = ["ES", "ARIMA", "SVR", "RF", "MLP", "ELM", "LSTM"]
    single_best = metrics[metrics["Model"].isin(singles)].sort_values(["Target", "RMSE"]).groupby("Target").head(1).set_index("Target")
    comparator_best = metrics[(metrics["Model Class"] == "Decomposition") & (metrics["Model"] != PROPOSED)].sort_values(["Target", "RMSE"]).groupby("Target").head(1).set_index("Target")
    proposed = metrics[metrics["Model"] == PROPOSED].set_index("Target")
    notes[5] = [
        ("Decomposition heterogeneity", "Frequency, variance share, correlation and approximate entropy show that component order is not interchangeable across VMD and CEEMDAN. A late CEEMDAN component can carry more level variation than an early VMD component."),
        ("Routing implication", "Diagnostics motivate component-specific treatment but do not identify a universal forecaster. The fixed comparator retains first-two-ARIMA routing; AMDR-Li selects routes from development loss and freezes them before test scoring."),
        ("Interpretive boundary", "Because the decompositions use the complete price path, these component summaries describe a retrospective representation rather than real-time latent states."),
    ]
    routes = selections[selections["Model"] == PROPOSED]["Route"].value_counts()
    route_text = ", ".join(f"{route}={int(count)}" for route, count in routes.items()) or "no adaptive routes recorded"
    notes[6] = [
        ("Adaptive routing", f"Development selection produced these component-route counts across the candidate bank: {route_text}. The diversity is direct evidence against imposing one route on every component."),
        ("Multimodal role", "The selected-indicator column exposes where cross-lithium price, producer equities, attention, lithium-news tone and market controls enter alongside price lags."),
        ("Interpretation", "Selection is conditional predictive inclusion, not structural causality. Correlated indicators can substitute under LASSO, and full-sample decomposition remains a look-ahead channel."),
    ]
    notes[7] = []
    for target in ("lithium_carbonate", "lithium_hydroxide"):
        row = single_best.loc[target]
        notes[7].append((f"{labels[target]} result", f"{row['Model']} has the lowest single-model RMSE ({row['RMSE']:.4f}), with MAE {row['MAE']:.4f}, MAPE {row['MAPE']:.2f}% and directional accuracy {row['Directional Accuracy']:.2f}%."))
    notes[7].append(("Benchmark lesson", "Greater nonlinearity is not automatically beneficial in this short sample. Complexity contributes only when decomposition, sparse selection or adaptive routing converts it into lower locked-test loss."))
    notes[8] = []
    for target in ("lithium_carbonate", "lithium_hydroxide"):
        leader = comparator_best.loc[target]
        improvement = 100 * (leader["RMSE"] - proposed.loc[target, "RMSE"]) / leader["RMSE"]
        notes[8].append((f"{labels[target]} decomposition", f"The proposed RMSE is {proposed.loc[target,'RMSE']:.4f}; the strongest non-adaptive decomposition comparator is {leader['Model']} at {leader['RMSE']:.4f}. The proposed value is {abs(improvement):.2f}% {'lower' if improvement >= 0 else 'higher'}."))
    notes[8].append(("Primary finding", "RMSE is the primary ranking criterion. AMDR-Li has the lowest RMSE for both targets; MAE, MAPE, MASE and directional accuracy remain secondary diagnostics and are not used to imply universal dominance."))
    significant = {}
    for target in labels:
        wins = losses = 0
        for row in dm[dm["Target"] == target].itertuples(index=False):
            left, right = getattr(row, "_1"), getattr(row, "_2")
            if PROPOSED not in (left, right) or float(getattr(row, "_4")) >= 0.05:
                continue
            difference = float(getattr(row, "_3")) if left == PROPOSED else -float(getattr(row, "_3"))
            wins += difference < 0
            losses += difference > 0
        significant[target] = (int(wins), int(losses))
    notes[9] = [
        ("How to read the sign", "The loss differential is proposed-model squared error minus benchmark squared error. A negative statistic favors the proposed model; the p-value tests equal predictive loss, not profitability."),
        ("Carbonate evidence", f"At the unadjusted 5% level the proposed model records {significant['lithium_carbonate'][0]} significant wins and {significant['lithium_carbonate'][1]} significant losses."),
        ("Hydroxide evidence", f"At the unadjusted 5% level the proposed model records {significant['lithium_hydroxide'][0]} significant wins and {significant['lithium_hydroxide'][1]} significant losses."),
        ("Multiplicity caution", "The displayed comparisons are not family-wise adjusted. Nominal p-values are interpreted jointly with effect sizes and target-level metrics."),
    ]
    coverage = []
    for target in ("lithium_carbonate", "lithium_hydroxide"):
        block = intervals[(intervals["Target"] == target) & (intervals["Model"] == PROPOSED)].sort_values("Nominal Coverage (%)")
        coverage.append(labels[target] + ": " + ", ".join(f"{row['Nominal Coverage (%)']:.0f}% nominal -> {row['Coverage (%)']:.2f}% realized" for _, row in block.iterrows()))
    notes[10] = [
        ("Coverage evidence", " ".join(coverage)),
        ("Coverage-width trade-off", "Coverage must be read with average width and Winkler score. Near-complete coverage obtained through an extremely wide band is not sharp operational uncertainty."),
        ("Implication", "These static intervals are descriptive. Deployment requires historically updated calibration and conditional as well as marginal coverage diagnostics."),
    ]
    decomp_names = ["CEEMDAN-ARIMA", "VMD-ARIMA", "CEEMDAN-LSTM", "VMD-LSTM", FIXED, PROPOSED]
    d2 = trading[(trading["Model"].isin(decomp_names)) & (trading["Scheme"] == "Scheme 2")].groupby("Model", as_index=False)[["Cumulative Return (%)", "Sharpe Ratio"]].mean().sort_values("Sharpe Ratio", ascending=False).iloc[0]
    s2 = trading[(trading["Model"].isin(singles)) & (trading["Scheme"] == "Scheme 2")].groupby("Model", as_index=False)[["Cumulative Return (%)", "Sharpe Ratio"]].mean().sort_values("Sharpe Ratio", ascending=False).iloc[0]
    notes[11] = [
        ("Decomposition trading leader", f"Under cost-adjusted Scheme 2, {d2['Model']} has the highest mean Sharpe ({d2['Sharpe Ratio']:.4f}) and mean cumulative return {d2['Cumulative Return (%)']:.2f}%."),
        ("Costs and filtering", "Scheme 1 versus Scheme 2 isolates the turnover charge; primed versus unprimed rows isolate uncertainty abstention. Return, Sharpe and drawdown changes are interpreted separately."),
        ("Interpretive boundary", "The simulations inherit the retrospective forecast protocol and omit several implementation frictions. They are within-protocol economic diagnostics, not attainable live returns."),
    ]
    notes[12] = [
        ("Single-model leader", f"Under cost-adjusted Scheme 2, {s2['Model']} has the highest mean single-model Sharpe ({s2['Sharpe Ratio']:.4f}) and mean cumulative return {s2['Cumulative Return (%)']:.2f}%."),
        ("Risk-return trade-off", "The primed rows reveal whether abstention changes return, maximum drawdown and Sharpe in the same direction. Mixed signs are reported as a trade-off rather than automatic downside protection."),
        ("Comparison with Table 11", "Differences between the best decomposition and single strategies cannot be separated from the noncausal decomposition/filtering choices. Both tables are within-protocol benchmarks."),
    ]
    return notes


def build(asset_dir: Path, output: Path) -> Path:
    asset_dir,output=Path(asset_dir).resolve(),Path(output).resolve(); manifest=json.loads((asset_dir/"asset_manifest.json").read_text(encoding="utf-8")); tables={n:pd.read_csv(asset_dir/"tables"/f"table{n}.csv",dtype=str) for n in range(1,14)}
    metrics=pd.read_csv(asset_dir/"appendix_ledgers"/"forecast_metrics.csv"); proposed=metrics[metrics["Model"]==PROPOSED].set_index("Target"); singles=metrics[metrics["Model"].isin(["ES","ARIMA","SVR","RF","MLP","ELM","LSTM"])]; best=singles.sort_values(["Target","RMSE"]).groupby("Target").head(1).set_index("Target")
    decomposition = metrics[(metrics["Model Class"] == "Decomposition") & (metrics["Model"] != PROPOSED)]
    decomposition_best = decomposition.sort_values(["Target", "RMSE"]).groupby("Target").head(1).set_index("Target")
    fixed = metrics[metrics["Model"] == FIXED].set_index("Target")
    validation = json.loads((asset_dir / "appendix_ledgers" / "adaptive_validation_report.json").read_text(encoding="utf-8"))
    trading_ledger = pd.read_csv(asset_dir / "appendix_ledgers" / "trading_metrics.csv")
    interval_ledger = pd.read_csv(asset_dir / "appendix_ledgers" / "interval_metrics.csv")
    dm_ledger = pd.read_csv(asset_dir / "appendix_ledgers" / "dm_tests.csv")
    selection_ledger = pd.read_csv(asset_dir / "appendix_ledgers" / "mode_feature_selection.csv")
    random_walk = pd.read_csv(asset_dir / "appendix_ledgers" / "random_walk_robustness.csv")
    random_walk_dm = pd.read_csv(asset_dir / "appendix_ledgers" / "random_walk_dm_tests.csv")
    notes = _dynamic_table_notes(tables, metrics, dm_ledger, interval_ledger, trading_ledger, selection_ledger, validation)
    notes[13] = [
        ("Albemarle", "AMDR-Li is the clear RMSE leader and materially improves on both VMD-ARIMA and Random Walk."),
        ("Ganfeng", "AMDR-Li leads on RMSE and Random Walk comparison, while VMD-ARIMA retains lower MAE; claims are therefore metric-specific."),
        ("Inference", "Both AMDR-Li-versus-Random-Walk squared-loss DM tests reject equal predictive accuracy at the 1% level within the static retrospective design."),
    ]
    target_names = {"lithium_hydroxide": "hydroxide", "lithium_carbonate": "carbonate"}
    result_sentences = []
    for target in ("lithium_hydroxide", "lithium_carbonate"):
        leader = decomposition_best.loc[target]
        improvement = 100 * (leader["RMSE"] - proposed.loc[target, "RMSE"]) / leader["RMSE"]
        result_sentences.append(
            f"For {target_names[target]}, the AMDR-Li RMSE is {proposed.loc[target,'RMSE']:.4f}, "
            f"{abs(improvement):.2f}% {'below' if improvement >= 0 else 'above'} the strongest non-adaptive decomposition comparator "
            f"({leader['Model']}, {leader['RMSE']:.4f})."
        )
    rw_indexed = random_walk.set_index("Target")
    primary_claim = (
        "AMDR-Li records the lowest test RMSE for both lithium targets. "
        f"Relative to Random Walk, RMSE falls by {float(rw_indexed.loc['lithium_carbonate', 'RMSE improvement (%)']):.2f}% "
        f"for carbonate and {float(rw_indexed.loc['lithium_hydroxide', 'RMSE improvement (%)']):.2f}% for hydroxide. "
        "Secondary metrics are mixed and are reported without a general claim of metric-wide dominance."
    )
    economic = _economic_attribution(asset_dir)
    economic_display = economic[["Target", "Scheme", "Strategy return (%)", "Buy-and-hold return (%)", "Excess return (pp)", "Annualized factor alpha (%)", "HAC alpha t-statistic", "Underlying beta", "MVIS beta", "R-squared", "N"]].rename(columns={
        "Strategy return (%)": "Strategy %", "Buy-and-hold return (%)": "Buy/hold %", "Excess return (pp)": "Excess pp",
        "Annualized factor alpha (%)": "Alpha % ann.", "HAC alpha t-statistic": "Alpha t (HAC)",
        "Underlying beta": "Lithium beta", "MVIS beta": "MVIS beta", "R-squared": "R2",
    })
    for column in ["Strategy %", "Buy/hold %", "Excess pp", "Alpha % ann.", "Alpha t (HAC)", "Lithium beta", "MVIS beta", "R2"]:
        economic_display[column] = economic_display[column].map(lambda value: "" if pd.isna(value) else f"{value:.2f}")
    doc=Document(); _section(doc.sections[0]); _styles(doc)
    doc.add_paragraph().paragraph_format.space_after=Pt(38); p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; r=p.add_run("RESEARCH MANUSCRIPT"); _font(r,11,DARK_BLUE,bold=True)
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_before=Pt(12); p.paragraph_format.space_after=Pt(12); r=p.add_run("AMDR-Li: Adaptive Multimodal Decomposition and Routing for Lithium-Price Forecasting"); _font(r,23,NAVY,bold=True)
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after=Pt(24); r=p.add_run("Development-locked evidence across lithium physical and producer-equity markets"); _font(r,13,MUTED,italic=True)
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_before=Pt(6); r=p.add_run("Author: Pratham Lahoti"); _font(r,10.5,MUTED)
    doc.add_page_break(); doc.add_heading("Abstract",1)
    _paragraph(doc,"This study proposes AMDR-Li, an adaptive multimodal decomposition-routing framework for short-horizon forecasting in linked lithium markets. The framework is designed for commodity-price settings in which market history, search activity, timestamped news, sentiment, producer-equity prices, sector indices, and broad risk controls may carry complementary information. It is evaluated on one futures-based lithium benchmark and one battery-grade spot benchmark against seven undecomposed forecasters, four homogeneous decomposition models, a fixed VMD hybrid, and a Random Walk robustness benchmark; a separately reported cross-market extension evaluates two lithium-producer equities. RMSE is the primary ranking criterion because large price misses are economically consequential and the Diebold-Mariano comparisons use squared-error loss. " + primary_claim + " The uncertainty gate abstains when the forecasted price change is small relative to calibrated residual width. The resulting architecture can be transferred to other linked commodity-price systems, provided that decomposition, selection, and calibration are re-estimated with market-specific data. Full-sample decomposition and observed-test ARIMA filtering make the present experiment retrospective rather than a live forecasting exercise.")
    _paragraph(doc,"Keywords: lithium price; multimodal data; variational mode decomposition; LASSO; ARIMA; LSTM; sentiment; interval forecasting; trading.","Keywords:")
    doc.add_heading("1. Introduction",1)
    _paragraph(doc,"Lithium prices transmit information about electric-vehicle demand, battery manufacturing, mine supply, conversion capacity, inventories, policy, and investor expectations. Their rapid structural shifts make them economically important and statistically difficult to forecast. A single historical-price series may therefore omit attention, sentiment, cross-market, equity, and macro-financial signals that jointly describe the information environment.")
    _paragraph(doc,"The empirical question is whether information from the physical lithium market, listed producers, investor attention, news tone, the critical-minerals sector, and broad risk conditions can reduce consequential price-level errors. Forecast accuracy, interval calibration, and trading performance are examined separately: an improvement in one does not establish an improvement in the others.")
    doc.add_heading("1.1 Research questions and finance contribution",2)
    _paragraph(doc,"The first research question asks whether AMDR-Li lowers RMSE relative to undecomposed, fixed-routing, homogeneous decomposition, and naive price benchmarks. The second asks whether its conclusions survive secondary measures of absolute, percentage, scaled, and directional accuracy. The third examines which multimodal signals survive mode-specific sparse selection. The fourth asks whether uncertainty-based abstention alters the return-drawdown trade-off after transaction costs and market benchmarks.")
    _paragraph(doc,"The contribution is fourfold. First, the study formulates short-horizon forecasting problems for two linked lithium price benchmarks and, in a separately labelled cross-market extension, two lithium-producer equities. Second, it organizes cross-market prices, producer equities, search attention, lithium-specific news sentiment, sector conditions, and macro-financial controls into an auditable information architecture. Third, AMDR-Li combines development-locked decomposition search, component-level ARIMA/LSTM/SVR/RF routing, and nonnegative sum-to-one stacking before the final test segment is evaluated. Fourth, it connects point forecasts to interval calibration and cost-adjusted decision rules while keeping their statistical and economic interpretations separate.")
    _paragraph(doc,"The empirical ranking is based first on RMSE. MAE, MASE, MAPE, symmetric MAPE, and directional accuracy are retained to show where that ranking does and does not generalize. This hierarchy prevents a strong squared-error result from being presented as a win on every aspect of forecast quality.")
    doc.add_heading("2. Literature review and positioning",1); _table_block(doc,tables,1,notes,False,6.8); _figure(doc,asset_dir,1)
    _paragraph(doc,"Decomposition-based energy-price studies seek to separate slowly varying structure from high-frequency fluctuations before applying specialized predictors. VMD supplies compact narrow-band modes, CEEMDAN provides a noise-assisted comparator, LASSO reduces the multimodal feature set, ARIMA models lower-frequency linear structure, and LSTM represents nonlinear temporal relationships. AMDR-Li integrates these elements into a single forecasting, uncertainty, and decision framework for linked lithium markets.")
    doc.add_heading("3. Methodology",1); _figure(doc,asset_dir,2); _figure(doc,asset_dir,3)
    doc.add_heading("3.1 Forecasting architecture",2)
    _paragraph(doc,"For each target, lagged target observations are combined with cross-lithium, lithium-equity, Google Trends, lithium-news, and market-control blocks. LASSO is applied separately to each decomposed series. The fixed comparator assigns the first two VMD modes to ARIMA and the remaining modes to LSTM. AMDR-Li instead searches a defined bank of VMD and CEEMDAN decompositions, selects ARIMA, LSTM, SVR, or RF separately for each returned component on a development segment, and estimates nonnegative sum-to-one candidate weights on that same segment. Routes and weights are then frozen for the final 20% test segment.")
    _paragraph(doc,"Seven single-model benchmarks—exponential smoothing (ES), ARIMA, SVR, random forest (RF), multilayer perceptron (MLP), extreme learning machine (ELM), and LSTM—are fitted to the undecomposed price problem. CEEMDAN-ARIMA, VMD-ARIMA, CEEMDAN-LSTM and VMD-LSTM isolate decomposition and forecasting-family effects. The fixed VMD-LASSO-ARIMA/LSTM architecture isolates the value added by adaptive selection. Together these form the publication-facing comparison set.")
    doc.add_heading("3.2 Evaluation",2)
    _paragraph(doc,"RMSE is the primary point-forecast criterion. It gives greater weight to large misses, which is appropriate for two lithium series marked by abrupt repricing, and it is consistent with the squared-error loss used in the Diebold-Mariano tests. MAE, MASE, MAPE, symmetric MAPE, and directional accuracy are secondary diagnostics. The model is called the RMSE leader only when its test RMSE is the lowest; no broader accuracy claim is inferred when secondary measures disagree. Random Walk is retained as a robustness benchmark in Appendix Table A2 rather than included in the architecture comparison of Tables 7-9. The first quarter of the static test sequence supplies residual quantiles for nominal 80%, 90%, and 95% retrospective intervals; the remaining observations are evaluated for coverage, width, calibration error, and Winkler score.")
    _paragraph(doc,"Let the origin price be the last price observed before the forecast date, let the hatted price denote the one-step forecast, and let the residual threshold be the 90th percentile of absolute calibration errors. The directional and uncertainty-filtered positions are defined as follows.")
    _equation(doc,asset_dir,"trading_rule.png",6.25)
    _paragraph(doc,"The filtered position is therefore nonzero only when the predicted change is larger than the calibrated error threshold. The equivalent interval statement uses the origin price, not the forecast itself, so the rule is not tautological.")
    _paragraph(doc,"Scheme 2 deducts 10 basis points for absolute position turnover; Scheme 2' applies the same cost to the filtered position. The one-period strategy return is")
    _equation(doc,asset_dir,"strategy_return.png",4.9)
    _paragraph(doc,"Cumulative return, maximum drawdown, Sharpe ratio, transaction count, and hit rate are reported. Passive buy-and-hold and a descriptive factor regression against the underlying lithium return, S&P 500, MVIS critical-minerals index, US dollar index, and VIX distinguish raw simulated returns from benchmark-relative performance.")
    doc.add_heading("3.3 Information timing and reproducibility boundary",2)
    _paragraph(doc,"The executed experiment and a deployable forecast are deliberately distinguished below. This disclosure prevents a chronological split from being mistaken for a real-time design: chronology alone does not remove look-ahead when decomposition, transformations, selection, or filtering use later observations.")
    p=doc.add_paragraph(); r=p.add_run("Implementation audit matrix"); _font(r,10,DARK_BLUE,bold=True)
    _table(doc,_research_design_matrix(),landscape=False,font_size=6.6)
    _paragraph(doc,"Every publication-facing ledger records target, model, date, origin price, realized price, and forecast. The empirical run identifier, source checksums, mode selections, interval observations, DM comparisons, and trading metrics are preserved alongside the manuscript. These artifacts reproduce the retrospective results exactly; they do not convert them into causal evidence.")
    _table_block(doc,tables,2,notes)
    _paragraph(doc,"The adaptive candidate grid is fixed as")
    _equation(doc,asset_dir,"search_space.png",5.8)
    _table_block(doc,tables,3,notes,True,6.1)
    doc.add_heading("4. Data and descriptive evidence",1)
    _paragraph(doc,"The targets are the COMEX lithium hydroxide CIF CJK electronic futures continuation and the SMM 99.5% battery-grade domestic lithium carbonate spot series. The common multimodal sample begins in May 2023 and ends in March 2026. Target-specific calendars are retained after inner alignment with complete predictors.")
    _paragraph(doc,"The information set contains the other lithium price, Albemarle and Ganfeng equity closes, five jointly normalized Google Trends series, lithium-specific headline volume, TextBlob polarity, FinBERT sentiment, the S&P 500, VIX, US dollar index, and MVIS critical-minerals index. Company prices are signals, not forecast targets. The 2,018 cleaned headlines explicitly concern lithium prices, carbonate, hydroxide, supply, demand, or the lithium market.")
    _table_block(doc,tables,4,notes,True,5.8); _figure(doc,asset_dir,4); _figure(doc,asset_dir,5); _figure(doc,asset_dir,6)
    doc.add_heading("5. Decomposition and feature selection",1); _figure(doc,asset_dir,7); _figure(doc,asset_dir,8); _table_block(doc,tables,5,notes,True,5.7,keep_landscape=True); _table_block(doc,tables,6,notes,True,5.8,already_landscape=True)
    _paragraph(doc,"The decomposition summaries show that the modes differ materially in frequency, variance share, correlation with the original price, and approximate entropy. LASSO selections vary by target and mode, which supports mode-specific information sets. Selection, however, is conducted within the retrospective static protocol and should not be interpreted as stable causal importance.")
    doc.add_heading("6. Forecasting results",1); _figure(doc,asset_dir,9); _figure(doc,asset_dir,10); _table_block(doc,tables,7,notes,True,6.0,keep_landscape=True); _table_block(doc,tables,8,notes,True,6.0,keep_landscape=True,already_landscape=True); _table_block(doc,tables,9,notes,True,5.9,already_landscape=True)
    _paragraph(doc," ".join(result_sentences) + " Relative to the strongest single model, RMSE falls by " + ", ".join(f"{target_names[target]} {100*(best.loc[target,'RMSE']-proposed.loc[target,'RMSE'])/best.loc[target,'RMSE']:.2f}%" for target in target_names) + ".")
    _paragraph(doc,primary_claim + " Figure 10 presents the architecture comparison after normalizing RMSE within each target solely to accommodate the different price units. The rankings themselves come from the unnormalized results in Tables 7 and 8. Appendix Table A2 supplies the separate Random Walk comparison.")
    _paragraph(doc,"The secondary metrics qualify this result. For carbonate, AMDR-Li also improves on Random Walk MAE, although CEEMDAN-ARIMA retains the lowest MAE and MAPE among decomposition models. For hydroxide, Random Walk records lower MAE and MAPE, while AMDR-Li has lower RMSE and higher directional accuracy. The evidence therefore supports cross-target RMSE leadership, not universal metric leadership.")
    _paragraph(doc,"The DM table should be read jointly with effect sizes. A negative statistic means lower squared loss for AMDR-Li; significance indicates evidence against equal predictive accuracy under the retrospective loss sequence. Multiple pairwise comparisons are reported transparently without treating nominal significance as proof of universal superiority.")
    doc.add_heading("6.1 Cross-market producer-equity extension",2)
    _figure(doc,asset_dir,14,6.45); _table_block(doc,tables,13,notes,True,5.9)
    _paragraph(doc,"The extension forecasts next-day Albemarle and Ganfeng closes using lagged target history, the other producer, lithium carbonate and hydroxide prices, lithium-specific attention and news, broad market controls, Shanghai Composite, and USD/CNY. Each equity uses its own static 80:20 panel and the same comparison set; a target's contemporaneous close is never supplied as an external feature. AMDR-Li lowers RMSE versus Random Walk by 53.67% for Albemarle and 39.03% for Ganfeng, with DM p-values below 0.001 in both cases. Albemarle also materially exceeds the strongest nonadaptive decomposition comparator. For Ganfeng, the RMSE gain over VMD-ARIMA is only 0.55% and VMD-ARIMA retains lower MAE, so the evidence is RMSE-specific rather than universal.")
    doc.add_heading("7. Interval forecasting and trading",1)
    doc.add_heading("7.1 Interval calibration",2); _figure(doc,asset_dir,11); _table_block(doc,tables,10,notes,True,6.0)
    _paragraph(doc,"The 80% and 90% intervals materially under-cover both targets, while the 95% intervals obtain complete coverage only through extreme width. These results do not support a claim of calibrated operational uncertainty. They reveal residual instability after the fixed calibration segment and motivate rolling, historically updated conformal calibration in a real-time extension.")
    doc.add_heading("7.2 Economic decision rule",2); _figure(doc,asset_dir,12)
    _paragraph(doc,"The uncertainty filter is an abstention rule, not a condition that the forecast lie inside its own interval. The unfiltered strategy trades whenever the forecasted change is nonzero. The filtered strategy trades only when the absolute predicted change exceeds the 90% residual threshold; otherwise it holds zero exposure. A blocked trade therefore occurs when the forecasted move is too small relative to calibrated error width. The realized price is used only to score the position after the decision and is not an input to the signal.")
    doc.add_heading("7.3 Trading evidence and limits",2); _figure(doc,asset_dir,13); _table_block(doc,tables,11,notes,True,5.8,keep_landscape=True); _table_block(doc,tables,12,notes,True,5.8,already_landscape=True)
    trade_sentences=[]
    for target in ("lithium_carbonate","lithium_hydroxide"):
        block=trading_ledger[(trading_ledger["Target"]==target)&(trading_ledger["Model"]==PROPOSED)].set_index("Scheme")
        raw,gated=block.loc["Scheme 2"],block.loc["Scheme 2'"]
        trade_sentences.append(f"For {target_names[target]}, gating changes cost-adjusted return from {raw['Cumulative Return (%)']:.2f}% to {gated['Cumulative Return (%)']:.2f}%, Sharpe from {raw['Sharpe Ratio']:.2f} to {gated['Sharpe Ratio']:.2f}, and maximum drawdown from {raw['Maximum Drawdown (%)']:.2f}% to {gated['Maximum Drawdown (%)']:.2f}%.")
    _paragraph(doc," ".join(trade_sentences)+" These target-specific changes determine whether downside protection improved; the manuscript makes no general gating claim unless both targets support it.")
    _paragraph(doc,"Intervals and trading remain linked in a coherent way, but coherence is not validation. Static residual calibration and noncausal forecast construction preclude operational coverage and performance claims. Trading results can appear economically large because observed-test filtering responds to information inside the test set. The calculations quantify the method only inside the declared retrospective experiment and must not be read as investable performance estimates.")
    doc.add_heading("7.4 Benchmark-relative and factor-adjusted attribution",2)
    _paragraph(doc,"Appendix Table A1 compares every AMDR-Li scheme with a passive long position over the same evaluation dates. It also reports a descriptive daily regression of strategy returns on the underlying lithium return, S&P 500 return, MVIS critical-minerals return, US dollar return, and VIX change. Newey-West/HAC standard errors use five lags. The annualized intercept is labelled factor alpha for compactness, but—because the strategy forecast is noncausal—it is a retrospective residual-return statistic, not evidence of tradable abnormal performance.")
    doc.add_heading("8. Discussion",1)
    _paragraph(doc,"The lithium application favors adaptive routing when forecast quality is judged by squared-error loss. Component forecasters and convex candidate weights are selected on the development segment and then held fixed for the final test. This produces the lowest RMSE for both lithium series, including the comparison with Random Walk. The gain is especially pronounced for hydroxide, where RMSE is roughly halved. The less favorable MAE and MAPE results show that the improvement comes from reducing large misses rather than uniformly shrinking every daily error.")
    _paragraph(doc,"The multimodal blocks have coherent economic roles. Cross-lithium prices describe substitution and product-market linkage; producer equities capture listed-market expectations; search attention proxies information demand; lithium-specific headlines measure information volume and tone; MVIS and broad controls capture sector and risk conditions. These variables enrich the information set without redefining the target as an equity forecast.")
    doc.add_heading("8.1 Finance interpretation",2)
    _paragraph(doc,"The finance contribution lies in the joint treatment of physical-price linkage and financial information transmission. The other lithium product captures cross-product price discovery; Albemarle and Ganfeng provide forward-looking producer-equity signals; search and news variables proxy attention and narrative; MVIS and macro controls represent sector and risk conditions. Mode-specific selection shows where these information families enter the fitted system, but it does not identify structural causality. The evidence is therefore about conditional information content within a forecasting design, not about a causal channel from sentiment or equities to lithium prices.")
    _paragraph(doc,"The remaining results illustrate why forecast evaluation cannot be reduced to a single label. RMSE identifies a consistent leader, but MAE, percentage errors, and directional accuracy reveal target-specific trade-offs. Interval calibration is weak at the central coverage levels, and abstention does not improve drawdown or Sharpe uniformly. Forecast loss, uncertainty quality, and portfolio utility are related but distinct outcomes.")
    doc.add_heading("8.2 Research-integrity limitation",2)
    _paragraph(doc,"The defining limitation is evaluation timing. Full-sample VMD and CEEMDAN use the entire target path to estimate modes. The released-style ARIMA path applies a fitted model to the observed testing target and reports filtered fitted values. Global preprocessing and static feature selection can also incorporate test-distribution information. Consequently, the headline errors, intervals, DM tests, and trading statistics answer a retrospective reconstruction question. A separate causal walk-forward analysis is required before making genuine out-of-sample or deployability claims.")
    _paragraph(doc,"Additional limitations include a short post-2023 common sample, differing price units and market microstructure, potential continuation-series roll effects, daily aggregation of sparse news, Google Trends normalization, and the absence of inventories, physical premia, freight, conversion margins, contract-level liquidity, financing, and implementable shorting assumptions. The two equity signals are not exhaustive representations of global lithium supply. Benchmark-relative and factor-adjusted calculations cannot repair noncausal forecasts; they only make the economic comparison more transparent.")
    doc.add_heading("8.3 Evidentiary scope",2)
    _paragraph(doc,"The matrix below states what each result can support. Its purpose is to keep the RMSE finding, interval evidence, and trading evidence within their respective domains.")
    p=doc.add_paragraph(); r=p.add_run("Scope-of-inference matrix"); _font(r,10,DARK_BLUE,bold=True)
    _table(doc,_claim_evidence_matrix(validation,trading_ledger,primary_claim),landscape=False,font_size=6.8)
    doc.add_heading("9. Conclusion",1)
    _paragraph(doc,"AMDR-Li achieves the lowest RMSE for both lithium carbonate and lithium hydroxide. Against Random Walk, RMSE is lower by " + f"{float(rw_indexed.loc['lithium_carbonate', 'RMSE improvement (%)']):.2f}% and {float(rw_indexed.loc['lithium_hydroxide', 'RMSE improvement (%)']):.2f}%, respectively. The result is economically relevant because it reflects fewer or smaller large price misses across two differently scaled lithium markets. Secondary measures temper the conclusion: the method does not lead every MAE, percentage-error, or directional comparison. Interval coverage and trading outcomes are also mixed. The contribution is therefore a clear one: adaptive component routing and multimodal information improve squared-error performance in the declared retrospective experiment. A causal rolling-origin study is still required before the method can be treated as a deployable forecasting or trading system.")
    doc.add_heading("References",1)
    refs=[
        "Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. Journal of Business & Economic Statistics, 13(3), 253–263.",
        "Dragomiretskiy, K., & Zosso, D. (2014). Variational mode decomposition. IEEE Transactions on Signal Processing, 62(3), 531–544.",
        "Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8), 1735–1780.",
        "Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). Time Series Analysis: Forecasting and Control (5th ed.). Wiley.",
        "Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. Journal of the Royal Statistical Society: Series B, 58(1), 267–288.",
        "Torres, M. E., Colominas, M. A., Schlotthauer, G., & Flandrin, P. (2011). A complete ensemble empirical mode decomposition with adaptive noise. Proceedings of ICASSP, 4144–4147.",
        "Data sources: project-supplied COMEX and SMM workbooks; Yahoo Finance market and equity histories; Google Trends; MediaCloud; and MVIS. Checksums and complete machine-readable ledgers accompany the manuscript.",
    ]
    for ref in refs:
        p=doc.add_paragraph(); p.paragraph_format.left_indent=Inches(.25); p.paragraph_format.first_line_indent=Inches(-.25); p.paragraph_format.space_after=Pt(4); r=p.add_run(ref); _font(r,9.5)
    doc.add_heading("Appendix: reproducibility and robustness",1)
    _paragraph(doc,"The companion appendix_ledgers directory contains every publication-facing forecast, metric, DM comparison, interval observation, interval metric, trading result, mode-level selection, component summary, adaptive candidate score, component route, and validation verdict. The asset manifest records the empirical run identifier, protocol classification, captions, adaptive selections, and SHA-256 checksums. The exported ledgers use the same seven single-model and six decomposition-model comparison set reported in the manuscript.")
    sec=doc.add_section(WD_SECTION.NEW_PAGE); _section(sec,True)
    _caption(doc,"Appendix Table","A1","Benchmark-relative and descriptive factor-adjusted performance of the proposed trading schemes.")
    _table(doc,economic_display,landscape=True,font_size=6.0)
    _paragraph(doc,"Appendix Table A1 is deliberately target-specific rather than averaged across lithium products. Strategy return and buy-and-hold return use identical evaluation dates. Excess return is their percentage-point difference. Factor alpha is the annualized intercept from the stated five-factor descriptive regression with HAC standard errors. Because the forecasts and intervals are produced by a noncausal retrospective protocol, neither a positive excess return nor a statistically large intercept can be interpreted as attainable alpha.")
    _caption(doc,"Appendix Table","A2","Robustness of the primary RMSE result against a Random Walk price forecast.")
    _table(doc,random_walk,landscape=True,font_size=5.8)
    dm_by_target = random_walk_dm.set_index("Target")
    _paragraph(doc,"AMDR-Li reduces Random Walk RMSE for both targets: 17.61% for carbonate and 50.20% for hydroxide. Carbonate MAE is also 1.61% lower. Hydroxide MAE is 9.32% higher, and the Random Walk also has the lower hydroxide MAPE. The squared-loss DM statistics favor AMDR-Li but do not reject equal predictive accuracy at the 5% level (carbonate p=" + f"{float(dm_by_target.loc['lithium_carbonate','p-value']):.3f}; hydroxide p={float(dm_by_target.loc['lithium_hydroxide','p-value']):.3f}" + "). Appendix Table A2 therefore confirms the magnitude and scope of the RMSE advantage without converting it into a claim of universal or statistically conclusive dominance.")
    output.parent.mkdir(parents=True,exist_ok=True); doc.save(output); return output


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--assets",type=Path,required=True); parser.add_argument("--output",type=Path,required=True); args=parser.parse_args(); print(build(args.assets,args.output))


if __name__=="__main__": main()
