"""Build a browser-printable PDF source for the causal critical-mineral paper.

This mirrors the DOCX manuscript with a standards-based HTML print layout.  It
is used as a deterministic PDF renderer where headless LibreOffice is absent.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd


TITLE = "Causal Multimodal Forecasting and Trading Evaluation for Critical-Mineral Equities"
PROPOSED = "VMD-LASSO-ARX/LSTM"


def _figure(asset_dir: Path, manifest: dict, number: int, note: str = "") -> str:
    metadata = manifest["figures"][f"Figure {number}"]
    image = (asset_dir / metadata["file"]).resolve().as_uri()
    caption = metadata["caption"]
    if note:
        caption += " " + note
    return f'''<figure><img src="{image}" alt="Figure {number}"><figcaption><b>Figure {number}.</b> {html.escape(caption)}</figcaption></figure>'''


def _table(asset_dir: Path, manifest: dict, number: int, *, landscape: bool = False) -> str:
    frame = pd.read_csv(asset_dir / "tables" / f"table{number}.csv")
    table = frame.to_html(index=False, escape=True, classes="results-table", border=0)
    style = " landscape" if landscape else ""
    return f'''<section class="table-block{style}"><p class="table-caption"><b>Table {number}.</b> {html.escape(manifest["tables"][f"Table {number}"])}</p>{table}</section>'''


def build(asset_dir: Path, output: Path) -> Path:
    asset_dir = asset_dir.resolve()
    manifest = json.loads((asset_dir / "asset_manifest.json").read_text(encoding="utf-8"))
    table7 = pd.read_csv(asset_dir / "tables" / "table7.csv").set_index("Model")
    table8 = pd.read_csv(asset_dir / "tables" / "table8.csv").set_index("Model")
    rw = float(table7.loc["Random Walk", "RMSE"])
    arima = float(table7.loc["ARIMA", "RMSE"])
    proposed = float(table8.loc[PROPOSED, "RMSE"])
    content = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{TITLE}</title>
<style>
@page {{ size: Letter; margin: .65in .62in .6in; }}
@page wide {{ size: Letter landscape; margin: .45in .42in; }}
* {{ box-sizing: border-box; }}
body {{ color:#182430; font-family: Calibri, Arial, sans-serif; font-size:10.2pt; line-height:1.34; margin:0; }}
h1,h2,h3 {{ color:#0B2545; font-family: Georgia, 'Times New Roman', serif; }}
h1 {{ font-size:19pt; border-bottom:1px solid #9EB8CF; padding-bottom:5px; margin:24px 0 10px; break-before:page; }}
h1.first {{ break-before:auto; }}
h2 {{ font-size:13.5pt; margin:15px 0 7px; }} h3 {{ font-size:11.2pt; margin:12px 0 4px; }}
p {{ margin:0 0 8px; text-align:justify; }}
.cover {{ min-height:8.8in; padding-top:1.15in; page-break-after:always; }}
.eyebrow {{ color:#2E74B5; font-weight:bold; font-size:10pt; letter-spacing:1.2px; text-transform:uppercase; }}
.cover h1 {{ border:0; color:#0B2545; font-size:29pt; line-height:1.12; margin:18px 0; }}
.subtitle {{ font-family:Georgia,serif; color:#41596F; font-size:15pt; line-height:1.35; max-width:6.4in; }}
.rule {{ background:#2E74B5; height:5px; width:1.5in; margin:32px 0 22px; }}
.meta {{ color:#526474; font-size:10pt; }}
.callout {{ border-left:4px solid #2E74B5; background:#F1F6FA; padding:10px 12px; margin:12px 0; color:#263B4D; }}
figure {{ margin:13px auto 16px; text-align:center; page-break-inside:avoid; }}
figure img {{ max-width:100%; max-height:6.65in; object-fit:contain; }}
figcaption {{ color:#445665; font-size:8.7pt; line-height:1.25; margin:4px auto 0; max-width:6.6in; text-align:left; }}
.table-block {{ margin:11px 0 16px; page-break-inside:avoid; }}
.table-block.landscape {{ page:wide; break-before:page; }}
.table-caption {{ color:#283D50; font-size:9pt; margin:0 0 5px; text-align:left; }}
table.results-table {{ border-collapse:collapse; width:100%; font-size:7.3pt; line-height:1.15; }}
table.results-table th {{ background:#0B2545; color:white; font-weight:bold; padding:4px 4px; text-align:left; vertical-align:bottom; }}
table.results-table td {{ border-bottom:1px solid #D7E0E7; padding:3px 4px; vertical-align:top; }}
table.results-table tr:nth-child(even) td {{ background:#F5F8FA; }}
.landscape table.results-table {{ font-size:6.5pt; }}
.refs p {{ padding-left:.25in; text-indent:-.25in; font-size:9pt; }}
.appendix {{ background:#F5F8FA; border-top:2px solid #2E74B5; padding:10px 12px; }}
.page-note {{ color:#5B6C7A; font-size:8.3pt; }}
</style></head><body>
<section class="cover"><div class="eyebrow">Empirical research manuscript</div><h1 class="first">{TITLE}</h1><p class="subtitle">A fully causal, one-step comparison of multimodal forecasts, uncertainty intervals, and cost-aware trading signals across eight global critical-mineral companies.</p><div class="rule"></div><p class="meta">Canonical run: {html.escape(manifest.get("canonical_run_id", "not recorded"))}<br>Prepared 13 August 2026<br>All reported empirical outputs originate from the completed causal pipeline.</p><div class="callout"><b>Research-integrity statement.</b> The released-paper shadow reconstruction is excluded from this manuscript because it is a noncausal forensic exercise. No final-test target is used for decomposition, scaling, feature selection, tuning, calibration, or model fitting in the reported analysis.</div></section>

<h1 class="first">Abstract</h1><p>Critical-mineral equity prices are influenced by commodity conditions, macroeconomic variables, market factors, search interest, and news sentiment. This paper evaluates whether multimodal forecasting systems can convert those signals into reliable one-day-ahead forecasts and transaction-cost-aware trading decisions. Eight companies are assessed with a fixed final holdout, scheduled walk-forward refits, train-only VMD or CEEMDAN decomposition, historical LASSO selection, conformal intervals, Diebold-Mariano comparisons, and calibrated trading thresholds. The contribution is an auditable causal evaluation rather than a claim that model complexity always dominates simple benchmarks. Complete company-level, model-level, interval, trading, feature-selection, and DM-test records accompany the paper.</p>
<p><b>Keywords:</b> critical minerals; equity forecasting; VMD; LASSO; LSTM; causal walk-forward validation; conformal prediction; trading evaluation.</p>

<h1>1. Introduction</h1><p>Critical-mineral producers operate at the intersection of volatile commodity demand, geopolitical supply risks, energy-transition investment, and company-specific execution. A forecasting model that is attractive in a static backtest may nevertheless fail once each decision is restricted to information that was genuinely observable before the next close. We therefore evaluate a broad single-model and decomposition-model set under a common one-step causal protocol.</p><p>The paper is inspired by Liu et al. (2025), which combines decomposition, feature selection, neural forecasting, interval prediction, and trading evaluation in carbon markets. The present application changes the target domain and, critically, carries each preprocessing and selection decision forward in time rather than deriving it from a full sample.</p>{_figure(asset_dir, manifest, 1)}{_table(asset_dir, manifest, 1)}

<h1>2. Methodology</h1><h2>2.1 Causal evaluation design</h2><p>For each company, historical observations up to a forecast origin are available to the model; the next closing price is the target. Calibration observations precede the final test set. All transformations, factor alignment, predictor selection, decomposition, refits, interval residuals, and trading thresholds follow this information boundary. The final test set is never used to optimize model choices.</p>{_figure(asset_dir, manifest, 3)}
<h2>2.2 Candidate forecasters</h2><p>Benchmarks include random walk, random walk with drift, ARIMA, sparse linear models, tree ensembles, gradient boosting, LSTM, and GRU. Decomposition candidates include VMD-LSTM, VMD-GRU, CEEMDAN-LSTM, and the proposed VMD-LASSO-ARX/LSTM model. The latter routes illustrative lower-entropy modes to a sparse autoregression and higher-entropy modes to a neural residual forecaster, then recombines the forecasts.</p>{_figure(asset_dir, manifest, 2)}{_table(asset_dir, manifest, 2)}{_table(asset_dir, manifest, 3)}
<h2>2.3 Evaluation, intervals, and trading</h2><p>Forecasting is evaluated with RMSE, MAE, MASE, MAPE, and directional accuracy. Pairwise DM tests compare squared forecast loss. Distribution-free conformal intervals are derived from calibration residuals. Directional and uncertainty-adjusted strategies are evaluated at zero, 10, 25, and 50 basis-point transaction costs; thresholds are selected only on calibration data.</p>

<h1>3. Data</h1><p>The final panel contains daily closing prices for Glencore, BHP Group, Rio Tinto, Freeport-McMoRan, Albemarle, Zijin Mining, Ganfeng Lithium, and Anglo American, together with the MVIS strategic-metals benchmark, macro series, regional factors, Google Trends, and news sentiment. Each firm retains its observed price scale; comparison tables therefore report company-average errors and within-company ranks together.</p>{_figure(asset_dir, manifest, 4)}{_table(asset_dir, manifest, 4)}{_figure(asset_dir, manifest, 5)}{_figure(asset_dir, manifest, 6)}

<h1>4. Decomposition and causal feature selection</h1><p>Figure 7 is deliberately limited to Glencore's pre-test training segment. It illustrates a VMD decomposition without exposing the model to later test observations. At every scheduled refit in the empirical run, the decomposition and feature-selection process is re-executed using the historical prefix available at that time.</p>{_figure(asset_dir, manifest, 7)}{_figure(asset_dir, manifest, 8)}{_table(asset_dir, manifest, 5)}{_table(asset_dir, manifest, 6)}

<h1>5. Forecasting results</h1><p>The results are heterogeneous across firms and model families. In the cross-company descriptive averages, RMSE is {rw:.4f} for the random walk, {arima:.4f} for ARIMA, and {proposed:.4f} for the proposed decomposition architecture. Such averages span different equity price scales, so the company-level ranks and DM results are essential complements. A negative DM statistic denotes lower squared loss for the proposed model under the documented loss-difference convention.</p>{_figure(asset_dir, manifest, 9)}{_figure(asset_dir, manifest, 10)}{_table(asset_dir, manifest, 7)}{_table(asset_dir, manifest, 8)}{_table(asset_dir, manifest, 9)}

<h1>6. Interval forecasts and trading evaluation</h1><p>Conformal intervals quantify forecast uncertainty using only residuals observed before the final test. Trading signals are formed at the forecast origin and scored against the next realized return. The uncertainty-adjusted strategy can avoid a trade when the interval is insufficiently decisive. This timing avoids the common error of selecting a signal after the realized return is known.</p>{_figure(asset_dir, manifest, 11)}{_figure(asset_dir, manifest, 12)}{_figure(asset_dir, manifest, 13)}{_table(asset_dir, manifest, 10)}{_table(asset_dir, manifest, 11, landscape=True)}{_table(asset_dir, manifest, 12, landscape=True)}

<h1>7. Discussion and limitations</h1><p>The empirical record does not support a blanket conclusion that a complex decomposed neural architecture dominates simple forecasts. Relative performance changes by company and metric, and transaction costs can alter a directional strategy's apparent attractiveness. This motivates the retained random-walk and ARIMA benchmarks, the interval evaluation, and the model-comparison tests.</p><p>The study uses closing prices and supplied daily proxy variables. It does not model liquidity, market impact, short-selling constraints, corporate actions beyond supplied histories, or asynchronous news availability. Factor coverage is frequency-dependent for emerging-market companies. These are limitations rather than evidence of absent risk.</p>

<h1>8. Conclusion</h1><p>This paper provides a transparent causal framework for critical-mineral equity forecasting and trading evaluation. Its central result is a traceable empirical record: every forecast, interval, model-comparison statistic, and cost-aware trade can be examined in the companion ledgers. The framework is therefore suitable for continued ablation, robustness testing, and peer review without relying on full-sample preprocessing.</p>

<h1>References</h1><section class="refs"><p>Diebold, F.X. and Mariano, R.S. (1995). Comparing predictive accuracy. <i>Journal of Business &amp; Economic Statistics</i>, 13(3), 253-263.</p><p>Dragomiretskiy, K. and Zosso, D. (2014). Variational mode decomposition. <i>IEEE Transactions on Signal Processing</i>, 62(3), 531-544.</p><p>Hochreiter, S. and Schmidhuber, J. (1997). Long short-term memory. <i>Neural Computation</i>, 9(8), 1735-1780.</p><p>Liu, S., Li, M., Yang, K., Wei, Y. and Wang, S. (2025). From forecasting to trading: A multimodal-data-driven approach to reversing carbon market losses. <i>Energy Economics</i>, 144, 108350. https://doi.org/10.1016/j.eneco.2025.108350.</p><p>Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. <i>Journal of the Royal Statistical Society: Series B</i>, 58(1), 267-288.</p></section>

<h1>Appendix: complete empirical ledgers and reproducibility</h1><section class="appendix"><p>The companion <i>appendix_ledgers</i> directory contains the complete unaggregated final-test forecast metrics, interval metrics, DM comparisons, trading metrics for every company/model/strategy/cost combination, and feature-selection records. The asset manifest records the causal run identifier, source checksums, captions, and the Crossref search snapshot used in Figure 1.</p><p class="page-note">The released-paper shadow is intentionally absent from the empirical ledgers and reported findings because it is a noncausal forensic reconstruction, not a valid forecast evaluation.</p></section>
</body></html>'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.assets, args.output))


if __name__ == "__main__":
    main()
