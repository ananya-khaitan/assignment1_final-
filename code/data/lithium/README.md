# Lithium study input contract

This directory isolates the inputs for the Liu-style lithium-price study from
the original eight-company causal pipeline.

## Targets and structured market inputs

`prepare_lithium_inputs.py` reads the two source workbooks in `code/data/`, the
Albemarle and Ganfeng closes from the project equity workbook, the MVIS close,
and the existing Yahoo market controls.  It writes one model panel per target
under `processed/`.  All ablations use the same common sample, currently
2023-05-04 through 2026-03-12, because MVIS is the earliest-ending required
structured source.  This prevents performance comparisons across different
test periods.

## Required external files

`external/lithium_google_trends.csv` must contain one observation date plus the
five queries listed below, retrieved together in a single worldwide Google
Trends comparison so the 0-100 indices share the same normalization:

- lithium
- lithium price
- lithium carbonate
- lithium hydroxide
- lithium battery

Accepted date column names are `Date`, `date`, `Week`, or `time`.  Google
weekly observations are dated at the end of their seven-day measurement
window before being aligned to a target calendar.

`external/lithium_news_headlines.csv` must contain:

- `published_at_utc`
- `title`
- `source`
- `url`

Optional precomputed columns are `textblob_polarity` and `finbert_score`.  The
preparation script computes TextBlob polarity when it is absent.  A
publication-grade full run requires daily coverage spanning the model sample
and a non-missing FinBERT score; it never substitutes generic SF Fed sentiment.

## Protocol warning

The Liu-style experiment deliberately uses full-sample VMD, global scaling,
and a static 80:20 split for direct methodological comparability.  Outputs are
retrospective static predictions and must not be represented as real-time
causal forecasts.
