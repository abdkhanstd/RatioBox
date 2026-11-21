# RatioBox

Code and results for **"RatioBox: Forecasting Shariah Screen Breaches under Threshold Rulebooks on Asset-Based Ratios"**.

RatioBox forecasts the joint distribution of the three financial ratios used by Shariah equity screens
(interest-bearing debt, cash plus interest-bearing securities, receivables plus cash, each over total assets)
with gradient-boosted quantile regressions and a Gaussian copula. The breach probability under any rulebook is the
forecast mass outside the rulebook's box in ratio space, so one model serves every rulebook, scores new or revised
thresholds without retraining, and gives ordered probabilities for nested rulebooks.

## Main results

Rolling-origin tests 2019-2025, macro average over screens, three seeds (`results/metrics.csv`, AUC):

| | published, 1 quarter | published, 1 year | unseen, 1 quarter | unseen, 1 year |
|---|---|---|---|---|
| Rulebook GBM (nearest rulebook for unseen screens) | 0.825 | 0.752 | 0.782 | 0.725 |
| Surface GBM (threshold-augmented, monotone) | 0.824 | 0.753 | 0.840 | 0.783 |
| **RatioBox** (trained on future ratios) | **0.825** | **0.754** | **0.848** | **0.788** |
| Refit GBM, relabelled and refitted on each unseen screen | | | 0.849 | 0.788 |

RatioBox never orders nested rulebooks incorrectly (`results/coherence.csv`). Ranking candidates by its one-year MSCI
breach probability lowers failures at the review one year later by 7.5% and failures at any review within the year by
2.9% relative to the MSCI 30% entry rule, similar to the boosted classifiers (within 0.2 percentage points)
(`results/decision_boot.csv`). Its one-year calibration error is higher than that of label-trained classifiers, and against
the holdings of the Wahed FTSE USA Shariah ETF all forecasters rank removals only weakly (`results/etf_validation.csv`).

## Data

All inputs are public SEC EDGAR bulk archives (company facts and submissions). The SEC asks automated clients to
declare a contact in the User-Agent header:

```bash
SEC_USER_AGENT="Your Name your.email@example.com" python src/download_sec.py
```

Values are used as first reported (earliest filing date), so restatements never reach earlier decisions.

## Reproduce

```bash
pip install -r requirements.txt
python run_all.py
```

| step | script | output |
|---|---|---|
| first-reported XBRL facts | `src/extract_xbrl.py` | `data/xbrl_facts.parquet` |
| firm metadata (SIC, entity type) | `src/extract_submissions.py` | `data/firms.parquet` |
| firm-quarter panel, screening ratios | `src/build_panel.py` | `data/panel.parquet` |
| decision-time features and future ratios | `src/features.py` | `data/tasks.parquet` |
| rolling-origin experiments (2019-2025, h = 1, 4, three seeds) | `src/run_experiments.py` | `results/preds/` |
| metrics, bootstrap tests, coherence, watch-list capture | `src/evaluate.py` | `results/*.csv` |
| admission decisions versus the 30% entry rule | `src/decision.py` | `results/decision*.csv` |
| robustness: tag changes, listed firms, attrition, quarter-block bootstrap, size of coherence violations | `src/robustness.py` | `results/robust_*.csv` |
| holdings of the Wahed FTSE USA Shariah ETF (Form N-PORT, needs `SEC_USER_AGENT`) | `src/download_etf.py` | `data_raw/etf/nport/` |
| removals from the fund's holdings versus the forecasts | `src/validate_etf.py` | `results/etf_*.csv` |
| empirical coverage of the per-ratio forecast intervals (needs `weights/`) | `src/interval_coverage.py` | `results/interval_coverage.csv` |
| training and scoring cost | `src/cost.py` | `results/cost.csv` |
| saved RatioBox models | `src/export_models.py` | `weights/` |

The forecasts of every method, test year, horizon and seed (`results/preds/*.parquet`) are attached to the GitHub
release as `predictions.zip`, so all tables can be recomputed with `src/evaluate.py` without rerunning the experiments.
`results/preds/*_qcop_earlier.parquet` holds the forecasts of an earlier RatioBox configuration (flat tails, 4,000 Monte
Carlo draws, 300 rounds, 31 leaves) that the paper reports as an ablation. They come from the run made with that
configuration and were renamed to the method `qcop-earlier`. The group `qcop_earlier` re-creates the configuration with the
current code. For one quarter ahead on the 2022 window, its per-screen AUC is within 0.003 of the saved forecasts
(slightly higher on seven of the eight screens) and its Brier score within 0.00013 (`src/check_reproduction.py`,
`results/check_qcop_earlier.csv`); the saved forecasts are the ones evaluated in the paper.

Methods: `src/models.py` (`QuantCopula` is RatioBox, `PointCopula` its ablation without conditional quantiles; `Surface`, `GBMDirect`, `LogitModel` and `Clim` are baselines).
Screens: `src/screens.py`.

## Weights

`weights/rolling/h{h}_Y{Y}/` holds the seed-0 RatioBox model of every rolling-origin window, and `weights/final/h{h}/`
a model trained on every decision whose future balance sheet is in the archive (RatioBox is trained on future ratios,
not on breach labels). Each folder has 45 LightGBM boosters (3 ratios x 15 quantile levels),
the copula correlation and `meta.json`. The weights are attached to the GitHub release as `weights.zip`.

Score passing firms of a quarter under any thresholds (0 switches a ratio off):

```bash
python src/score.py --weights weights/final/h4 --quarter 2026Q2 --debt 0.3333 --cash 0.3333 --rec 0.49
```

## Licence

MIT. The screen definitions emulate published methodologies for research only. They are not the official
constituent lists of any index provider.
