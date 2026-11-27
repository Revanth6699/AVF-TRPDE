# Adaptive Volatility Forecasting, Tail-Risk & Portfolio Decision Engine (AVF-TRPDE)

AVF-TRPDE is a quantitative finance and machine learning project focused on forecasting market volatility, identifying changing market regimes, measuring portfolio tail risk, and evaluating risk-targeted portfolio decisions.

The project brings together statistical volatility models, machine learning, regime detection, risk backtesting, and portfolio analysis in a single research workflow. The objective is to study how volatility forecasts and market conditions affect risk estimates and portfolio decisions across financial assets.

## Research Objective

Volatility is not constant over time. Financial markets move through periods of relatively stable conditions, elevated uncertainty, and sharp changes in risk. A model that performs well in one market environment may behave differently in another.

AVF-TRPDE investigates whether combining volatility forecasts with market-regime information can improve risk estimation and portfolio decision-making compared with simpler statistical baselines.

The focus is not just on producing a forecast. It is on evaluating the forecast, measuring the risk associated with it, and examining whether the resulting portfolio decisions remain defensible under different market conditions.

## Model Architecture

The project uses the following models and methods.

| Component | Model or method | Purpose |
|---|---|---|
| Volatility baseline | EWMA | Estimates changing volatility using exponentially weighted historical observations. |
| Conditional volatility | GJR-GARCH | Models volatility clustering and asymmetric responses to positive and negative returns. |
| Machine learning | XGBoost | Learns nonlinear relationships in engineered financial features. |
| Market regimes | Hidden Markov Model (HMM) | Estimates latent market states from observed financial data. |
| Regime-aware forecasting | Regime-Aware XGBoost | Incorporates estimated regime information into volatility forecasting. |
| Tail-risk measurement | VaR and Expected Shortfall | Quantifies portfolio losses at specified confidence levels and beyond the VaR threshold. |
| Risk backtesting | Kupiec and Christoffersen tests | Evaluates VaR exception frequency and the independence of exceptions. |
| Portfolio evaluation | Static and model-driven risk-targeted portfolios | Compares portfolio outcomes under defined sizing constraints and transaction costs. |

These components serve different purposes. EWMA and GJR-GARCH provide statistical volatility estimates, XGBoost models nonlinear relationships, and HMM estimates latent market regimes. The regime-aware model investigates whether regime information adds predictive value.

## Risk Measurement

The risk engine evaluates portfolio losses using Value at Risk and Expected Shortfall.

### Value at Risk (VaR)

VaR estimates a loss threshold at a selected confidence level over a specified horizon.

The project evaluates VaR at:

- 95% confidence level
- 99% confidence level

VaR is a quantile-based risk measure. It does not describe the magnitude of losses beyond the estimated threshold.

### Expected Shortfall (ES)

Expected Shortfall measures the average loss in the tail beyond the corresponding VaR threshold, subject to the project's loss convention and estimation method.

Together, VaR and ES provide complementary views of portfolio downside risk.

### VaR Backtesting

Two statistical tests are included in the risk-validation framework:

- **Kupiec test:** Evaluates whether the observed frequency of VaR violations is consistent with the expected exception rate.
- **Christoffersen test:** Evaluates the independence of VaR violations and helps identify clustering in exceptions.

Passing a backtest does not guarantee that a risk model will remain accurate under future market conditions. Results must be interpreted alongside the sample size, forecast horizon, and market environment.

## Portfolio Decision Engine

The portfolio component compares two approaches:

**Static portfolio**

A reference portfolio used to establish a consistent comparison point.

**Model-driven risk-targeted portfolio**

A portfolio whose sizing responds to model-derived risk estimates, subject to the project's mathematical constraints and transaction-cost assumptions.

The comparison examines how risk estimates translate into portfolio exposure and how changes in exposure affect realized portfolio outcomes.

Transaction costs are included because portfolio turnover can reduce the practical benefit of a strategy that appears attractive before costs.

The portfolio analysis is intended to evaluate model-driven decisions rather than assume that improved forecasts automatically produce better investment results.

## Validation Methodology

Financial time series require careful validation because observations are ordered in time and future information must not enter past predictions.

AVF-TRPDE uses a validation framework designed around the following principles.

### Walk-Forward Validation

Models are evaluated across successive chronological folds. Training uses information available before each evaluation period, and forecasts are assessed on subsequent observations.

### Regime Analysis

Forecast and risk behavior are examined across estimated market regimes to investigate whether performance varies with market conditions.

### Ablation Analysis

Individual model components are compared to assess whether regime information and other selected features contribute meaningful value.

### Cross-Asset Validation

Models are evaluated across multiple assets to examine whether results generalize beyond a single financial instrument.

### Stress Testing

Stress scenarios are used to examine portfolio and risk behavior under adverse market conditions. Stress-test outcomes are scenario-dependent and should not be interpreted as predictions of future losses.

### Statistical Significance

Forecast comparisons should distinguish observed performance differences from differences that are supported by appropriate statistical evidence.

## Preventing Data Leakage

Leakage is a central concern in financial forecasting. A model can appear accurate when its features, targets, or validation process contain information that would not have been available at prediction time.

The project addresses the following risks:

- **Chronological leakage:** Future observations must not enter training data for an earlier forecast.
- **Feature leakage:** Features must be constructed using information available at the forecast origin.
- **Regime-probability leakage:** HMM state probabilities used for forecasting must be estimated without access to future evaluation observations.
- **Target definition:** The realized-volatility target and its forecast horizon must be defined consistently with the intended prediction task.
- **Walk-forward estimation:** Model fitting and HMM re-estimation must respect the boundaries of each training fold.
- **Evaluation integrity:** Model selection must not use held-out test results as though they were available during training.

These controls are necessary for credible evaluation. Their effectiveness depends on the implementation and should be verified through tests and inspection of the complete pipeline.

## Data

The project uses historical financial time series for volatility estimation, regime analysis, risk measurement, and portfolio evaluation.

Depending on the configured data source and available instruments, relevant inputs may include:

- Historical asset prices
- Calculated returns
- Volatility-related features
- Cross-asset observations
- Market data used to construct model features

The exact assets, date ranges, data sources, and available observations should be taken from the project's configured datasets and data-ingestion code. Results should not be interpreted without documenting the data used to produce them.

## Technology Stack

The project is implemented in Python and uses quantitative analysis and machine learning libraries for model development, evaluation, and risk calculations.

The implementation includes model-specific components, shared utilities, a runner, and validation-related modules.

The exact package versions and installation requirements are maintained in the repository's dependency files.

## Project Status

**Backend status: research prototype; end-to-end validation is not complete.** The experiment API connects registered datasets to chronological walk-forward model execution and forecast evaluation. It supports per-asset MAE, RMSE, variance-based QLIKE, pairwise Diebold–Mariano comparisons, moving-block bootstrap confidence intervals, and Model Confidence Set analysis on common out-of-sample timestamps. The experiment executor also supports the locked A0–A3 ablations through the `parameters.ablations` setting. The /risk API generates FHS VaR95/VaR99/ES95 forecasts and exposes Kupiec/Christoffersen VaR backtests; the /forecasts API persists and evaluates model predictions. The /portfolios API now executes static, buy-and-hold, and covariance-based risk-targeted portfolios from persisted out-of-sample forecast records, applies position/turnover limits and transaction costs, persists return series and performance metrics, and compares strategies on common dates. The /stress API builds configured retrospective scenarios from persisted portfolio net returns and stores scenario results. The /reports API generates and persists evidence-based Markdown and JSON research reports from available experiment, forecast, risk, portfolio, and stress outputs. These integrations have automated synthetic regression coverage; they do not establish real-market validity or production readiness.

The research data resolver now distinguishes daily OHLCV from intraday OHLCV. Daily bars provide end-of-day features; intraday bars provide realized variance. A daily dataset must be paired with an overlapping intraday dataset to create next-trading-day realized-volatility labels. Intraday input is aggregated to one OHLCV row per asset and trading day before walk-forward evaluation. Provider history, overlap, data quality, and runtime behavior still require environment-specific verification.

The repository does not currently contain the full eight-year daily / three-year intraday research dataset. No empirical performance or risk conclusion is claimed.

## Running the Project

Clone the repository:

```bash
git clone https://github.com/Revanth6699/AVF-TRPDE.git
cd AVF-TRPDE
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the backend and research dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `TWELVE_DATA_API_KEY` only when using the Twelve Data provider. CSV and Parquet workflows do not require a provider key. Keep `.env` out of version control.

Run the API from the repository root:

```bash
uvicorn backend.app.main:app --reload
```

Run the isolated automated tests:

```bash
pytest
```

The live Twelve Data smoke tests in `backend/tests/test_data_download.py` require a valid provider key and are skipped when it is absent. The other legacy research smoke scripts are not collected by pytest.

The exact execution command depends on the repository's current runner, configuration, and data requirements. Do not assume that installing the dependencies alone makes every model or data pipeline ready to execute.

## Running an A0–A3 Ablation Experiment

Register an experiment with a registered dataset name and include the ablation list in its parameters. The `models` field remains required by the experiment schema; use `XGBoost` as the base learner for the locked ablation set.

Create the experiment with `POST /experiments`:

```json
{
  "name": "volatility-ablation",
  "description": "Walk-forward A0-A3 ablation comparison",
  "models": ["XGBoost"],
  "metrics": ["MAE", "RMSE", "QLIKE"],
  "risk_measures": ["VaR95", "VaR99", "ES95"],
  "parameters": {
    "dataset_name": "YOUR_REGISTERED_DATASET",
    "ablations": ["A0", "A1", "A2", "A3"],
    "train_size": 120,
    "test_size": 1,
    "step_size": 1,
    "bootstrap_samples": 1000,
    "confidence_level": 0.95
  },
  "metadata": {}
}
```

Then execute it with `POST /experiments/run`:

```json
{
  "experiment_names": ["volatility-ablation"]
}
```

The response contains executed ablations, fold and prediction counts, per-asset forecast metrics, pairwise DM results, confidence intervals, and MCS output when enough common out-of-sample observations are available. Configured risk measures are returned under `risk_measures_requested`; `risk_analysis.status` is explicitly `not_implemented` until VaR/ES estimation and backtesting are connected. The MCS calculation is not run when fewer than 10 common observations are available; the response reports that limitation rather than inventing a result. QLIKE is calculated on variance: the API receives volatility standard deviations and squares their ratio when computing the loss.

This workflow does not execute VaR/ES backtesting or portfolio analysis. Those remain separate backend work.
## Forecast Retrieval and Evaluation APIs

After a research experiment completes, use `POST /forecasts` to persist one model's out-of-sample forecasts from its Parquet artifact. The request uses the same experiment and dataset names, one model, assets, and an inclusive target-date range. Each returned forecast distinguishes the feature/origin date from the target date and includes the aligned actual return and realized volatility.

Use `POST /forecasts/evaluate` with the same request to calculate MAE, RMSE, and variance-based QLIKE separately for each asset. Use `POST /forecasts/compare` to compare every persisted model; omit `model` from this request. The comparison intersects the common asset/target-date observations first and reports metrics per model and asset, rather than pooling raw errors across different assets.

Forecast persistence can be repeated for the same experiment/model; the stored record is replaced. Artifacts created before the next-day return alignment fields were added must be regenerated by re-running the experiment.

## Running FHS Risk Analysis

Run the experiment first. Its prediction artifact must contain `target_timestamp` and `target_return`; re-run experiments created before this alignment contract was added. The forecast at feature date t predicts next-day realized volatility, so the FHS risk observation is aligned to the return at `target_timestamp` (t+1), not the return at t.

Create risk forecasts with `POST /risk`:

```json
{
  "experiment_name": "volatility-ablation",
  "dataset_name": "YOUR_REGISTERED_DATASET",
  "model": "A0",
  "assets": ["AAPL"],
  "confidence_levels": [0.95, 0.99],
  "start_timestamp": "2024-01-01T00:00:00",
  "end_timestamp": "2026-01-01T00:00:00"
}
```

The current FHS workflow uses an expanding history of prior out-of-sample standardized returns, independently per asset. It requires at least 100 prior out-of-sample observations before producing the first risk point and draws 10,000 empirical-residual simulations for each forecast. The current return is added to history only after that date's risk forecast has been produced. Output measures are VaR95, VaR99, and ES95.

Run VaR backtesting with `POST /risk/backtest` using the same request. The response includes per-asset Kupiec unconditional-coverage and Christoffersen independence/conditional-coverage diagnostics. It also returns warnings when the sample is short or the expected exception count is small; do not interpret those p-values as strong evidence when warnings are present.

Compare at least two completed model risk results with `POST /risk/compare`. The request uses the same experiment, dataset, assets, confidence levels, and date range; omit `model` to compare all stored models. The reported VaR/ES values are means over the common asset/date observations, not the latest point or a portfolio-level risk estimate.

Risk results are generated from the persisted experiment Parquet artifact. The full eight-year daily / three-year intraday market dataset has not yet been validated, so synthetic CI results are integration evidence only, not empirical risk findings.

## Portfolio Execution API

Persist forecasts first with POST /forecasts. Then call POST /portfolios with the forecast model, assets, target-date range, and constraints. Supported strategies are:
- static: equal-weight constant-mix weights, rebalanced subject to the turnover limit;
- buy_and_hold: equal-weight initial holdings with return-driven weight drift; if the position cap is breached, execution fails rather than silently rebalancing;
- risk_targeted: equal-weight base exposures scaled to the requested daily volatility target using model forecast volatilities and a historical correlation matrix estimated only from strictly prior out-of-sample returns. The first 20 dates are held in cash until enough prior observations exist.

The API applies a per-position cap, gross L1 turnover limit, and transaction cost per unit of gross turnover. Results include gross/net returns, turnover, costs, exposures, predicted portfolio volatility (risk-targeted only), compounded performance, Sharpe, Sortino, drawdown, Calmar, empirical VaR95/VaR99/ES95, and gross/net P&L. POST /portfolios/compare compares stored strategies over their common evaluation dates. These are historical backtest outputs, not live execution or investment recommendations.

Example request:

```json
{
  "experiment_name": "volatility-ablation",
  "dataset_name": "YOUR_REGISTERED_DATASET",
  "strategy": "risk_targeted",
  "model": "XGBoost",
  "assets": ["AAPL", "MSFT"],
  "start_timestamp": "2024-01-01T00:00:00",
  "end_timestamp": "2026-01-01T00:00:00",
  "volatility_target": 0.01,
  "max_position": 0.8,
  "max_turnover": 0.2,
  "transaction_cost": 0.0004,
  "covariance_window": 60
}
```

The volatility target is a daily standard-deviation target. The model must already have persisted forecast records for the requested experiment and dataset. Risk-targeted execution needs at least 20 prior complete out-of-sample return observations before it begins taking positions.

## Stress Testing API

Run POST /stress/run with an experiment name and a completed portfolio strategy. The service loads thresholds from configs/stress.yaml, evaluates normal-volatility, high-volatility, volatility-spike, and prolonged-drawdown subsets, and persists scenario metrics in DuckDB. GET /stress/{experiment_name} retrieves stored results; add the strategy query parameter when selecting one strategy.

Stress scenarios are retrospective descriptive selections from the observed portfolio return history. Percentile thresholds are calculated from the supplied sample. No historical date windows are assumed by default; any fixed windows must be added to the configuration before the evaluation protocol is run. Stress outputs are not forward predictions.

## Research Report Generation API

POST /reports/run generates Markdown and JSON artifacts under reports/{experiment_name}/ and persists the report sections in DuckDB. The report includes only outputs that actually exist for the selected experiment, dataset, and period. Missing forecasts, risk results, portfolio results, or stress results are reported as unavailable; the generator does not fill gaps with synthetic numbers. GET /reports/{experiment_name}?report_type=... retrieves the persisted report. POST /reports can register a report request before execution.

## Research Interpretation

The project is designed to evaluate several connected questions:

- How do EWMA, GJR-GARCH, and XGBoost compare when forecasting volatility?
- Does regime information improve forecasts beyond the selected baselines?
- Do forecast differences translate into meaningful changes in VaR and Expected Shortfall?
- Do risk-targeted portfolio decisions remain useful after transaction costs?
- How stable are the findings across assets, time periods, and market regimes?

These are evaluation questions, not predetermined conclusions. The answers must come from reproducible experiments and documented results.

## Limitations

- Historical financial relationships can change over time.
- Volatility estimates and regime classifications are uncertain.
- VaR and Expected Shortfall depend on the quality of the underlying forecasts and estimation assumptions.
- Backtesting results can be inconclusive when the evaluation sample contains few tail events.
- Portfolio performance depends on transaction costs, constraints, data quality, and execution assumptions.
- Performance on selected historical assets does not establish universal generalization.

No model accuracy, risk reduction, or investment return is claimed without supporting empirical results.

## Author

**Revanth Kumar**

GitHub: [@Revanth6699](https://github.com/Revanth6699)

## Disclaimer

AVF-TRPDE is a quantitative research and software engineering project for educational and analytical purposes. It is not financial advice, and its outputs should not be treated as guarantees of future market behavior or portfolio performance.
