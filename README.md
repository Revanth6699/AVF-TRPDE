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

**Backend status: research prototype; end-to-end validation is not complete.** The experiment API connects registered datasets to chronological walk-forward model execution and forecast evaluation. It supports per-asset MAE, RMSE, variance-based QLIKE, pairwise Diebold–Mariano comparisons, moving-block bootstrap confidence intervals, and Model Confidence Set analysis on common out-of-sample timestamps. The experiment executor also supports the locked A0–A3 ablations through the `parameters.ablations` setting. Separate risk backtesting/comparison, portfolio comparison, report-generation, and complete stress API workflows remain incomplete. Do not treat module presence or a synthetic CI test as proof of full production readiness.

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
