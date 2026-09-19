from pathlib import Path

readme = r"""# Adaptive Volatility Forecasting, Tail-Risk & Portfolio Decision Engine

**AVF-TRPDE** is a research-grade quantitative finance system for evaluating volatility forecasting models, latent market regimes, tail-risk measures, and risk-driven portfolio decisions through strict out-of-sample validation.

> **Research question:**  
> Does incorporating latent regime information into an ML volatility forecasting model provide statistically significant and economically meaningful improvement over established volatility models and a non-regime ML baseline?

---

## Project Status

The project is currently in the **core quantitative research pipeline construction stage**.

### Implemented and smoke-tested

- Data ingestion and validation
- Canonical dataset construction
- Dataset fingerprinting
- Realized volatility construction from intraday data
- Next-day volatility target construction
- Lagged features
- Rolling features
- Cross-asset features
- GARCH feature integration
- EWMA
- GJR-GARCH
- XGBoost
- HMM regime detection
- Regime-Aware XGBoost
- MAE
- RMSE
- QLIKE
- Diebold-Mariano test

### Remaining research pipeline

- Bootstrap confidence intervals
- Model Confidence Set (MCS)
- Walk-forward validation engine
- Filtered Historical Simulation
- VaR 95% / 99%
- Expected Shortfall
- Kupiec backtesting
- Christoffersen backtesting
- Portfolio risk targeting
- Position and turnover constraints
- Transaction costs
- Stress testing
- Cross-asset analysis
- Failure analysis
- Research conclusion
- API, schemas, and dashboard integration

Synthetic smoke-test outputs are used only to verify implementation correctness. They are **not research findings or model-performance claims**.

---

## Core Architecture

```text
USER / RESEARCHER
        ↓
APPLICATION INTERFACE
   FastAPI / React
        ↓
RESEARCH ORCHESTRATOR
        ↓
DATA INPUT LAYER
 CSV / Parquet
        ↓
SCHEMA VALIDATION
        ↓
CANONICAL DATASET
        ↓
DATASET FINGERPRINT
        ↓
REALIZED VOLATILITY ENGINE
        ↓
FEATURE ENGINEERING
        ↓
WALK-FORWARD ENGINE
        ↓
EWMA / GJR-GARCH / XGBoost
        ↓
HMM → Regime probabilities → Regime-XGBoost
        ↓
FORECAST EVALUATION
        ↓
DM / Bootstrap / MCS
        ↓
FHS → VaR / ES
        ↓
Kupiec / Christoffersen
        ↓
PORTFOLIO ENGINE
        ↓
Risk targeting → constraints → transaction costs
        ↓
Stress testing
        ↓
Cross-asset analysis
        ↓
Failure analysis
        ↓
Research verdict
        ↓
Reports / Dashboard
```

---

## Model Set

AVF-TRPDE uses the following locked model set:

### EWMA

Exponentially Weighted Moving Average volatility:

\[
\sigma_t^2 =
\lambda\sigma_{t-1}^2 +
(1-\lambda)r_{t-1}^2
\]

### GJR-GARCH

The GJR-GARCH model captures volatility clustering, persistence, and asymmetric response to negative returns:

\[
\sigma_t^2 =
\omega +
\alpha\epsilon_{t-1}^2 +
\gamma I_{t-1}\epsilon_{t-1}^2 +
\beta\sigma_{t-1}^2
\]

where:

\[
I_{t-1}=1
\]

when the previous innovation is negative.

### XGBoost

XGBoost forecasts next-period realized volatility using information available at time \(t\), including:

- Lagged returns
- Rolling volatility
- Volume
- Price range
- Momentum
- Cross-asset information
- GJR-GARCH forecast

### HMM

A Hidden Markov Model detects latent market regimes.

The regime detector produces:

\[
P(S_t=k\mid O_{1:t})
\]

where \(S_t\) represents the latent market regime.

The HMM is fitted inside each walk-forward training fold to prevent look-ahead leakage.

### Regime-Aware XGBoost

The regime-aware model extends XGBoost with HMM regime probabilities:

```text
Market features
      +
GJR-GARCH forecast
      +
HMM regime probabilities
      ↓
Regime-Aware XGBoost
```

---

## Ablation Design

The locked ablation framework contains:

```text
A0 = XGBoost

A1 = XGBoost + GARCH forecast

A2 = XGBoost + HMM probabilities

A3 = XGBoost + GARCH + HMM
```

This allows the contribution of GARCH information and regime information to be evaluated separately.

---

## Realized Volatility

Intraday returns are used to construct daily realized volatility.

For intraday log returns \(r_{t,j}\):

\[
RV_t = \sum_{j=1}^{M_t} r_{t,j}^2
\]

and:

\[
RVOL_t = \sqrt{RV_t}
\]

The next-day realized volatility is used as the machine-learning forecasting target:

\[
y_{t+1}=RVOL_{t+1}
\]

Intraday data is used for target construction, not for high-frequency trading.

---

## Forecast Evaluation

The project evaluates forecasts using:

### MAE

\[
MAE =
\frac{1}{N}
\sum_{t=1}^{N}
|y_t-\hat y_t|
\]

### RMSE

\[
RMSE =
\sqrt{
\frac{1}{N}
\sum_{t=1}^{N}
(y_t-\hat y_t)^2
}
\]

### QLIKE

\[
QLIKE =
\frac{1}{N}
\sum_{t=1}^{N}
\left(
\frac{y_t}{\hat y_t}
-
\log\frac{y_t}{\hat y_t}
-
1
\right)
\]

Lower forecast loss indicates better forecast performance for these metrics.

---

## Statistical Comparison

The project uses:

- Diebold-Mariano test
- Bootstrap confidence intervals
- Model Confidence Set

The Diebold-Mariano framework compares the loss differential between two forecasting models:

\[
d_t=L(e_{A,t})-L(e_{B,t})
\]

with:

\[
H_0:E[d_t]=0
\]

The purpose is to test whether forecast-performance differences are statistically distinguishable rather than relying only on point estimates.

---

## Tail Risk

Tail-risk estimation uses **Filtered Historical Simulation (FHS)**.

Standardized returns are calculated as:

\[
z_t=\frac{r_t}{\hat\sigma_t}
\]

The empirical standardized-return distribution is then combined with the current volatility forecast.

The project evaluates:

- VaR 95%
- VaR 99%
- Expected Shortfall 95%

Risk forecasts are backtested using:

- Kupiec unconditional coverage test
- Christoffersen independence / clustering test

---

## Portfolio Decision Layer

The portfolio layer evaluates:

- Static / Buy & Hold
- EWMA risk targeting
- GJR-GARCH risk targeting
- XGBoost risk targeting
- Regime-XGBoost risk targeting

Risk targeting follows:

\[
w_{i,t}
=
w_{i,base}
\frac{\sigma_{target}}
{\hat\sigma_{i,t}}
\]

with position constraints:

\[
|w_{i,t}|\leq w_{i,max}
\]

and turnover:

\[
Turnover_t
=
\sum_i |w_{i,t}-w_{i,t-1}|
\]

Transaction costs are incorporated as:

\[
TC_t=c\times Turnover_t
\]

and net portfolio return:

\[
R_t^{net}=R_t^{gross}-TC_t
\]

Portfolio evaluation includes:

- Return
- Volatility
- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- Calmar ratio
- VaR
- Expected Shortfall
- Turnover
- Transaction costs
- Gross P&L
- Net P&L

---

## Walk-Forward Validation

The project does **not** use a random train/test split for the research evaluation.

Each walk-forward fold follows:

```text
Historical training window
        ↓
Fit preprocessing
        ↓
Fit GJR-GARCH
        ↓
Generate GARCH features
        ↓
Fit HMM
        ↓
Generate regime probabilities
        ↓
Build XGBoost data
        ↓
Train XGBoost
        ↓
Train Regime-XGBoost
        ↓
Generate out-of-sample forecasts
        ↓
Move the window forward
        ↓
Repeat
```

The HMM is re-estimated inside each training fold.

No future information is allowed to enter preprocessing, GARCH estimation, HMM estimation, model training, or hyperparameter selection.

---

## Data Requirements

### Daily data

Minimum requirement:

- At least 8 years
- `timestamp`
- `asset_id`
- `open`
- `high`
- `low`
- `close`
- `volume`

### Intraday data

Minimum requirement:

- At least 3 years
- `timestamp`
- `asset_id`
- `open`
- `high`
- `low`
- `close`
- `volume`

Intraday data is primarily used to construct the realized-volatility target.

---

## Technology Stack

### Backend / Research

- Python 3.11+
- NumPy
- Pandas
- SciPy
- arch
- scikit-learn
- XGBoost
- hmmlearn

### Data

- CSV
- Parquet
- DuckDB

### API

- FastAPI
- Pydantic

### Frontend

- React
- TypeScript
- Vite
- Plotly

### Configuration / Development

- YAML
- pytest
- Black
- Ruff
- Git
- GitHub

---

## Project Structure

```text
adaptive-volatility-risk-engine/
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/
│   │   │   └── v1/
│   │   │       ├── datasets.py
│   │   │       ├── experiments.py
│   │   │       ├── forecasts.py
│   │   │       ├── risk.py
│   │   │       ├── portfolios.py
│   │   │       └── reports.py
│   │   ├── core/
│   │   ├── data/
│   │   │   ├── loaders/
│   │   │   │   ├── csv.py
│   │   │   │   └── parquet.py
│   │   │   ├── validation.py
│   │   │   ├── canonical.py
│   │   │   └── fingerprint.py
│   │   ├── research/
│   │   │   ├── returns/
│   │   │   ├── volatility/
│   │   │   ├── features/
│   │   │   ├── models/
│   │   │   ├── regimes/
│   │   │   ├── regime_models/
│   │   │   ├── evaluation/
│   │   │   └── walk_forward/
│   │   ├── risk/
│   │   ├── portfolio/
│   │   ├── stress/
│   │   ├── experiments/
│   │   └── schemas/
│   └── tests/
│       ├── unit/
│       ├── integration/
│       └── research/
│
├── frontend/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── intraday/
│   ├── realized/
│   └── samples/
├── configs/
├── experiments/
├── notebooks/
├── reports/
├── docs/
├── storage/
│   ├── duckdb/
│   └── metadata/
├── pyproject.toml
├── README.md
├── .env.example
├── .gitignore
└── LICENSE
```

---

## Data Provider

The current development ingestion uses **Twelve Data** for market OHLCV data.

The API key must be stored locally in `.env` and must not be committed to Git:

```text
TWELVE_DATA_API_KEY=your_api_key_here
```

The repository contains `.env.example` for configuration reference.

---

## Current Research Scope

The project is designed to answer whether latent regime information improves volatility forecasting and downstream risk decisions.

The research progression is:

```text
Model volatility
      ↓
Compare classical and ML forecasts
      ↓
Evaluate latent regimes
      ↓
Add regime information to ML
      ↓
Test statistical significance
      ↓
Convert forecasts to VaR / ES
      ↓
Backtest tail risk
      ↓
Evaluate portfolio decisions
      ↓
Include transaction costs
      ↓
Stress test
      ↓
Cross-asset validation
      ↓
Failure analysis
      ↓
Research conclusion
```

The project does **not** assume that Regime-XGBoost will outperform the baseline models. The final conclusion will be determined by the out-of-sample experimental results.

---

## Leakage Controls

Temporal integrity is a core requirement.

The following are fitted or generated using only information available at the relevant point in time:

- Preprocessing
- GJR-GARCH
- HMM
- HMM regime probabilities
- XGBoost
- Regime-XGBoost
- Hyperparameter selection
- Feature generation
- Portfolio weights

The HMM is never fitted on the full dataset before walk-forward evaluation.

---

## Development Status

Current completed research components:

```text
[✓] Data ingestion
[✓] Data validation
[✓] Canonical dataset
[✓] Dataset fingerprint
[✓] Realized volatility
[✓] Target construction
[✓] Lagged features
[✓] Rolling features
[✓] Cross-asset features
[✓] GARCH features
[✓] EWMA
[✓] GJR-GARCH
[✓] XGBoost
[✓] HMM
[✓] Regime-XGBoost
[✓] MAE
[✓] RMSE
[✓] QLIKE
[✓] Diebold-Mariano

[ ] Bootstrap
[ ] Model Confidence Set
[ ] Walk-forward splitter
[ ] Walk-forward fold
[ ] Walk-forward runner
[ ] Filtered Historical Simulation
[ ] VaR
[ ] Expected Shortfall
[ ] Kupiec test
[ ] Christoffersen test
[ ] Portfolio engine
[ ] Transaction costs
[ ] Stress testing
[ ] Cross-asset analysis
[ ] Failure analysis
[ ] Research verdict
[ ] API integration
[ ] Frontend dashboard
```

---

## Research Integrity

This project separates:

1. **Implementation verification**
2. **Out-of-sample empirical results**
3. **Statistical significance**
4. **Economic significance**
5. **Failure analysis**

Smoke-test outputs are not presented as evidence of model superiority.

The final research conclusion will be based on the locked walk-forward methodology, statistical tests, risk backtesting, portfolio evaluation, stress testing, and cross-asset validation.

---

## License

This project is licensed under the terms specified in `LICENSE`.
"""

path = Path("/mnt/data/README.md")
path.write_text(readme, encoding="utf-8")
print(path)
