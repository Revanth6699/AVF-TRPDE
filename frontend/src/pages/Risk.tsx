import {
  AlertCircle,
  BarChart3,
  CheckCircle2,
  ChevronDown,
  CircleDollarSign,
  Database,
  FlaskConical,
  Play,
  RefreshCw,
  Search,
  ShieldAlert,
  SlidersHorizontal,
  Target,
  X,
} from "lucide-react";
import { FormEvent, useMemo, useState } from "react";
import { apiClient } from "../api/client";

type RiskPoint = {
  timestamp: string;
  asset_id: string;
  var_95: number;
  var_99: number;
  es_95: number;
};

type RiskResponse = {
  experiment_name: string;
  dataset_name: string;
  model: string;
  risk_points: RiskPoint[];
  observation_count: number;
  asset_count: number;
  status: string;
};

type RiskBacktestResult = {
  model: string;
  confidence_level: number;
  measure: string;
  observations: number;
  violations: number;
  violation_rate: number;
  statistic: number | null;
  p_value: number | null;
};

type RiskBacktestResponse = {
  experiment_name: string;
  results: RiskBacktestResult[];
  observation_count: number;
  asset_count: number;
};

type RiskComparisonItem = {
  model: string;
  var_95: number;
  var_99: number;
  es_95: number;
};

type RiskComparisonResponse = {
  experiment_name: string;
  models: RiskComparisonItem[];
  observation_count: number;
  asset_count: number;
};

type RiskRequest = {
  experiment_name: string;
  dataset_name: string;
  model: string;
  assets: string[];
  confidence_levels: number[];
  start_timestamp: string;
  end_timestamp: string;
};

const MODELS = [
  "EWMA",
  "GJR-GARCH",
  "XGBoost",
  "Regime-XGBoost",
];

const CONFIDENCE_LEVELS = ["95%", "99%"];

function formatNumber(value: number | null | undefined, digits = 6) {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "—";
  }

  return value.toFixed(digits);
}

function formatPercent(value: number | null | undefined, digits = 2) {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "—";
  }

  return `${(value * 100).toFixed(digits)}%`;
}

function formatTimestamp(value: string) {
  if (!value) return "—";

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return date.toLocaleString();
}

function getInitialRequest(): RiskRequest {
  return {
    experiment_name: "",
    dataset_name: "",
    model: "EWMA",
    assets: [],
    confidence_levels: [0.95, 0.99],
    start_timestamp: "",
    end_timestamp: "",
  };
}

export default function Risk() {
  const [request, setRequest] = useState<RiskRequest>(
    getInitialRequest(),
  );

  const [assetInput, setAssetInput] = useState("");

  const [riskResult, setRiskResult] =
    useState<RiskResponse | null>(null);

  const [backtestResult, setBacktestResult] =
    useState<RiskBacktestResponse | null>(null);

  const [comparisonResult, setComparisonResult] =
    useState<RiskComparisonResponse | null>(null);

  const [loading, setLoading] = useState(false);
  const [backtesting, setBacktesting] = useState(false);
  const [comparing, setComparing] = useState(false);

  const [error, setError] = useState<string | null>(null);
  const [backtestError, setBacktestError] =
    useState<string | null>(null);
  const [comparisonError, setComparisonError] =
    useState<string | null>(null);

  const [showConfig, setShowConfig] = useState(false);

  const [search, setSearch] = useState("");

  const filteredRiskPoints = useMemo(() => {
    if (!riskResult?.risk_points) {
      return [];
    }

    const query = search.trim().toLowerCase();

    if (!query) {
      return riskResult.risk_points;
    }

    return riskResult.risk_points.filter((point) => {
      return (
        point.asset_id.toLowerCase().includes(query) ||
        point.timestamp.toLowerCase().includes(query)
      );
    });
  }, [riskResult, search]);

  const updateRequest = <K extends keyof RiskRequest>(
    key: K,
    value: RiskRequest[K],
  ) => {
    setRequest((current) => ({
      ...current,
      [key]: value,
    }));
  };

  const addAsset = () => {
    const asset = assetInput.trim().toUpperCase();

    if (!asset) {
      return;
    }

    if (request.assets.includes(asset)) {
      setAssetInput("");
      return;
    }

    setRequest((current) => ({
      ...current,
      assets: [...current.assets, asset],
    }));

    setAssetInput("");
  };

  const removeAsset = (asset: string) => {
    setRequest((current) => ({
      ...current,
      assets: current.assets.filter(
        (currentAsset) => currentAsset !== asset,
      ),
    }));
  };

  const toggleConfidence = (level: number) => {
    setRequest((current) => {
      const exists = current.confidence_levels.includes(level);

      if (exists) {
        return {
          ...current,
          confidence_levels:
            current.confidence_levels.filter(
              (item) => item !== level,
            ),
        };
      }

      return {
        ...current,
        confidence_levels: [
          ...current.confidence_levels,
          level,
        ].sort(),
      };
    });
  };

  const createRiskAnalysis = async (
    event: FormEvent<HTMLFormElement>,
  ) => {
    event.preventDefault();

    setError(null);
    setRiskResult(null);
    setBacktestResult(null);
    setComparisonResult(null);

    if (!request.experiment_name.trim()) {
      setError("Experiment name is required.");
      return;
    }

    if (!request.dataset_name.trim()) {
      setError("Dataset name is required.");
      return;
    }

    if (!request.assets.length) {
      setError("Add at least one asset.");
      return;
    }

    if (!request.start_timestamp) {
      setError("Start timestamp is required.");
      return;
    }

    if (!request.end_timestamp) {
      setError("End timestamp is required.");
      return;
    }

    if (!request.confidence_levels.length) {
      setError("Select at least one confidence level.");
      return;
    }

    setLoading(true);

    try {
      const response = await apiClient.post<RiskResponse>(
        "/risk",
        request,
      );

      setRiskResult(response.data);
      setShowConfig(false);
    } catch (requestError: any) {
      const detail =
        requestError?.response?.data?.detail ??
        requestError?.message ??
        "Unable to create the risk analysis.";

      setError(String(detail));
    } finally {
      setLoading(false);
    }
  };

  const loadRiskAnalysis = async () => {
    if (!request.experiment_name.trim()) {
      setError("Enter an experiment name first.");
      return;
    }

    setError(null);
    setLoading(true);

    try {
      const params = request.model
        ? { model: request.model }
        : undefined;

      const response = await apiClient.get<RiskResponse>(
        `/risk/${encodeURIComponent(
          request.experiment_name.trim(),
        )}`,
        {
          params,
        },
      );

      setRiskResult(response.data);
      setBacktestResult(null);
      setComparisonResult(null);
    } catch (requestError: any) {
      const detail =
        requestError?.response?.data?.detail ??
        requestError?.message ??
        "Unable to load the risk analysis.";

      setError(String(detail));
      setRiskResult(null);
    } finally {
      setLoading(false);
    }
  };

  const runBacktest = async () => {
    setBacktestError(null);

    if (!request.experiment_name.trim()) {
      setBacktestError("Enter an experiment name first.");
      return;
    }

    if (!request.dataset_name.trim()) {
      setBacktestError("Enter a dataset name first.");
      return;
    }

    if (!request.assets.length) {
      setBacktestError("Add at least one asset first.");
      return;
    }

    if (!request.start_timestamp || !request.end_timestamp) {
      setBacktestError(
        "Start and end timestamps are required.",
      );
      return;
    }

    setBacktesting(true);

    try {
      const response =
        await apiClient.post<RiskBacktestResponse>(
          "/risk/backtest",
          request,
        );

      setBacktestResult(response.data);
    } catch (requestError: any) {
      const detail =
        requestError?.response?.data?.detail ??
        requestError?.message ??
        "Risk backtesting is not available for this analysis.";

      setBacktestError(String(detail));
      setBacktestResult(null);
    } finally {
      setBacktesting(false);
    }
  };

  const compareModels = async () => {
    setComparisonError(null);

    if (!request.experiment_name.trim()) {
      setComparisonError(
        "Enter an experiment name first.",
      );
      return;
    }

    if (!request.dataset_name.trim()) {
      setComparisonError(
        "Enter a dataset name first.",
      );
      return;
    }

    if (!request.assets.length) {
      setComparisonError("Add at least one asset first.");
      return;
    }

    if (!request.start_timestamp || !request.end_timestamp) {
      setComparisonError(
        "Start and end timestamps are required.",
      );
      return;
    }

    setComparing(true);

    try {
      const response =
        await apiClient.post<RiskComparisonResponse>(
          "/risk/compare",
          request,
        );

      setComparisonResult(response.data);
    } catch (requestError: any) {
      const detail =
        requestError?.response?.data?.detail ??
        requestError?.message ??
        "Risk comparison is not available yet.";

      setComparisonError(String(detail));
      setComparisonResult(null);
    } finally {
      setComparing(false);
    }
  };

  const hasRiskData =
    Boolean(riskResult?.risk_points?.length);

  return (
    <div className="page risk-page">
      {/* =====================================================
          HEADER
         ===================================================== */}

      <section className="risk-page-header">
        <div className="risk-header-copy">
          <div className="page-eyebrow">
            TAIL-RISK RESEARCH
          </div>

          <h1>Risk Engine</h1>

          <p>
            Inspect filtered historical simulation, Value at
            Risk, Expected Shortfall and statistical
            backtesting without introducing results that have
            not been produced by the research pipeline.
          </p>
        </div>

        <div className="risk-header-actions">
          <button
            className="secondary-button"
            type="button"
            onClick={loadRiskAnalysis}
            disabled={loading}
          >
            <RefreshCw size={15} />
            {loading ? "Loading..." : "Load analysis"}
          </button>

          <button
            className="primary-button"
            type="button"
            onClick={() => setShowConfig(true)}
          >
            <SlidersHorizontal size={16} />
            Configure risk
          </button>
        </div>
      </section>

      {/* =====================================================
          SUMMARY CARDS
         ===================================================== */}

      <section className="risk-summary-grid">
        <div className="risk-summary-card">
          <div className="risk-summary-icon risk-icon-cyan">
            <ShieldAlert size={18} />
          </div>

          <div>
            <span>STATUS</span>
            <strong>
              {riskResult?.status ?? "—"}
            </strong>
          </div>
        </div>

        <div className="risk-summary-card">
          <div className="risk-summary-icon risk-icon-violet">
            <BarChart3 size={18} />
          </div>

          <div>
            <span>OBSERVATIONS</span>
            <strong>
              {riskResult
                ? riskResult.observation_count
                : "—"}
            </strong>
          </div>
        </div>

        <div className="risk-summary-card">
          <div className="risk-summary-icon risk-icon-blue">
            <Database size={18} />
          </div>

          <div>
            <span>ASSETS</span>
            <strong>
              {riskResult
                ? riskResult.asset_count
                : request.assets.length || "—"}
            </strong>
          </div>
        </div>

        <div className="risk-summary-card">
          <div className="risk-summary-icon risk-icon-green">
            <Target size={18} />
          </div>

          <div>
            <span>MODEL</span>
            <strong>
              {riskResult?.model ?? request.model}
            </strong>
          </div>
        </div>
      </section>

      {/* =====================================================
          ERROR
         ===================================================== */}

      {error && (
        <div className="risk-alert risk-alert-error">
          <AlertCircle size={17} />

          <div>
            <strong>Risk analysis unavailable</strong>
            <span>{error}</span>
          </div>

          <button
            type="button"
            onClick={() => setError(null)}
            aria-label="Dismiss error"
          >
            <X size={15} />
          </button>
        </div>
      )}

      {/* =====================================================
          MAIN RESEARCH AREA
         ===================================================== */}

      <section className="risk-main-grid">
        {/* Risk trajectory / observation area */}

        <div className="risk-observation-panel">
          <div className="risk-panel-header">
            <div>
              <div className="section-kicker">
                FILTERED HISTORICAL SIMULATION
              </div>

              <h2>Tail-risk observations</h2>
            </div>

            {hasRiskData && (
              <div className="risk-panel-count">
                {filteredRiskPoints.length} observations
              </div>
            )}
          </div>

          {hasRiskData ? (
            <>
              <div className="risk-observation-toolbar">
                <div className="risk-search">
                  <Search size={16} />

                  <input
                    type="text"
                    value={search}
                    onChange={(event) =>
                      setSearch(event.target.value)
                    }
                    placeholder="Search asset or timestamp..."
                  />
                </div>
              </div>

              <div className="risk-table-wrapper">
                <table className="risk-table">
                  <thead>
                    <tr>
                      <th>Timestamp</th>
                      <th>Asset</th>
                      <th>VaR 95%</th>
                      <th>VaR 99%</th>
                      <th>ES 95%</th>
                    </tr>
                  </thead>

                  <tbody>
                    {filteredRiskPoints.map(
                      (point, index) => (
                        <tr
                          key={`${point.timestamp}-${point.asset_id}-${index}`}
                        >
                          <td>
                            {formatTimestamp(
                              point.timestamp,
                            )}
                          </td>

                          <td>
                            <span className="risk-asset-badge">
                              {point.asset_id}
                            </span>
                          </td>

                          <td>
                            {formatNumber(point.var_95)}
                          </td>

                          <td>
                            {formatNumber(point.var_99)}
                          </td>

                          <td>
                            {formatNumber(point.es_95)}
                          </td>
                        </tr>
                      ),
                    )}
                  </tbody>
                </table>
              </div>
            </>
          ) : (
            <div className="risk-empty-state">
              <div className="risk-empty-icon">
                <ShieldAlert size={23} />
              </div>

              <div>
                <h3>No risk observations loaded</h3>

                <p>
                  Configure an experiment, dataset, model and
                  asset range, then load or create the risk
                  analysis.
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Research context */}

        <aside className="risk-context-panel">
          <div className="risk-context-header">
            <div className="risk-context-icon">
              <CircleDollarSign size={18} />
            </div>

            <div>
              <div className="section-kicker">
                RISK CONTEXT
              </div>

              <h2>Research specification</h2>
            </div>
          </div>

          <div className="risk-context-list">
            <div>
              <span>Experiment</span>
              <strong>
                {riskResult?.experiment_name ||
                  request.experiment_name ||
                  "—"}
              </strong>
            </div>

            <div>
              <span>Dataset</span>
              <strong>
                {riskResult?.dataset_name ||
                  request.dataset_name ||
                  "—"}
              </strong>
            </div>

            <div>
              <span>Model</span>
              <strong>
                {riskResult?.model ||
                  request.model ||
                  "—"}
              </strong>
            </div>

            <div>
              <span>Assets</span>
              <strong>
                {riskResult?.asset_count ||
                  request.assets.length ||
                  "—"}
              </strong>
            </div>

            <div>
              <span>Confidence</span>
              <strong>
                {request.confidence_levels.length
                  ? request.confidence_levels
                      .sort()
                      .map(
                        (level) =>
                          `${level * 100}%`,
                      )
                      .join(" · ")
                  : "—"}
              </strong>
            </div>
          </div>

          <div className="risk-method-stack">
            <div className="risk-method-item">
              <span className="risk-method-dot" />
              FHS
            </div>

            <div className="risk-method-item">
              <span className="risk-method-dot" />
              VaR 95%
            </div>

            <div className="risk-method-item">
              <span className="risk-method-dot" />
              VaR 99%
            </div>

            <div className="risk-method-item">
              <span className="risk-method-dot" />
              ES 95%
            </div>
          </div>

          <div className="risk-integrity-note">
            <CheckCircle2 size={16} />

            <span>
              Risk observations are displayed exactly as
              returned by the backend.
            </span>
          </div>
        </aside>
      </section>

      {/* =====================================================
          BACKTESTING
         ===================================================== */}

      <section className="risk-research-section">
        <div className="risk-section-header">
          <div>
            <div className="section-kicker">
              STATISTICAL VALIDATION
            </div>

            <h2>Risk backtesting</h2>

            <p>
              Evaluate VaR coverage using the configured
              Kupiec and Christoffersen research contracts.
            </p>
          </div>

          <button
            className="secondary-button"
            type="button"
            onClick={runBacktest}
            disabled={backtesting}
          >
            <Play size={15} />
            {backtesting
              ? "Running..."
              : "Run backtest"}
          </button>
        </div>

        {backtestError && (
          <div className="risk-alert risk-alert-warning">
            <AlertCircle size={17} />

            <div>
              <strong>Backtest unavailable</strong>
              <span>{backtestError}</span>
            </div>
          </div>
        )}

        {backtestResult?.results?.length ? (
          <div className="risk-backtest-grid">
            {backtestResult.results.map(
              (result, index) => (
                <article
                  className="risk-backtest-card"
                  key={`${result.model}-${result.measure}-${index}`}
                >
                  <div className="risk-backtest-top">
                    <span>{result.measure}</span>

                    <strong>
                      {result.confidence_level * 100}%
                    </strong>
                  </div>

                  <div className="risk-backtest-model">
                    {result.model}
                  </div>

                  <div className="risk-backtest-stats">
                    <div>
                      <span>Observations</span>
                      <strong>
                        {result.observations}
                      </strong>
                    </div>

                    <div>
                      <span>Violations</span>
                      <strong>
                        {result.violations}
                      </strong>
                    </div>

                    <div>
                      <span>Violation rate</span>
                      <strong>
                        {formatPercent(
                          result.violation_rate,
                        )}
                      </strong>
                    </div>

                    <div>
                      <span>Statistic</span>
                      <strong>
                        {formatNumber(
                          result.statistic,
                        )}
                      </strong>
                    </div>

                    <div>
                      <span>P-value</span>
                      <strong>
                        {formatNumber(
                          result.p_value,
                        )}
                      </strong>
                    </div>
                  </div>
                </article>
              ),
            )}
          </div>
        ) : (
          <div className="risk-empty-strip">
            <FlaskConical size={19} />

            <div>
              <strong>
                No backtesting results loaded
              </strong>

              <span>
                Results will appear here only when the
                backend returns actual statistical
                backtesting output.
              </span>
            </div>
          </div>
        )}
      </section>

      {/* =====================================================
          MODEL COMPARISON
         ===================================================== */}

      <section className="risk-research-section">
        <div className="risk-section-header">
          <div>
            <div className="section-kicker">
              MODEL EVALUATION
            </div>

            <h2>Risk comparison</h2>

            <p>
              Compare VaR and Expected Shortfall across the
              models registered for the experiment.
            </p>
          </div>

          <button
            className="secondary-button"
            type="button"
            onClick={compareModels}
            disabled={comparing}
          >
            <BarChart3 size={15} />

            {comparing
              ? "Comparing..."
              : "Compare models"}
          </button>
        </div>

        {comparisonError && (
          <div className="risk-alert risk-alert-warning">
            <AlertCircle size={17} />

            <div>
              <strong>
                Comparison unavailable
              </strong>

              <span>{comparisonError}</span>
            </div>
          </div>
        )}

        {comparisonResult?.models?.length ? (
          <div className="risk-comparison-table-wrapper">
            <table className="risk-comparison-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>VaR 95%</th>
                  <th>VaR 99%</th>
                  <th>ES 95%</th>
                </tr>
              </thead>

              <tbody>
                {comparisonResult.models.map(
                  (model) => (
                    <tr key={model.model}>
                      <td>
                        <strong>{model.model}</strong>
                      </td>

                      <td>
                        {formatNumber(model.var_95)}
                      </td>

                      <td>
                        {formatNumber(model.var_99)}
                      </td>

                      <td>
                        {formatNumber(model.es_95)}
                      </td>
                    </tr>
                  ),
                )}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="risk-empty-strip">
            <BarChart3 size={19} />

            <div>
              <strong>
                Comparison is research-backed
              </strong>

              <span>
                VaR 95%, VaR 99% and ES 95% will appear
                here only when the backend comparison
                pipeline returns real results.
              </span>
            </div>
          </div>
        )}
      </section>

      {/* =====================================================
          CONFIGURATION MODAL
         ===================================================== */}

      {showConfig && (
        <div
          className="modal-backdrop"
          role="presentation"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) {
              setShowConfig(false);
            }
          }}
        >
          <form
            className="modal risk-config-modal"
            onSubmit={createRiskAnalysis}
          >
            <div className="modal-header">
              <div>
                <div className="page-eyebrow">
                  RISK CONFIGURATION
                </div>

                <h2>Configure risk analysis</h2>

                <p>
                  Define the research context used by the
                  tail-risk API.
                </p>
              </div>

              <button
                type="button"
                className="modal-close"
                onClick={() => setShowConfig(false)}
                aria-label="Close configuration"
              >
                <X size={18} />
              </button>
            </div>

            <div className="risk-config-body">
              <div className="risk-config-grid">
                <label className="form-field">
                  <span>Experiment name</span>

                  <input
                    value={request.experiment_name}
                    onChange={(event) =>
                      updateRequest(
                        "experiment_name",
                        event.target.value,
                      )
                    }
                    placeholder="e.g. AVF_TEST_001"
                  />
                </label>

                <label className="form-field">
                  <span>Dataset name</span>

                  <input
                    value={request.dataset_name}
                    onChange={(event) =>
                      updateRequest(
                        "dataset_name",
                        event.target.value,
                      )
                    }
                    placeholder="e.g. NIFTY50"
                  />
                </label>

                <label className="form-field">
                  <span>Model</span>

                  <div className="select-wrapper">
                    <select
                      value={request.model}
                      onChange={(event) =>
                        updateRequest(
                          "model",
                          event.target.value,
                        )
                      }
                    >
                      {MODELS.map((model) => (
                        <option
                          value={model}
                          key={model}
                        >
                          {model}
                        </option>
                      ))}
                    </select>

                    <ChevronDown size={15} />
                  </div>
                </label>

                <div className="form-field">
                  <span>Confidence levels</span>

                  <div className="risk-toggle-row">
                    {CONFIDENCE_LEVELS.map(
                      (label) => {
                        const value =
                          Number.parseInt(
                            label,
                            10,
                          ) / 100;

                        const active =
                          request.confidence_levels.includes(
                            value,
                          );

                        return (
                          <button
                            type="button"
                            key={label}
                            className={`risk-toggle ${
                              active
                                ? "risk-toggle-active"
                                : ""
                            }`}
                            onClick={() =>
                              toggleConfidence(
                                value,
                              )
                            }
                          >
                            {label}
                          </button>
                        );
                      },
                    )}
                  </div>
                </div>

                <label className="form-field">
                  <span>Start timestamp</span>

                  <input
                    type="datetime-local"
                    value={request.start_timestamp}
                    onChange={(event) =>
                      updateRequest(
                        "start_timestamp",
                        event.target.value,
                      )
                    }
                  />
                </label>

                <label className="form-field">
                  <span>End timestamp</span>

                  <input
                    type="datetime-local"
                    value={request.end_timestamp}
                    onChange={(event) =>
                      updateRequest(
                        "end_timestamp",
                        event.target.value,
                      )
                    }
                  />
                </label>
              </div>

              <div className="form-field">
                <span>Assets</span>

                <div className="asset-input-row">
                  <input
                    value={assetInput}
                    onChange={(event) =>
                      setAssetInput(event.target.value)
                    }
                    onKeyDown={(event) => {
                      if (event.key === "Enter") {
                        event.preventDefault();
                        addAsset();
                      }
                    }}
                    placeholder="e.g. NIFTY50"
                  />

                  <button
                    type="button"
                    className="secondary-button"
                    onClick={addAsset}
                  >
                    Add
                  </button>
                </div>

                {request.assets.length > 0 && (
                  <div className="asset-chip-list">
                    {request.assets.map((asset) => (
                      <span
                        className="asset-chip"
                        key={asset}
                      >
                        {asset}

                        <button
                          type="button"
                          onClick={() =>
                            removeAsset(asset)
                          }
                          aria-label={`Remove ${asset}`}
                        >
                          <X size={12} />
                        </button>
                      </span>
                    ))}
                  </div>
                )}
              </div>

              <div className="risk-config-note">
                <ShieldAlert size={16} />

                <span>
                  The API registers the requested risk
                  analysis. Actual FHS, VaR, ES and
                  statistical backtesting outputs are shown
                  only when produced by the backend research
                  pipeline.
                </span>
              </div>
            </div>

            <div className="modal-footer">
              <button
                type="button"
                className="secondary-button"
                onClick={() => setShowConfig(false)}
              >
                Cancel
              </button>

              <button
                type="submit"
                className="primary-button"
                disabled={loading}
              >
                <Play size={15} />

                {loading
                  ? "Registering..."
                  : "Create risk analysis"}
              </button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
}