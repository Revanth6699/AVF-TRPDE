import {
  Activity,
  AlertCircle,
  BarChart3,
  CheckCircle2,
  ChevronDown,
  Clock3,
  Database,
  Gauge,
  LineChart,
  Play,
  RefreshCw,
  Search,
  SlidersHorizontal,
  Sparkles,
  Target,
  TrendingUp,
  X,
  Zap,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import {
  compareForecasts,
  getForecast,
  type ForecastComparisonResponse,
  type ForecastPoint,
  type ForecastResponse,
} from "../api/forecasts";

const MODELS = [
  {
    key: "EWMA",
    label: "EWMA",
    description: "Exponentially weighted volatility baseline",
    tone: "cyan",
  },
  {
    key: "GJR-GARCH",
    label: "GJR-GARCH",
    description: "Asymmetric conditional volatility model",
    tone: "violet",
  },
  {
    key: "XGBoost",
    label: "XGBoost",
    description: "Machine-learning volatility forecaster",
    tone: "blue",
  },
  {
    key: "Regime-XGBoost",
    label: "Regime-XGBoost",
    description: "Regime-aware ML volatility forecaster",
    tone: "green",
  },
];

function formatNumber(value: number, digits = 6) {
  if (!Number.isFinite(value)) {
    return "—";
  }

  return value.toFixed(digits);
}

function formatDate(value: string) {
  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return date.toLocaleDateString(undefined, {
    day: "2-digit",
    month: "short",
    year: "numeric",
  });
}

function formatDateTime(value: string) {
  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return date.toLocaleString(undefined, {
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function getErrorMessage(error: unknown) {
  const candidate = error as {
    response?: {
      data?: {
        detail?: string;
      };
    };
    message?: string;
  };

  return (
    candidate?.response?.data?.detail ??
    candidate?.message ??
    "Unable to load forecast data."
  );
}

function StatusBadge({ status }: { status: string }) {
  const normalized = status.toLowerCase();

  const className =
    normalized === "completed"
      ? "forecast-status forecast-status-success"
      : normalized === "accepted"
        ? "forecast-status forecast-status-info"
        : "forecast-status";

  return (
    <span className={className}>
      <span className="forecast-status-dot" />
      {status}
    </span>
  );
}

function MetricCard({
  label,
  value,
  icon: Icon,
  tone,
}: {
  label: string;
  value: string;
  icon: typeof Gauge;
  tone: string;
}) {
  return (
    <div className={`forecast-metric-card forecast-metric-${tone}`}>
      <div className="forecast-metric-icon">
        <Icon size={18} strokeWidth={1.7} />
      </div>

      <div className="forecast-metric-copy">
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
    </div>
  );
}

function EmptyForecastState({
  title,
  description,
}: {
  title: string;
  description: string;
}) {
  return (
    <div className="forecast-empty-state">
      <div className="forecast-empty-icon">
        <LineChart size={24} strokeWidth={1.5} />
      </div>

      <div>
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

function ForecastChart({
  points,
  model,
}: {
  points: ForecastPoint[];
  model: string;
}) {
  const validPoints = points.filter(
    (point) =>
      Number.isFinite(point.predicted_volatility) &&
      Number.isFinite(new Date(point.timestamp).getTime()),
  );

  if (!validPoints.length) {
    return (
      <EmptyForecastState
        title="No forecast observations"
        description="The backend has not produced forecast points for this model yet."
      />
    );
  }

  const values = validPoints.flatMap((point) => [
    point.predicted_volatility,
    ...(point.actual_volatility === null
      ? []
      : [point.actual_volatility]),
  ]);

  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;

  const width = 1000;
  const height = 300;
  const paddingX = 32;
  const paddingY = 28;

  const x = (index: number) =>
    paddingX +
    (index / Math.max(validPoints.length - 1, 1)) *
      (width - paddingX * 2);

  const y = (value: number) =>
    height -
    paddingY -
    ((value - min) / range) * (height - paddingY * 2);

  const predictedPath = validPoints
    .map((point, index) => {
      const command = index === 0 ? "M" : "L";

      return `${command} ${x(index).toFixed(2)} ${y(
        point.predicted_volatility,
      ).toFixed(2)}`;
    })
    .join(" ");

  const actualPoints = validPoints.filter(
    (point) => point.actual_volatility !== null,
  );

  const actualPath =
    actualPoints.length > 1
      ? actualPoints
          .map((point, index) => {
            const originalIndex = validPoints.indexOf(point);
            const command = index === 0 ? "M" : "L";

            return `${command} ${x(originalIndex).toFixed(
              2,
            )} ${y(point.actual_volatility ?? 0).toFixed(2)}`;
          })
          .join(" ")
      : "";

  return (
    <div className="forecast-chart">
      <div className="forecast-chart-header">
        <div>
          <div className="forecast-chart-kicker">Forecast trajectory</div>
          <h3>{model}</h3>
        </div>

        <div className="forecast-chart-legend">
          <span>
            <i className="forecast-legend-predicted" />
            Predicted
          </span>

          {actualPoints.length > 0 && (
            <span>
              <i className="forecast-legend-actual" />
              Actual
            </span>
          )}
        </div>
      </div>

      <div className="forecast-chart-canvas">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          preserveAspectRatio="none"
          role="img"
          aria-label={`${model} volatility forecast`}
        >
          <defs>
            <linearGradient
              id="forecastArea"
              x1="0"
              x2="0"
              y1="0"
              y2="1"
            >
              <stop offset="0%" stopOpacity="0.20" />
              <stop offset="100%" stopOpacity="0" />
            </linearGradient>
          </defs>

          {[0, 1, 2, 3].map((row) => {
            const gridY =
              paddingY +
              (row / 3) * (height - paddingY * 2);

            return (
              <line
                key={row}
                x1={paddingX}
                x2={width - paddingX}
                y1={gridY}
                y2={gridY}
                className="forecast-grid-line"
              />
            );
          })}

          <path
            d={`${predictedPath} L ${x(validPoints.length - 1)} ${
              height - paddingY
            } L ${paddingX} ${height - paddingY} Z`}
            fill="url(#forecastArea)"
            className="forecast-area"
          />

          <path
            d={predictedPath}
            fill="none"
            className="forecast-line-predicted"
          />

          {actualPath && (
            <path
              d={actualPath}
              fill="none"
              className="forecast-line-actual"
            />
          )}
        </svg>
      </div>

      <div className="forecast-chart-range">
        <span>{formatDate(validPoints[0].timestamp)}</span>

        <strong>
          {validPoints.length.toLocaleString()} observations
        </strong>

        <span>
          {formatDate(
            validPoints[validPoints.length - 1].timestamp,
          )}
        </span>
      </div>
    </div>
  );
}

export default function Forecasts() {
  const [experimentName, setExperimentName] = useState("");
  const [model, setModel] = useState("EWMA");

  const [forecast, setForecast] = useState<ForecastResponse | null>(
    null,
  );

  const [comparison, setComparison] =
    useState<ForecastComparisonResponse | null>(null);

  const [search, setSearch] = useState("");

  const [loading, setLoading] = useState(false);
  const [comparisonLoading, setComparisonLoading] = useState(false);

  const [error, setError] = useState<string | null>(null);
  const [comparisonError, setComparisonError] =
    useState<string | null>(null);

  const [showQueryPanel, setShowQueryPanel] = useState(false);

  useEffect(() => {
    const savedExperiment = sessionStorage.getItem(
      "avf-selected-experiment",
    );

    if (savedExperiment) {
      setExperimentName(savedExperiment);
    }
  }, []);

  const filteredPoints = useMemo(() => {
    if (!forecast) {
      return [];
    }

    const query = search.trim().toLowerCase();

    if (!query) {
      return forecast.forecasts;
    }

    return forecast.forecasts.filter((point) =>
      [
        point.asset_id,
        point.timestamp,
        String(point.predicted_volatility),
        String(point.actual_volatility ?? ""),
      ]
        .join(" ")
        .toLowerCase()
        .includes(query),
    );
  }, [forecast, search]);

  const latestPoint = forecast?.forecasts.at(-1);


  const handleLoadForecast = async () => {
    if (!experimentName.trim()) {
      setError("Enter an experiment name first.");
      return;
    }

    setLoading(true);
    setError(null);
    setComparison(null);
    setComparisonError(null);

    try {
      const response = await getForecast(
        experimentName.trim(),
        model,
      );

      setForecast(response);

      sessionStorage.setItem(
        "avf-selected-experiment",
        experimentName.trim(),
      );
    } catch (requestError) {
      setForecast(null);
      setError(getErrorMessage(requestError));
    } finally {
      setLoading(false);
    }
  };

  const handleCompare = async () => {
    if (!experimentName.trim()) {
      setComparisonError("Enter an experiment name first.");
      return;
    }

    setComparisonLoading(true);
    setComparisonError(null);

    try {
      const response = await compareForecasts({
        experiment_name: experimentName.trim(),
        dataset_name: "",
        model: model,
        assets: ["*"],
        start_timestamp: "1970-01-01T00:00:00Z",
        end_timestamp: "2100-01-01T00:00:00Z",
      });

      setComparison(response);
    } catch (requestError) {
      setComparison(null);
      setComparisonError(getErrorMessage(requestError));
    } finally {
      setComparisonLoading(false);
    }
  };

  return (
    <div className="page forecasts-page">
      <section className="forecast-page-header">
        <div className="forecast-header-copy">
          <div className="page-eyebrow">
            VOLATILITY RESEARCH
          </div>

          <h1>Forecasts</h1>

          <p>
            Inspect model-level volatility forecasts and compare
            out-of-sample forecasting behaviour without introducing
            results that have not been produced by the research
            pipeline.
          </p>
        </div>

        <div className="forecast-header-actions">
          <button
            className="button button-secondary"
            type="button"
            onClick={() => setShowQueryPanel(true)}
          >
            <SlidersHorizontal size={15} />
            Configure view
          </button>

          <button
            className="button button-primary"
            type="button"
            onClick={handleLoadForecast}
            disabled={loading}
          >
            {loading ? (
              <RefreshCw className="spin" size={15} />
            ) : (
              <Play size={15} />
            )}
            {loading ? "Loading..." : "Load forecast"}
          </button>
        </div>
      </section>

      <section className="forecast-query-bar">
        <div className="forecast-query-field">
          <Search size={17} />

          <input
            value={experimentName}
            onChange={(event) =>
              setExperimentName(event.target.value)
            }
            placeholder="Experiment name"
            aria-label="Experiment name"
          />
        </div>

        <div className="forecast-model-select">
          <label htmlFor="forecast-model">MODEL</label>

          <div className="forecast-select-wrap">
            <select
              id="forecast-model"
              value={model}
              onChange={(event) => setModel(event.target.value)}
            >
              {MODELS.map((item) => (
                <option key={item.key} value={item.key}>
                  {item.label}
                </option>
              ))}
            </select>

            <ChevronDown size={15} />
          </div>
        </div>
      </section>

      {error && (
        <div className="forecast-alert forecast-alert-error">
          <AlertCircle size={18} />

          <div>
            <strong>Forecast unavailable</strong>
            <p>{error}</p>
          </div>

          <button
            type="button"
            onClick={() => setError(null)}
            aria-label="Dismiss error"
          >
            <X size={16} />
          </button>
        </div>
      )}

      <section className="forecast-summary-grid">
        <MetricCard
          label="Model"
          value={forecast?.model ?? "—"}
          icon={Sparkles}
          tone="cyan"
        />

        <MetricCard
          label="Observations"
          value={
            forecast
              ? forecast.observation_count.toLocaleString()
              : "—"
          }
          icon={Activity}
          tone="violet"
        />

        <MetricCard
          label="Assets"
          value={
            forecast
              ? forecast.asset_count.toLocaleString()
              : "—"
          }
          icon={Database}
          tone="blue"
        />

        <MetricCard
          label="Latest forecast"
          value={
            latestPoint
              ? formatNumber(latestPoint.predicted_volatility)
              : "—"
          }
          icon={TrendingUp}
          tone="green"
        />
      </section>

      <section className="forecast-main-grid">
        <div className="surface forecast-chart-surface">
          {forecast ? (
            <ForecastChart
              points={forecast.forecasts}
              model={forecast.model}
            />
          ) : (
            <EmptyForecastState
              title="Select an experiment and model"
              description="Load a registered forecast to inspect its out-of-sample volatility trajectory."
            />
          )}
        </div>

        <aside className="surface forecast-context-card">
          <div className="forecast-context-heading">
            <div className="forecast-context-icon">
              <Target size={18} />
            </div>

            <div>
              <span>FORECAST CONTEXT</span>
              <h3>Research specification</h3>
            </div>
          </div>

          <div className="forecast-context-list">
            <div>
              <span>Experiment</span>
              <strong>{forecast?.experiment_name ?? "—"}</strong>
            </div>

            <div>
              <span>Dataset</span>
              <strong>{forecast?.dataset_name ?? "—"}</strong>
            </div>

            <div>
              <span>Model</span>
              <strong>{forecast?.model ?? model}</strong>
            </div>

            <div>
              <span>Assets</span>
              <strong>{forecast?.asset_count ?? "—"}</strong>
            </div>

            <div>
              <span>Status</span>
              {forecast ? (
                <StatusBadge status={forecast.status} />
              ) : (
                <strong>—</strong>
              )}
            </div>
          </div>

          <div className="forecast-context-note">
            <Clock3 size={15} />

            <span>
              Forecast observations are displayed exactly as
              returned by the backend.
            </span>
          </div>
        </aside>
      </section>

      <section className="forecast-observations-section">
        <div className="forecast-section-heading">
          <div>
            <div className="section-kicker">
              OUT-OF-SAMPLE OUTPUT
            </div>

            <h2>Forecast observations</h2>
          </div>

          <div className="forecast-observation-tools">
            <div className="forecast-search">
              <Search size={15} />

              <input
                value={search}
                onChange={(event) => setSearch(event.target.value)}
                placeholder="Search asset or observation..."
              />
            </div>

            <span>
              {filteredPoints.length.toLocaleString()} rows
            </span>
          </div>
        </div>

        <div className="surface forecast-table-surface">
          {!forecast ? (
            <EmptyForecastState
              title="No observations loaded"
              description="Load a forecast above to populate this table."
            />
          ) : filteredPoints.length === 0 ? (
            <EmptyForecastState
              title="No matching observations"
              description="Try a different asset, timestamp or search term."
            />
          ) : (
            <div className="forecast-table-wrapper">
              <table className="forecast-table">
                <thead>
                  <tr>
                    <th>Timestamp</th>
                    <th>Asset</th>
                    <th>Actual volatility</th>
                    <th>Predicted volatility</th>
                    <th>Difference</th>
                  </tr>
                </thead>

                <tbody>
                  {filteredPoints.slice(0, 100).map((point) => {
                    const difference =
                      point.actual_volatility === null
                        ? null
                        : point.predicted_volatility -
                          point.actual_volatility;

                    return (
                      <tr
                        key={`${point.timestamp}-${point.asset_id}`}
                      >
                        <td>
                          {formatDateTime(point.timestamp)}
                        </td>

                        <td>
                          <span className="forecast-asset">
                            {point.asset_id}
                          </span>
                        </td>

                        <td>
                          {point.actual_volatility === null
                            ? "—"
                            : formatNumber(
                                point.actual_volatility,
                              )}
                        </td>

                        <td className="forecast-predicted-value">
                          {formatNumber(
                            point.predicted_volatility,
                          )}
                        </td>

                        <td>
                          {difference === null
                            ? "—"
                            : formatNumber(difference)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>

              {filteredPoints.length > 100 && (
                <div className="forecast-table-limit">
                  Showing the first 100 rows of{" "}
                  {filteredPoints.length.toLocaleString()} matching
                  observations.
                </div>
              )}
            </div>
          )}
        </div>
      </section>

      <section className="forecast-research-section">
        <div className="forecast-section-heading">
          <div>
            <div className="section-kicker">
              MODEL EVALUATION
            </div>

            <h2>Forecast comparison</h2>
          </div>

          <button
            className="button button-secondary"
            type="button"
            onClick={handleCompare}
            disabled={comparisonLoading}
          >
            {comparisonLoading ? (
              <RefreshCw className="spin" size={15} />
            ) : (
              <BarChart3 size={15} />
            )}
            Compare models
          </button>
        </div>

        {comparisonError && (
          <div className="forecast-alert forecast-alert-warning">
            <AlertCircle size={18} />

            <div>
              <strong>Comparison unavailable</strong>
              <p>{comparisonError}</p>
            </div>
          </div>
        )}

        {comparison ? (
          <div className="surface forecast-comparison-surface">
            <div className="forecast-comparison-meta">
              <span>
                {comparison.observation_count.toLocaleString()}{" "}
                observations
              </span>

              <span>
                {comparison.asset_count.toLocaleString()} assets
              </span>
            </div>

            <div className="forecast-comparison-grid">
              {comparison.models.map((item) => (
                <article
                  className="forecast-comparison-card"
                  key={item.model}
                >
                  <div className="forecast-comparison-card-top">
                    <span>{item.model}</span>
                    <CheckCircle2 size={16} />
                  </div>

                  <div className="forecast-comparison-values">
                    <div>
                      <span>MAE</span>
                      <strong>{formatNumber(item.mae)}</strong>
                    </div>

                    <div>
                      <span>RMSE</span>
                      <strong>{formatNumber(item.rmse)}</strong>
                    </div>

                    <div>
                      <span>QLIKE</span>
                      <strong>{formatNumber(item.qlike)}</strong>
                    </div>
                  </div>
                </article>
              ))}
            </div>
          </div>
        ) : (
          <div className="surface forecast-comparison-empty">
            <div className="forecast-empty-icon">
              <Zap size={21} />
            </div>

            <div>
              <h3>Comparison is research-backed</h3>
              <p>
                MAE, RMSE and QLIKE will appear here only when the
                backend evaluation pipeline returns real comparison
                results.
              </p>
            </div>
          </div>
        )}
      </section>

      <section className="forecast-integrity-panel">
        <div className="forecast-integrity-icon">
          <CheckCircle2 size={18} />
        </div>

        <div>
          <div className="forecast-integrity-label">
            RESEARCH INTEGRITY
          </div>

          <h3>No fabricated forecast results</h3>

          <p>
            This interface displays registered forecast observations
            and backend evaluation results only. It does not
            generate synthetic metrics or infer model performance
            when the research pipeline has not produced them.
          </p>
        </div>
      </section>

      {showQueryPanel && (
        <div
          className="forecast-modal-backdrop"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) {
              setShowQueryPanel(false);
            }
          }}
        >
          <div className="forecast-query-modal">
            <div className="forecast-modal-header">
              <div>
                <div className="page-eyebrow">
                  FORECAST CONFIGURATION
                </div>

                <h2>Forecast view</h2>

                <p>
                  Select the registered experiment and model to
                  inspect.
                </p>
              </div>

              <button
                type="button"
                onClick={() => setShowQueryPanel(false)}
                aria-label="Close"
              >
                <X size={18} />
              </button>
            </div>

            <div className="forecast-modal-body">
              <label>
                Experiment name

                <input
                  value={experimentName}
                  onChange={(event) =>
                    setExperimentName(event.target.value)
                  }
                  placeholder="e.g. AVF_TEST_001"
                />
              </label>

              <label>
                Model

                <select
                  value={model}
                  onChange={(event) =>
                    setModel(event.target.value)
                  }
                >
                  {MODELS.map((item) => (
                    <option key={item.key} value={item.key}>
                      {item.label}
                    </option>
                  ))}
                </select>
              </label>

              <div className="forecast-model-description">
                <Sparkles size={16} />

                <div>
                  <strong>
                    {
                      MODELS.find(
                        (item) => item.key === model,
                      )?.label
                    }
                  </strong>

                  <span>
                    {
                      MODELS.find(
                        (item) => item.key === model,
                      )?.description
                    }
                  </span>
                </div>
              </div>
            </div>

            <div className="forecast-modal-footer">
              <button
                className="button button-secondary"
                type="button"
                onClick={() => setShowQueryPanel(false)}
              >
                Cancel
              </button>

              <button
                className="button button-primary"
                type="button"
                onClick={() => {
                  setShowQueryPanel(false);
                  void handleLoadForecast();
                }}
              >
                <Play size={15} />
                Load forecast
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}