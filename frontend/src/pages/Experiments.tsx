import {
  Beaker,
  Check,
  ChevronRight,
  CircleAlert,
  FlaskConical,
  Play,
  Plus,
  RefreshCw,
  Search,
  Settings2,
  X,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  createExperiment,
  getExperiments,
  runExperiments,
} from "../api/experiments";
import type {
  ExperimentRequest,
  ExperimentResponse,
  ExperimentRunResponse,
} from "../api/experiments";

const MODELS = [
  "EWMA",
  "GJR-GARCH",
  "XGBoost",
  "HMM",
  "Regime-XGBoost",
];

const METRICS = ["MAE", "RMSE", "QLIKE"];

const RISK_MEASURES = ["VaR95", "VaR99", "ES95"];

const emptyForm: ExperimentRequest = {
  name: "",
  description: "",
  models: [],
  metrics: [],
  risk_measures: [],
  parameters: {},
  metadata: {},
};

function formatStatus(status: string) {
  return status
    .replace(/[-_]/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function StatusBadge({ status }: { status: string }) {
  const normalized = status.toLowerCase();

  return (
    <span
      className={`experiment-status ${
        normalized === "configured"
          ? "experiment-status-configured"
          : normalized === "running"
            ? "experiment-status-running"
            : normalized === "completed" || normalized === "success"
              ? "experiment-status-success"
              : "experiment-status-error"
      }`}
    >
      <span className="experiment-status-dot" />
      {formatStatus(status)}
    </span>
  );
}

function SelectionGroup({
  label,
  values,
  selected,
  onToggle,
}: {
  label: string;
  values: string[];
  selected: string[];
  onToggle: (value: string) => void;
}) {
  return (
    <div className="experiment-selection-group">
      <div className="experiment-field-label">{label}</div>

      <div className="experiment-chip-grid">
        {values.map((value) => {
          const active = selected.includes(value);

          return (
            <button
              key={value}
              type="button"
              className={`experiment-chip ${
                active ? "experiment-chip-active" : ""
              }`}
              onClick={() => onToggle(value)}
            >
              {active && <Check size={14} />}
              {value}
            </button>
          );
        })}
      </div>
    </div>
  );
}

function ExperimentCard({
  experiment,
  selected,
  onSelect,
  onRun,
}: {
  experiment: ExperimentResponse;
  selected: boolean;
  onSelect: () => void;
  onRun: () => void;
}) {
  return (
    <article
      className={`experiment-card ${
        selected ? "experiment-card-selected" : ""
      }`}
    >
      <div className="experiment-card-top">
        <div className="experiment-card-icon">
          <FlaskConical size={19} />
        </div>

        <div className="experiment-card-heading">
          <div className="experiment-card-title-row">
            <h3>{experiment.name}</h3>
            <StatusBadge status={experiment.status} />
          </div>

          <p>{experiment.description}</p>
        </div>

        <button
          type="button"
          className="experiment-select-button"
          onClick={onSelect}
          aria-label={`Select ${experiment.name}`}
        >
          <span className={selected ? "experiment-check-active" : ""}>
            {selected ? <Check size={15} /> : null}
          </span>
        </button>
      </div>

      <div className="experiment-card-section">
        <span className="experiment-card-label">Models</span>

        <div className="experiment-tag-row">
          {experiment.models.map((model) => (
            <span className="experiment-tag" key={model}>
              {model}
            </span>
          ))}
        </div>
      </div>

      <div className="experiment-card-section">
        <span className="experiment-card-label">Forecast metrics</span>

        <div className="experiment-tag-row">
          {experiment.metrics.map((metric) => (
            <span className="experiment-tag experiment-tag-metric" key={metric}>
              {metric}
            </span>
          ))}
        </div>
      </div>

      <div className="experiment-card-footer">
        <div>
          <span className="experiment-card-label">Risk</span>
          <strong>
            {experiment.risk_measures.length
              ? experiment.risk_measures.join(" · ")
              : "None configured"}
          </strong>
        </div>

        <button
          type="button"
          className="experiment-run-small"
          onClick={onRun}
        >
          <Play size={14} />
          Run
        </button>
      </div>
    </article>
  );
}

export default function Experiments() {
  const [experiments, setExperiments] = useState<ExperimentResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState("");
  const [search, setSearch] = useState("");

  const [showCreate, setShowCreate] = useState(false);
  const [form, setForm] = useState<ExperimentRequest>(emptyForm);
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState("");

  const [selectedNames, setSelectedNames] = useState<string[]>([]);
  const [running, setRunning] = useState(false);
  const [runResult, setRunResult] =
    useState<ExperimentRunResponse | null>(null);

  async function loadExperiments(showRefresh = false) {
    try {
      if (showRefresh) {
        setRefreshing(true);
      } else {
        setLoading(true);
      }

      setError("");

      const response = await getExperiments();
      setExperiments(response.experiments);

      setSelectedNames((current) =>
        current.filter((name) =>
          response.experiments.some(
            (experiment) => experiment.name === name,
          ),
        ),
      );
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to load experiments from the backend.",
      );
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }

  useEffect(() => {
    void loadExperiments();
  }, []);

  const filteredExperiments = useMemo(() => {
    const query = search.trim().toLowerCase();

    if (!query) {
      return experiments;
    }

    return experiments.filter((experiment) => {
      return (
        experiment.name.toLowerCase().includes(query) ||
        experiment.description.toLowerCase().includes(query) ||
        experiment.models.some((model) =>
          model.toLowerCase().includes(query),
        ) ||
        experiment.metrics.some((metric) =>
          metric.toLowerCase().includes(query),
        )
      );
    });
  }, [experiments, search]);

  const configuredCount = experiments.length;

  const runningCount = experiments.filter(
    (experiment) =>
      experiment.status.toLowerCase() === "running",
  ).length;

  const completedCount = experiments.filter((experiment) => {
    const status = experiment.status.toLowerCase();

    return status === "completed" || status === "success";
  }).length;

  function toggleSelection(name: string) {
    setSelectedNames((current) =>
      current.includes(name)
        ? current.filter((item) => item !== name)
        : [...current, name],
    );
  }

  function toggleFormValue(
    field: "models" | "metrics" | "risk_measures",
    value: string,
  ) {
    setForm((current) => {
      const values = current[field];

      return {
        ...current,
        [field]: values.includes(value)
          ? values.filter((item) => item !== value)
          : [...values, value],
      };
    });
  }

  function updateForm(
    field: keyof ExperimentRequest,
    value:
      | string
      | string[]
      | Record<string, unknown>,
  ) {
    setForm((current) => ({
      ...current,
      [field]: value as never,
    }));
  }

  async function handleCreate(
    event: React.FormEvent<HTMLFormElement>,
  ) {
    event.preventDefault();

    if (
      !form.name.trim() ||
      !form.description.trim() ||
      form.models.length === 0 ||
      form.metrics.length === 0 ||
      form.risk_measures.length === 0
    ) {
      setCreateError(
        "Name, description, at least one model, metric and risk measure are required.",
      );
      return;
    }

    try {
      setCreating(true);
      setCreateError("");

      const payload: ExperimentRequest = {
        ...form,
        name: form.name.trim(),
        description: form.description.trim(),
      };

      const response = await createExperiment(payload);

      setExperiments((current) => [...current, response]);
      setForm(emptyForm);
      setShowCreate(false);
    } catch (err) {
      setCreateError(
        err instanceof Error
          ? err.message
          : "Unable to create the experiment.",
      );
    } finally {
      setCreating(false);
    }
  }

  async function handleRun(names: string[]) {
    if (names.length === 0 || running) {
      return;
    }

    try {
      setRunning(true);
      setError("");
      setRunResult(null);

      const response = await runExperiments({
        experiment_names: names,
      });

      setRunResult(response);

      await loadExperiments(true);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to run the selected experiments.",
      );
    } finally {
      setRunning(false);
    }
  }

  return (
    <section className="page experiments-page">
      <header className="module-header">
        <div>
          <div className="page-eyebrow">Research orchestration</div>

          <h1>Experiments</h1>

          <p className="page-description">
            Configure, inspect and execute the controlled experiments
            that evaluate the AVF-TRPDE forecasting and risk pipeline.
          </p>
        </div>

        <div className="module-header-actions">
          <button
            className="secondary-button"
            type="button"
            onClick={() => void loadExperiments(true)}
            disabled={loading || refreshing}
          >
            <RefreshCw
              size={15}
              className={refreshing ? "spin" : ""}
            />
            Refresh
          </button>

          <button
            className="primary-button"
            type="button"
            onClick={() => {
              setCreateError("");
              setShowCreate(true);
            }}
          >
            <Plus size={16} />
            New Experiment
          </button>
        </div>
      </header>

      <section className="experiment-stat-grid">
        <div className="experiment-stat-card">
          <div className="experiment-stat-icon">
            <Beaker size={18} />
          </div>
          <div>
            <span>Configured</span>
            <strong>{configuredCount}</strong>
          </div>
        </div>

        <div className="experiment-stat-card">
          <div className="experiment-stat-icon">
            <Play size={18} />
          </div>
          <div>
            <span>Running</span>
            <strong>{runningCount}</strong>
          </div>
        </div>

        <div className="experiment-stat-card">
          <div className="experiment-stat-icon">
            <Check size={18} />
          </div>
          <div>
            <span>Completed</span>
            <strong>{completedCount}</strong>
          </div>
        </div>

        <div className="experiment-stat-card experiment-stat-accent">
          <div className="experiment-stat-icon">
            <Settings2 size={18} />
          </div>
          <div>
            <span>Selected</span>
            <strong>{selectedNames.length}</strong>
          </div>
        </div>
      </section>

      {error && (
        <div className="experiment-alert experiment-alert-error">
          <CircleAlert size={17} />
          <span>{error}</span>
          <button
            type="button"
            onClick={() => setError("")}
            aria-label="Dismiss error"
          >
            <X size={15} />
          </button>
        </div>
      )}

      {runResult && (
        <div
          className={`experiment-alert ${
            runResult.all_successful
              ? "experiment-alert-success"
              : "experiment-alert-warning"
          }`}
        >
          <Check size={17} />

          <div>
            <strong>
              Experiment execution finished
            </strong>

            <span>
              {runResult.successful_count} successful ·{" "}
              {runResult.failed_count} failed ·{" "}
              {runResult.experiment_count} requested
            </span>
          </div>

          <button
            type="button"
            onClick={() => setRunResult(null)}
            aria-label="Dismiss execution result"
          >
            <X size={15} />
          </button>
        </div>
      )}

      <section className="experiment-toolbar">
        <div className="datasets-search">
          <Search size={17} />
          <input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search experiments, models or metrics..."
          />
          {search && (
            <button
              type="button"
              onClick={() => setSearch("")}
              aria-label="Clear search"
            >
              <X size={15} />
            </button>
          )}
        </div>

        <div className="experiment-toolbar-actions">
          {selectedNames.length > 0 && (
            <button
              type="button"
              className="secondary-button"
              onClick={() => void handleRun(selectedNames)}
              disabled={running}
            >
              <Play size={15} />
              {running
                ? "Running..."
                : `Run selected (${selectedNames.length})`}
            </button>
          )}

          <span className="experiment-result-count">
            {filteredExperiments.length} experiment
            {filteredExperiments.length === 1 ? "" : "s"}
          </span>
        </div>
      </section>

      {loading ? (
        <section className="experiment-loading">
          <div className="experiment-loading-orb">
            <FlaskConical size={24} />
          </div>

          <h3>Loading experiments</h3>

          <p>
            Connecting to the AVF-TRPDE research configuration
            service.
          </p>
        </section>
      ) : filteredExperiments.length === 0 ? (
        <section className="experiment-empty">
          <div className="experiment-empty-icon">
            <FlaskConical size={28} />
          </div>

          <h2>
            {experiments.length === 0
              ? "No experiments configured"
              : "No experiments match your search"}
          </h2>

          <p>
            {experiments.length === 0
              ? "Create the first research configuration to begin an experiment."
              : "Try another experiment name, model or metric."}
          </p>

          {experiments.length === 0 && (
            <button
              type="button"
              className="primary-button"
              onClick={() => setShowCreate(true)}
            >
              <Plus size={16} />
              Create experiment
            </button>
          )}
        </section>
      ) : (
        <section className="experiment-grid">
          {filteredExperiments.map((experiment) => (
            <ExperimentCard
              key={experiment.name}
              experiment={experiment}
              selected={selectedNames.includes(experiment.name)}
              onSelect={() => toggleSelection(experiment.name)}
              onRun={() =>
                void handleRun([experiment.name])
              }
            />
          ))}
        </section>
      )}

      {showCreate && (
        <div
          className="modal-backdrop"
          role="presentation"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) {
              setShowCreate(false);
            }
          }}
        >
          <div
            className="modal-card experiment-create-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="create-experiment-title"
          >
            <div className="modal-header">
              <div>
                <div className="page-eyebrow">
                  Research configuration
                </div>
                <h2 id="create-experiment-title">
                  New Experiment
                </h2>
              </div>

              <button
                type="button"
                className="modal-close"
                onClick={() => setShowCreate(false)}
                aria-label="Close"
              >
                <X size={18} />
              </button>
            </div>

            <form onSubmit={handleCreate}>
              <div className="experiment-form-body">
                <div className="experiment-form-row">
                  <label className="experiment-form-field">
                    <span>Name</span>
                    <input
                      value={form.name}
                      maxLength={100}
                      onChange={(event) =>
                        updateForm("name", event.target.value)
                      }
                      placeholder="e.g. baseline-volatility-study"
                    />
                  </label>

                  <label className="experiment-form-field">
                    <span>Description</span>
                    <input
                      value={form.description}
                      maxLength={500}
                      onChange={(event) =>
                        updateForm(
                          "description",
                          event.target.value,
                        )
                      }
                      placeholder="Describe the research experiment"
                    />
                  </label>
                </div>

                <SelectionGroup
                  label="Models"
                  values={MODELS}
                  selected={form.models}
                  onToggle={(value) =>
                    toggleFormValue("models", value)
                  }
                />

                <SelectionGroup
                  label="Forecast metrics"
                  values={METRICS}
                  selected={form.metrics}
                  onToggle={(value) =>
                    toggleFormValue("metrics", value)
                  }
                />

                <SelectionGroup
                  label="Risk measures"
                  values={RISK_MEASURES}
                  selected={form.risk_measures}
                  onToggle={(value) =>
                    toggleFormValue(
                      "risk_measures",
                      value,
                    )
                  }
                />

                <div className="experiment-json-grid">
                  <label className="experiment-form-field">
                    <span>Parameters JSON</span>
                    <textarea
                      rows={5}
                      value={JSON.stringify(
                        form.parameters,
                        null,
                        2,
                      )}
                      onChange={(event) => {
                        try {
                          updateForm(
                            "parameters",
                            JSON.parse(event.target.value),
                          );
                        } catch {
                          // Keep the current valid object until JSON is valid.
                        }
                      }}
                    />
                  </label>

                  <label className="experiment-form-field">
                    <span>Metadata JSON</span>
                    <textarea
                      rows={5}
                      value={JSON.stringify(
                        form.metadata,
                        null,
                        2,
                      )}
                      onChange={(event) => {
                        try {
                          updateForm(
                            "metadata",
                            JSON.parse(event.target.value),
                          );
                        } catch {
                          // Keep the current valid object until JSON is valid.
                        }
                      }}
                    />
                  </label>
                </div>

                {createError && (
                  <div className="experiment-form-error">
                    <CircleAlert size={16} />
                    {createError}
                  </div>
                )}
              </div>

              <div className="modal-footer">
                <button
                  type="button"
                  className="secondary-button"
                  onClick={() => setShowCreate(false)}
                  disabled={creating}
                >
                  Cancel
                </button>

                <button
                  type="submit"
                  className="primary-button"
                  disabled={creating}
                >
                  {creating ? (
                    <>
                      <RefreshCw
                        size={15}
                        className="spin"
                      />
                      Creating...
                    </>
                  ) : (
                    <>
                      <Plus size={16} />
                      Create experiment
                    </>
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </section>
  );
}