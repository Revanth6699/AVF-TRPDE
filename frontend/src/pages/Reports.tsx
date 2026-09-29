import { useMemo, useState } from "react";
import type { FormEvent } from "react";
import { isAxiosError } from "axios";
import {
  AlertCircle,
  BarChart3,
  CheckCircle2,
  ClipboardList,
  FileBarChart,
  FileCheck2,
  FileText,
  FlaskConical,
  Gauge,
  History,
  Layers3,
  Plus,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  Target,
  X,
} from "lucide-react";

import {
  createReport,
  getReport,
  runReport,
  type ReportRequest,
  type ReportResponse,
  type ReportRunResponse,
} from "../api/reports";

type ReportRecord = {
  response: ReportResponse;
  request: ReportRequest;
};

const REPORT_TYPE_SUGGESTIONS = [
  "Full Research Report",
  "Volatility & Forecasting",
  "Tail-Risk Analysis",
  "Portfolio Decision Report",
  "Stress & Robustness Report",
];

const EMPTY_FORM: ReportRequest = {
  experiment_name: "",
  dataset_name: "",
  report_type: "Full Research Report",
  start_timestamp: "",
  end_timestamp: "",
  include_forecasts: true,
  include_risk: true,
  include_portfolio: true,
  include_stress: true,
  include_statistical_tests: true,
};

function getApiErrorMessage(
  error: unknown,
  fallback: string,
): string {
  if (isAxiosError(error)) {
    const detail = error.response?.data?.detail;

    if (typeof detail === "string" && detail.trim()) {
      return detail;
    }

    if (error.message) {
      return error.message;
    }
  }

  if (error instanceof Error && error.message) {
    return error.message;
  }

  return fallback;
}

function formatDateTime(value: string): string {
  if (!value) {
    return "—";
  }

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return date.toLocaleString([], {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

function formatMetric(value: number): string {
  if (!Number.isFinite(value)) {
    return "—";
  }

  return value.toFixed(4);
}

function getScopeCount(request: ReportRequest): number {
  return [
    request.include_forecasts,
    request.include_risk,
    request.include_portfolio,
    request.include_stress,
    request.include_statistical_tests,
  ].filter(Boolean).length;
}

export default function Reports() {
  const [records, setRecords] = useState<ReportRecord[]>([]);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);

  const [form, setForm] = useState<ReportRequest>(EMPTY_FORM);
  const [search, setSearch] = useState("");

  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [isRunning, setIsRunning] = useState(false);

  const [error, setError] = useState("");
  const [runResult, setRunResult] = useState<ReportRunResponse | null>(null);

  const filteredRecords = useMemo(() => {
    const query = search.trim().toLowerCase();

    if (!query) {
      return records;
    }

    return records.filter((record) => {
      const { response, request } = record;

      return [
        response.experiment_name,
        response.dataset_name,
        response.report_type,
        response.status,
        request.start_timestamp,
        request.end_timestamp,
      ]
        .join(" ")
        .toLowerCase()
        .includes(query);
    });
  }, [records, search]);

  const selectedRecord =
    selectedIndex !== null ? records[selectedIndex] : null;

  const acceptedCount = records.filter(
    (record) => record.response.status === "accepted",
  ).length;

  const outputCount = records.filter(
    (record) => record.response.sections.length > 0,
  ).length;

  const totalSections = records.reduce(
    (total, record) => total + record.response.sections.length,
    0,
  );

  function openCreateModal() {
    setError("");
    setForm(EMPTY_FORM);
    setIsModalOpen(true);
  }

  function closeCreateModal() {
    if (!isSubmitting) {
      setIsModalOpen(false);
    }
  }

  function updateForm<K extends keyof ReportRequest>(
    key: K,
    value: ReportRequest[K],
  ) {
    setForm((current) => ({
      ...current,
      [key]: value,
    }));
  }

  function toggleScope(key: keyof ReportRequest) {
    const value = form[key];

    if (typeof value !== "boolean") {
      return;
    }

    updateForm(key, !value);
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    setError("");
    setRunResult(null);

    if (!form.experiment_name.trim()) {
      setError("Experiment name is required.");
      return;
    }

    if (!form.dataset_name.trim()) {
      setError("Dataset name is required.");
      return;
    }

    if (!form.report_type.trim()) {
      setError("Report type is required.");
      return;
    }

    if (!form.start_timestamp || !form.end_timestamp) {
      setError("Start and end timestamps are required.");
      return;
    }

    if (
      new Date(form.start_timestamp).getTime() >
      new Date(form.end_timestamp).getTime()
    ) {
      setError("Start timestamp must be before or equal to the end timestamp.");
      return;
    }

    if (getScopeCount(form) === 0) {
      setError("Select at least one report scope.");
      return;
    }

    setIsSubmitting(true);

    try {
      const response = await createReport({
        ...form,
        experiment_name: form.experiment_name.trim(),
        dataset_name: form.dataset_name.trim(),
        report_type: form.report_type.trim(),
      });

      const record: ReportRecord = {
        response,
        request: {
          ...form,
          experiment_name: form.experiment_name.trim(),
          dataset_name: form.dataset_name.trim(),
          report_type: form.report_type.trim(),
        },
      };

      setRecords((current) => {
        const next = [...current];
        next.push(record);
        return next;
      });

      setSelectedIndex(records.length);
      setIsModalOpen(false);
      setForm(EMPTY_FORM);
    } catch (requestError) {
      setError(
        getApiErrorMessage(
          requestError,
          "Unable to register the report.",
        ),
      );
    } finally {
      setIsSubmitting(false);
    }
  }

  async function refreshSelectedReport() {
    if (!selectedRecord || selectedIndex === null) {
      return;
    }

    setError("");
    setIsRefreshing(true);

    try {
      const response = await getReport(
        selectedRecord.response.experiment_name,
        selectedRecord.response.report_type,
      );

      setRecords((current) => {
        const next = [...current];

        next[selectedIndex] = {
          ...next[selectedIndex],
          response,
        };

        return next;
      });

      setRunResult(null);
    } catch (requestError) {
      setError(
        getApiErrorMessage(
          requestError,
          "Unable to retrieve the registered report.",
        ),
      );
    } finally {
      setIsRefreshing(false);
    }
  }

  async function executeSelectedReport() {
    if (!selectedRecord) {
      return;
    }

    setError("");
    setRunResult(null);
    setIsRunning(true);

    try {
      const result = await runReport(selectedRecord.request);

      setRunResult(result);
    } catch (requestError) {
      setError(
        getApiErrorMessage(
          requestError,
          "Unable to run the report.",
        ),
      );
    } finally {
      setIsRunning(false);
    }
  }

  function selectRecord(record: ReportRecord) {
    const index = records.indexOf(record);

    if (index !== -1) {
      setSelectedIndex(index);
      setError("");
      setRunResult(null);
    }
  }

  return (
    <main className="reports-page page-shell">
      <section className="reports-hero">
        <div className="reports-hero-copy">
          <div className="page-eyebrow">
            <FileBarChart size={15} />
            Research Reporting
          </div>

          <h1>Reports &amp; Research Output</h1>

          <p>
            Register research reports, define the analysis scope, retrieve
            generated sections, and execute the reporting workflow from one
            controlled workspace.
          </p>

          <div className="reports-hero-actions">
            <button
              type="button"
              className="primary-button"
              onClick={openCreateModal}
            >
              <Plus size={17} />
              Create report
            </button>

            <div className="reports-hero-note">
              <ShieldCheck size={16} />
              Research outputs are sourced from the reporting engine.
            </div>
          </div>
        </div>

        <div className="reports-hero-visual" aria-hidden="true">
          <div className="report-orbit report-orbit-one" />
          <div className="report-orbit report-orbit-two" />
          <div className="report-orbit report-orbit-three" />

          <div className="report-visual-core">
            <FileText size={31} />
            <span>REPORT</span>
          </div>
        </div>
      </section>

      {error && (
        <div className="reports-alert" role="alert">
          <AlertCircle size={18} />
          <div>
            <strong>Action could not be completed</strong>
            <span>{error}</span>
          </div>

          <button
            type="button"
            className="icon-button"
            onClick={() => setError("")}
            aria-label="Dismiss error"
          >
            <X size={16} />
          </button>
        </div>
      )}

      <section className="report-stat-grid">
        <article className="report-stat-card">
          <div className="report-stat-icon">
            <ClipboardList size={20} />
          </div>
          <div>
            <span>REGISTERED</span>
            <strong>{records.length}</strong>
            <small>Reports in this workspace</small>
          </div>
        </article>

        <article className="report-stat-card">
          <div className="report-stat-icon">
            <CheckCircle2 size={20} />
          </div>
          <div>
            <span>ACCEPTED</span>
            <strong>{acceptedCount}</strong>
            <small>Accepted report registrations</small>
          </div>
        </article>

        <article className="report-stat-card">
          <div className="report-stat-icon">
            <Layers3 size={20} />
          </div>
          <div>
            <span>OUTPUT SECTIONS</span>
            <strong>{totalSections}</strong>
            <small>Returned sections across reports</small>
          </div>
        </article>

        <article className="report-stat-card">
          <div className="report-stat-icon">
            <Sparkles size={20} />
          </div>
          <div>
            <span>WITH OUTPUT</span>
            <strong>{outputCount}</strong>
            <small>Reports containing sections</small>
          </div>
        </article>
      </section>

      <section className="reports-toolbar">
        <div className="reports-toolbar-heading">
          <div>
            <span className="section-kicker">REPORT REGISTER</span>
            <h2>Research reports</h2>
          </div>

          <p>
            Reports created during this frontend session are shown here.
          </p>
        </div>

        <div className="reports-toolbar-actions">
          <label className="search-field">
            <FileText size={16} />
            <input
              type="search"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search reports..."
            />
          </label>

          <button
            type="button"
            className="secondary-button"
            onClick={openCreateModal}
          >
            <Plus size={16} />
            New report
          </button>
        </div>
      </section>

      <section className="reports-workspace">
        <div className="reports-list-panel">
          {filteredRecords.length === 0 ? (
            <div className="reports-empty-state">
              <div className="reports-empty-icon">
                <FileBarChart size={28} />
              </div>

              <h3>
                {records.length === 0
                  ? "No reports registered yet"
                  : "No matching reports"}
              </h3>

              <p>
                {records.length === 0
                  ? "Create a report registration to define the research output you want the reporting engine to produce."
                  : "Try another experiment, dataset, or report-type search."}
              </p>

              {records.length === 0 && (
                <button
                  type="button"
                  className="primary-button"
                  onClick={openCreateModal}
                >
                  <Plus size={16} />
                  Register first report
                </button>
              )}
            </div>
          ) : (
            <div className="reports-list">
              {filteredRecords.map((record) => {
                const index = records.indexOf(record);
                const isSelected = index === selectedIndex;

                return (
                  <button
                    type="button"
                    key={`${record.response.experiment_name}-${record.response.report_type}-${index}`}
                    className={`report-list-item ${
                      isSelected ? "is-selected" : ""
                    }`}
                    onClick={() => selectRecord(record)}
                  >
                    <div className="report-list-item-icon">
                      <FileText size={19} />
                    </div>

                    <div className="report-list-item-main">
                      <div className="report-list-item-topline">
                        <strong>{record.response.report_type}</strong>

                        <span
                          className={`status-pill status-${record.response.status}`}
                        >
                          {record.response.status}
                        </span>
                      </div>

                      <span className="report-list-item-experiment">
                        {record.response.experiment_name}
                      </span>

                      <span className="report-list-item-meta">
                        {record.response.dataset_name}
                        <span>•</span>
                        {getScopeCount(record.request)} scope areas
                      </span>
                    </div>

                    <div className="report-list-item-arrow">
                      →
                    </div>
                  </button>
                );
              })}
            </div>
          )}
        </div>

        <aside className="reports-context-panel">
          {!selectedRecord ? (
            <div className="reports-context-empty">
              <div className="reports-context-empty-icon">
                <Target size={23} />
              </div>

              <span className="section-kicker">REPORT CONTEXT</span>

              <h3>Select a report</h3>

              <p>
                Choose a registered report from the register to inspect its
                configuration, returned sections, and execution state.
              </p>
            </div>
          ) : (
            <>
              <div className="reports-context-header">
                <div>
                  <span className="section-kicker">SELECTED REPORT</span>
                  <h2>{selectedRecord.response.report_type}</h2>
                </div>

                <span
                  className={`status-pill status-${selectedRecord.response.status}`}
                >
                  {selectedRecord.response.status}
                </span>
              </div>

              <div className="reports-context-actions">
                <button
                  type="button"
                  className="secondary-button compact"
                  onClick={refreshSelectedReport}
                  disabled={isRefreshing}
                >
                  <RefreshCw
                    size={15}
                    className={isRefreshing ? "spin" : ""}
                  />
                  {isRefreshing ? "Refreshing..." : "Refresh"}
                </button>

                <button
                  type="button"
                  className="primary-button compact"
                  onClick={executeSelectedReport}
                  disabled={isRunning}
                >
                  <FlaskConical size={15} />
                  {isRunning ? "Running..." : "Run report"}
                </button>
              </div>

              <div className="report-detail-grid">
                <div className="report-detail-item">
                  <span>Experiment</span>
                  <strong>{selectedRecord.response.experiment_name}</strong>
                </div>

                <div className="report-detail-item">
                  <span>Dataset</span>
                  <strong>{selectedRecord.response.dataset_name}</strong>
                </div>

                <div className="report-detail-item">
                  <span>Generated</span>
                  <strong>
                    {formatDateTime(selectedRecord.response.generated_at)}
                  </strong>
                </div>

                <div className="report-detail-item">
                  <span>Analysis window</span>
                  <strong>
                    {formatDateTime(
                      selectedRecord.request.start_timestamp,
                    )}{" "}
                    →{" "}
                    {formatDateTime(selectedRecord.request.end_timestamp)}
                  </strong>
                </div>
              </div>

              <div className="report-scope-section">
                <div className="report-section-heading">
                  <div>
                    <span className="section-kicker">ANALYSIS SCOPE</span>
                    <h3>Included components</h3>
                  </div>

                  <span className="scope-count">
                    {getScopeCount(selectedRecord.request)}/5
                  </span>
                </div>

                <div className="report-scope-grid">
                  <ScopeItem
                    label="Forecasts"
                    enabled={selectedRecord.request.include_forecasts}
                    icon={<Gauge size={16} />}
                  />

                  <ScopeItem
                    label="Risk"
                    enabled={selectedRecord.request.include_risk}
                    icon={<ShieldCheck size={16} />}
                  />

                  <ScopeItem
                    label="Portfolio"
                    enabled={selectedRecord.request.include_portfolio}
                    icon={<BarChart3 size={16} />}
                  />

                  <ScopeItem
                    label="Stress"
                    enabled={selectedRecord.request.include_stress}
                    icon={<Target size={16} />}
                  />

                  <ScopeItem
                    label="Statistical tests"
                    enabled={
                      selectedRecord.request.include_statistical_tests
                    }
                    icon={<FlaskConical size={16} />}
                  />
                </div>
              </div>

              <div className="report-output-section">
                <div className="report-section-heading">
                  <div>
                    <span className="section-kicker">REPORT OUTPUT</span>
                    <h3>Generated sections</h3>
                  </div>

                  <span className="scope-count">
                    {selectedRecord.response.sections.length}
                  </span>
                </div>

                {selectedRecord.response.sections.length === 0 ? (
                  <div className="report-output-empty">
                    <FileCheck2 size={20} />

                    <div>
                      <strong>No sections returned</strong>
                      <p>
                        The registered report currently contains no generated
                        sections.
                      </p>
                    </div>
                  </div>
                ) : (
                  <div className="report-sections">
                    {selectedRecord.response.sections.map(
                      (section, sectionIndex) => (
                        <article
                          className="report-section-card"
                          key={`${section.title}-${sectionIndex}`}
                        >
                          <div className="report-section-card-header">
                            <div className="report-section-number">
                              {String(sectionIndex + 1).padStart(2, "0")}
                            </div>

                            <div>
                              <h4>{section.title}</h4>
                            </div>
                          </div>

                          <p>{section.content}</p>

                          {Object.keys(section.metrics).length > 0 && (
                            <div className="report-metrics">
                              {Object.entries(section.metrics).map(
                                ([metric, value]) => (
                                  <div
                                    className="report-metric"
                                    key={metric}
                                  >
                                    <span>{metric}</span>
                                    <strong>
                                      {formatMetric(value)}
                                    </strong>
                                  </div>
                                ),
                              )}
                            </div>
                          )}
                        </article>
                      ),
                    )}
                  </div>
                )}
              </div>

              {runResult && (
                <div className="report-run-result">
                  <div className="report-run-result-header">
                    <div>
                      <span className="section-kicker">RUN RESULT</span>
                      <h3>Reporting workflow</h3>
                    </div>

                    <span
                      className={`status-pill status-${runResult.status}`}
                    >
                      {runResult.status}
                    </span>
                  </div>

                  {runResult.conclusion ? (
                    <div className="report-conclusion">
                      <div className="report-conclusion-row">
                        <span>Research question</span>
                        <p>{runResult.conclusion.research_question}</p>
                      </div>

                      <div className="report-conclusion-row">
                        <span>Null hypothesis</span>
                        <p>{runResult.conclusion.null_hypothesis}</p>
                      </div>

                      <div className="report-conclusion-row">
                        <span>Alternative hypothesis</span>
                        <p>
                          {runResult.conclusion.alternative_hypothesis}
                        </p>
                      </div>

                      <div className="report-conclusion-row">
                        <span>Conclusion</span>
                        <p>{runResult.conclusion.conclusion}</p>
                      </div>

                      <div className="report-conclusion-significance">
                        <div>
                          <span>Statistical significance</span>
                          <strong>
                            {runResult.conclusion.statistical_significance}
                          </strong>
                        </div>

                        <div>
                          <span>Economic significance</span>
                          <strong>
                            {runResult.conclusion.economic_significance}
                          </strong>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <p className="report-run-status">
                      The reporting workflow returned{" "}
                      <strong>{runResult.status}</strong>.
                    </p>
                  )}
                </div>
              )}

              <div className="report-context-footer">
                <History size={15} />
                <span>
                  Registered at{" "}
                  {formatDateTime(selectedRecord.response.generated_at)}
                </span>
              </div>
            </>
          )}
        </aside>
      </section>

      {isModalOpen && (
        <div
          className="modal-backdrop"
          role="presentation"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) {
              closeCreateModal();
            }
          }}
        >
          <div
            className="modal-card report-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="create-report-title"
          >
            <div className="modal-header">
              <div>
                <span className="section-kicker">NEW REPORT</span>
                <h2 id="create-report-title">
                  Register Research Output
                </h2>
                <p>
                  Define the experiment, reporting window, type, and analysis
                  components.
                </p>
              </div>

              <button
                type="button"
                className="icon-button"
                onClick={closeCreateModal}
                disabled={isSubmitting}
                aria-label="Close report form"
              >
                <X size={18} />
              </button>
            </div>

            <form onSubmit={handleSubmit}>
              <div className="report-form-body">
                <div className="report-form-grid">
                  <label className="form-field">
                    <span>Experiment name</span>
                    <input
                      type="text"
                      value={form.experiment_name}
                      onChange={(event) =>
                        updateForm(
                          "experiment_name",
                          event.target.value,
                        )
                      }
                      placeholder="e.g. avf_walk_forward_v1"
                    />
                  </label>

                  <label className="form-field">
                    <span>Dataset name</span>
                    <input
                      type="text"
                      value={form.dataset_name}
                      onChange={(event) =>
                        updateForm(
                          "dataset_name",
                          event.target.value,
                        )
                      }
                      placeholder="e.g. market_data_daily"
                    />
                  </label>
                </div>

                <label className="form-field">
                  <span>Report type</span>
                  <input
                    type="text"
                    value={form.report_type}
                    onChange={(event) =>
                      updateForm(
                        "report_type",
                        event.target.value,
                      )
                    }
                    placeholder="e.g. Full Research Report"
                  />

                  <div className="report-type-suggestions">
                    {REPORT_TYPE_SUGGESTIONS.map((suggestion) => (
                      <button
                        type="button"
                        key={suggestion}
                        className={
                          form.report_type === suggestion
                            ? "suggestion-chip is-active"
                            : "suggestion-chip"
                        }
                        onClick={() =>
                          updateForm("report_type", suggestion)
                        }
                      >
                        {suggestion}
                      </button>
                    ))}
                  </div>
                </label>

                <div className="report-form-grid">
                  <label className="form-field">
                    <span>Start timestamp</span>
                    <input
                      type="datetime-local"
                      value={form.start_timestamp}
                      onChange={(event) =>
                        updateForm(
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
                      value={form.end_timestamp}
                      onChange={(event) =>
                        updateForm(
                          "end_timestamp",
                          event.target.value,
                        )
                      }
                    />
                  </label>
                </div>

                <div className="report-scope-form">
                  <div className="report-form-heading">
                    <div>
                      <span className="section-kicker">REPORT SCOPE</span>
                      <h3>Include analysis components</h3>
                    </div>

                    <span>
                      {getScopeCount(form)}/5 selected
                    </span>
                  </div>

                  <div className="report-scope-form-grid">
                    <ScopeToggle
                      label="Forecasts"
                      description="Volatility forecasts and model outputs"
                      enabled={form.include_forecasts}
                      icon={<Gauge size={18} />}
                      onClick={() =>
                        toggleScope("include_forecasts")
                      }
                    />

                    <ScopeToggle
                      label="Risk"
                      description="VaR, Expected Shortfall and risk analysis"
                      enabled={form.include_risk}
                      icon={<ShieldCheck size={18} />}
                      onClick={() =>
                        toggleScope("include_risk")
                      }
                    />

                    <ScopeToggle
                      label="Portfolio"
                      description="Portfolio decisions and performance"
                      enabled={form.include_portfolio}
                      icon={<BarChart3 size={18} />}
                      onClick={() =>
                        toggleScope("include_portfolio")
                      }
                    />

                    <ScopeToggle
                      label="Stress"
                      description="Stress and robustness analysis"
                      enabled={form.include_stress}
                      icon={<Target size={18} />}
                      onClick={() =>
                        toggleScope("include_stress")
                      }
                    />

                    <ScopeToggle
                      label="Statistical tests"
                      description="Research significance and validation"
                      enabled={form.include_statistical_tests}
                      icon={<FlaskConical size={18} />}
                      onClick={() =>
                        toggleScope(
                          "include_statistical_tests",
                        )
                      }
                    />
                  </div>
                </div>
              </div>

              <div className="modal-footer">
                <button
                  type="button"
                  className="secondary-button"
                  onClick={closeCreateModal}
                  disabled={isSubmitting}
                >
                  Cancel
                </button>

                <button
                  type="submit"
                  className="primary-button"
                  disabled={isSubmitting}
                >
                  <FileCheck2 size={16} />
                  {isSubmitting
                    ? "Registering..."
                    : "Register report"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </main>
  );
}

function ScopeItem({
  label,
  enabled,
  icon,
}: {
  label: string;
  enabled: boolean;
  icon: React.ReactNode;
}) {
  return (
    <div
      className={`report-scope-item ${
        enabled ? "is-enabled" : "is-disabled"
      }`}
    >
      <div className="report-scope-item-icon">{icon}</div>

      <div>
        <strong>{label}</strong>
        <span>{enabled ? "Included" : "Excluded"}</span>
      </div>

      <div className="report-scope-state">
        {enabled ? <CheckCircle2 size={15} /> : <X size={15} />}
      </div>
    </div>
  );
}

function ScopeToggle({
  label,
  description,
  enabled,
  icon,
  onClick,
}: {
  label: string;
  description: string;
  enabled: boolean;
  icon: React.ReactNode;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      className={`report-scope-toggle ${
        enabled ? "is-enabled" : ""
      }`}
      onClick={onClick}
    >
      <div className="report-scope-toggle-icon">{icon}</div>

      <div className="report-scope-toggle-content">
        <strong>{label}</strong>
        <span>{description}</span>
      </div>

      <div className="report-scope-toggle-check">
        {enabled && <CheckCircle2 size={18} />}
      </div>
    </button>
  );
}