import {
  BarChart3,
  CheckCircle2,
  CircleDollarSign,
  Database,
  Layers3,
  Play,
  Plus,
  RefreshCw,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  TrendingUp,
  WalletCards,
  X,
} from "lucide-react";
import { useMemo, useState } from "react";
import type { FormEvent, KeyboardEvent } from "react";

import {
  createPortfolio,
  type PortfolioRequest,
  type PortfolioResponse,
} from "../api/portfolios";

type Strategy =
  | "Buy & Hold"
  | "EWMA"
  | "GJR-GARCH"
  | "XGBoost"
  | "Regime-XGBoost";

type PortfolioRecord = PortfolioResponse;

const STRATEGIES: Strategy[] = [
  "Buy & Hold",
  "EWMA",
  "GJR-GARCH",
  "XGBoost",
  "Regime-XGBoost",
];

const EMPTY_FORM: PortfolioRequest = {
  experiment_name: "",
  dataset_name: "",
  strategy: "EWMA",
  assets: [],
  start_timestamp: "",
  end_timestamp: "",
  volatility_target: 0.1,
  max_position: 1,
  max_turnover: 1,
  transaction_cost: 0,
};

function formatNumber(value: number) {
  return new Intl.NumberFormat("en-IN", {
    maximumFractionDigits: 4,
  }).format(value);
}

function formatDate(value: string) {
  if (!value) return "—";

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return date.toLocaleString();
}

function statusLabel(status: string) {
  return status
    .replace(/[-_]/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export default function Portfolios() {
  const [records, setRecords] = useState<PortfolioRecord[]>([]);
  const [form, setForm] = useState<PortfolioRequest>(EMPTY_FORM);
  const [assetInput, setAssetInput] = useState("");
  const [search, setSearch] = useState("");
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selected, setSelected] = useState<PortfolioRecord | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const filteredRecords = useMemo(() => {
    const query = search.trim().toLowerCase();

    if (!query) return records;

    return records.filter((record) =>
      [
        record.experiment_name,
        record.dataset_name,
        record.strategy,
        record.status,
      ]
        .join(" ")
        .toLowerCase()
        .includes(query),
    );
  }, [records, search]);

  const acceptedCount = records.filter(
    (record) => record.status.toLowerCase() === "accepted",
  ).length;

  const completedCount = records.filter(
    (record) =>
      record.status.toLowerCase() === "completed" ||
      record.status.toLowerCase() === "complete",
  ).length;

  const totalAssets = records.reduce(
    (total, record) => total + record.asset_count,
    0,
  );

  function openModal() {
    setError("");
    setForm(EMPTY_FORM);
    setAssetInput("");
    setIsModalOpen(true);
  }

  function closeModal() {
    if (loading) return;
    setIsModalOpen(false);
  }

  function addAsset() {
    const asset = assetInput.trim().toUpperCase();

    if (!asset) return;

    if (!form.assets.includes(asset)) {
      setForm((current) => ({
        ...current,
        assets: [...current.assets, asset],
      }));
    }

    setAssetInput("");
  }

  function removeAsset(asset: string) {
    setForm((current) => ({
      ...current,
      assets: current.assets.filter((item) => item !== asset),
    }));
  }

function handleAssetKeyDown(
  event: KeyboardEvent<HTMLInputElement>,
){
        if (event.key === "Enter") {
      event.preventDefault();
      addAsset();
    }
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");

    if (!form.experiment_name.trim()) {
      setError("Experiment name is required.");
      return;
    }

    if (!form.dataset_name.trim()) {
      setError("Dataset name is required.");
      return;
    }

    if (form.assets.length === 0) {
      setError("Add at least one asset.");
      return;
    }

    if (!form.start_timestamp || !form.end_timestamp) {
      setError("Start and end timestamps are required.");
      return;
    }

    setLoading(true);

    try {
      const response = await createPortfolio({
        ...form,
        experiment_name: form.experiment_name.trim(),
        dataset_name: form.dataset_name.trim(),
      });

      setRecords((current) => [response, ...current]);
      setSelected(response);
      setIsModalOpen(false);
    } catch (requestError) {
      const message =
        requestError instanceof Error
          ? requestError.message
          : "Unable to create portfolio analysis.";

      setError(message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="portfolio-page">
      <section className="page-hero portfolio-hero">
        <div>
          <div className="eyebrow">PORTFOLIO RESEARCH</div>

          <h1>Portfolios</h1>

          <p>
            Configure risk-targeted portfolio experiments, inspect registered
            strategies, and review portfolio outputs produced by the research
            pipeline.
          </p>
        </div>

        <div className="page-hero-actions">
          <button
            className="secondary-button"
            type="button"
            onClick={() => setSelected(null)}
          >
            <RefreshCw size={15} />
            Reset view
          </button>

          <button
            className="primary-button"
            type="button"
            onClick={openModal}
          >
            <Plus size={16} />
            New portfolio
          </button>
        </div>
      </section>

      <section className="stat-grid portfolio-stat-grid">
        <article className="stat-card">
          <div className="stat-icon stat-icon-teal">
            <WalletCards size={18} />
          </div>

          <div>
            <span>REGISTERED</span>
            <strong>{records.length}</strong>
          </div>
        </article>

        <article className="stat-card">
          <div className="stat-icon stat-icon-violet">
            <Play size={18} />
          </div>

          <div>
            <span>ACCEPTED</span>
            <strong>{acceptedCount}</strong>
          </div>
        </article>

        <article className="stat-card">
          <div className="stat-icon stat-icon-green">
            <CheckCircle2 size={18} />
          </div>

          <div>
            <span>COMPLETED</span>
            <strong>{completedCount}</strong>
          </div>
        </article>

        <article className="stat-card">
          <div className="stat-icon stat-icon-blue">
            <Database size={18} />
          </div>

          <div>
            <span>ASSET REFERENCES</span>
            <strong>{totalAssets}</strong>
          </div>
        </article>
      </section>

      <section className="portfolio-toolbar">
        <div className="search-field">
          <Search size={17} />
          <input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search experiments, datasets or strategies..."
          />
        </div>

        <div className="toolbar-meta">
          <SlidersHorizontal size={15} />
          <span>
            {filteredRecords.length}{" "}
            {filteredRecords.length === 1 ? "portfolio" : "portfolios"}
          </span>
        </div>
      </section>

      <section className="portfolio-workspace">
        <div className="portfolio-list-panel">
          <div className="section-heading">
            <div>
              <div className="section-kicker">PORTFOLIO REGISTER</div>
              <h2>Research portfolios</h2>
            </div>
          </div>

          {filteredRecords.length === 0 ? (
            <div className="portfolio-empty-state">
              <div className="empty-icon">
                <WalletCards size={23} />
              </div>

              <h3>No portfolio analyses registered</h3>

              <p>
                Configure an experiment, strategy, asset universe, risk target
                and portfolio constraints to register a portfolio analysis.
              </p>

              <button
                className="primary-button"
                type="button"
                onClick={openModal}
              >
                <Plus size={16} />
                Configure portfolio
              </button>
            </div>
          ) : (
            <div className="portfolio-record-list">
              {filteredRecords.map((record) => (
                <button
                  className={`portfolio-record ${
                    selected?.experiment_name === record.experiment_name &&
                    selected?.strategy === record.strategy
                      ? "portfolio-record-selected"
                      : ""
                  }`}
                  key={`${record.experiment_name}-${record.strategy}`}
                  type="button"
                  onClick={() => setSelected(record)}
                >
                  <div className="portfolio-record-icon">
                    <TrendingUp size={18} />
                  </div>

                  <div className="portfolio-record-main">
                    <div className="portfolio-record-title">
                      <strong>{record.experiment_name}</strong>

                      <span
                        className={`portfolio-status portfolio-status-${record.status.toLowerCase()}`}
                      >
                        {statusLabel(record.status)}
                      </span>
                    </div>

                    <span className="portfolio-record-subtitle">
                      {record.dataset_name} · {record.strategy}
                    </span>

                    <div className="portfolio-record-meta">
                      <span>{record.asset_count} assets</span>
                      <span>{record.observation_count} observations</span>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>

        <aside className="portfolio-context-panel">
          <div className="context-heading">
            <div className="context-icon">
              <ShieldCheck size={18} />
            </div>

            <div>
              <div className="section-kicker">PORTFOLIO CONTEXT</div>
              <h3>Research specification</h3>
            </div>
          </div>

          {selected ? (
            <>
              <div className="context-rows">
                <ContextRow
                  label="Experiment"
                  value={selected.experiment_name}
                />

                <ContextRow
                  label="Dataset"
                  value={selected.dataset_name}
                />

                <ContextRow
                  label="Strategy"
                  value={selected.strategy}
                />

                <ContextRow
                  label="Assets"
                  value={String(selected.asset_count)}
                />

                <ContextRow
                  label="Observations"
                  value={String(selected.observation_count)}
                />

                <ContextRow
                  label="Status"
                  value={statusLabel(selected.status)}
                />
              </div>

              <div className="context-note">
                <CheckCircle2 size={15} />
                <span>
                  Portfolio observations are displayed exactly as returned by
                  the backend.
                </span>
              </div>

              <div className="portfolio-position-preview">
                <div className="section-kicker">POSITIONS</div>

                {selected.positions.length === 0 ? (
                  <div className="mini-empty">
                    <CircleDollarSign size={16} />
                    <span>
                      No portfolio positions have been produced yet.
                    </span>
                  </div>
                ) : (
                  <div className="position-list">
                    {selected.positions.slice(0, 8).map((position) => (
                      <div
                        className="position-row"
                        key={`${position.timestamp}-${position.asset_id}`}
                      >
                        <span>{position.asset_id}</span>
                        <strong>{formatNumber(position.weight)}</strong>
                        <small>{formatDate(position.timestamp)}</small>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </>
          ) : (
            <div className="context-empty">
              <Layers3 size={23} />

              <h4>Select a portfolio</h4>

              <p>
                Registered portfolio specifications and generated positions
                will appear here.
              </p>
            </div>
          )}
        </aside>
      </section>

      <section className="portfolio-research-section">
        <div className="research-section-header">
          <div>
            <div className="section-kicker">PORTFOLIO EVALUATION</div>
            <h2>Performance comparison</h2>
            <p>
              Compare portfolio strategies only when the backend research
              pipeline returns actual comparison results.
            </p>
          </div>

          <button className="secondary-button" type="button">
            <BarChart3 size={16} />
            Compare strategies
          </button>
        </div>

        <div className="research-placeholder">
          <BarChart3 size={22} />

          <div>
            <strong>Comparison is research-backed</strong>
            <p>
              Total return, volatility, Sharpe, Sortino, drawdown, VaR, ES,
              turnover and transaction-cost metrics will appear here only when
              produced by the portfolio comparison pipeline.
            </p>
          </div>
        </div>
      </section>

      {isModalOpen && (
        <div
          className="modal-backdrop"
          role="presentation"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) {
              closeModal();
            }
          }}
        >
          <div
            className="portfolio-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="portfolio-modal-title"
          >
            <div className="modal-header">
              <div>
                <div className="section-kicker">PORTFOLIO CONFIGURATION</div>
                <h2 id="portfolio-modal-title">
                  Configure Portfolio Analysis
                </h2>
                <p>
                  Define the research context, risk target and portfolio
                  constraints.
                </p>
              </div>

              <button
                className="modal-close"
                type="button"
                onClick={closeModal}
                aria-label="Close"
              >
                <X size={18} />
              </button>
            </div>

            <form onSubmit={handleSubmit}>
              <div className="modal-body">
                <div className="form-grid">
                  <label className="form-field">
                    <span>EXPERIMENT NAME</span>
                    <input
                      value={form.experiment_name}
                      onChange={(event) =>
                        setForm((current) => ({
                          ...current,
                          experiment_name: event.target.value,
                        }))
                      }
                      placeholder="e.g. AVF_TEST_001"
                    />
                  </label>

                  <label className="form-field">
                    <span>DATASET NAME</span>
                    <input
                      value={form.dataset_name}
                      onChange={(event) =>
                        setForm((current) => ({
                          ...current,
                          dataset_name: event.target.value,
                        }))
                      }
                      placeholder="e.g. NIFTY50"
                    />
                  </label>

                  <label className="form-field">
                    <span>STRATEGY</span>
                    <select
                      value={form.strategy}
                      onChange={(event) =>
                        setForm((current) => ({
                          ...current,
                          strategy: event.target.value,
                        }))
                      }
                    >
                      {STRATEGIES.map((strategy) => (
                        <option key={strategy} value={strategy}>
                          {strategy}
                        </option>
                      ))}
                    </select>
                  </label>

                  <div className="form-field">
                    <span>ASSET UNIVERSE</span>

                    <div className="asset-input-row">
                      <input
                        value={assetInput}
                        onChange={(event) => setAssetInput(event.target.value)}
                        onKeyDown={handleAssetKeyDown}
                        placeholder="e.g. NIFTY50"
                      />

                      <button
                        className="secondary-button"
                        type="button"
                        onClick={addAsset}
                      >
                        Add
                      </button>
                    </div>

                    <div className="asset-chip-list">
                      {form.assets.map((asset) => (
                        <span className="asset-chip" key={asset}>
                          {asset}

                          <button
                            type="button"
                            onClick={() => removeAsset(asset)}
                            aria-label={`Remove ${asset}`}
                          >
                            <X size={12} />
                          </button>
                        </span>
                      ))}
                    </div>
                  </div>

                  <label className="form-field">
                    <span>START TIMESTAMP</span>
                    <input
                      type="datetime-local"
                      value={form.start_timestamp}
                      onChange={(event) =>
                        setForm((current) => ({
                          ...current,
                          start_timestamp: event.target.value,
                        }))
                      }
                    />
                  </label>

                  <label className="form-field">
                    <span>END TIMESTAMP</span>
                    <input
                      type="datetime-local"
                      value={form.end_timestamp}
                      onChange={(event) =>
                        setForm((current) => ({
                          ...current,
                          end_timestamp: event.target.value,
                        }))
                      }
                    />
                  </label>
                </div>

                <div className="configuration-divider">
                  <div className="section-kicker">
                    RISK TARGET & CONSTRAINTS
                  </div>
                </div>

                <div className="form-grid form-grid-four">
                  <label className="form-field">
                    <span>VOLATILITY TARGET</span>
                    <input
                      type="number"
                      min="0.000001"
                      step="0.01"
                      value={form.volatility_target}
                      onChange={(event) =>
                        setForm((current) => ({
                          ...current,
                          volatility_target: Number(event.target.value),
                        }))
                      }
                    />
                  </label>

                  <label className="form-field">
                    <span>MAX POSITION</span>
                    <input
                      type="number"
                      min="0.000001"
                      step="0.01"
                      value={form.max_position}
                      onChange={(event) =>
                        setForm((current) => ({
                          ...current,
                          max_position: Number(event.target.value),
                        }))
                      }
                    />
                  </label>

                  <label className="form-field">
                    <span>MAX TURNOVER</span>
                    <input
                      type="number"
                      min="0.000001"
                      step="0.01"
                      value={form.max_turnover}
                      onChange={(event) =>
                        setForm((current) => ({
                          ...current,
                          max_turnover: Number(event.target.value),
                        }))
                      }
                    />
                  </label>

                  <label className="form-field">
                    <span>TRANSACTION COST</span>
                    <input
                      type="number"
                      min="0"
                      step="0.0001"
                      value={form.transaction_cost}
                      onChange={(event) =>
                        setForm((current) => ({
                          ...current,
                          transaction_cost: Number(event.target.value),
                        }))
                      }
                    />
                  </label>
                </div>

                {error && (
                  <div className="form-error">
                    <ShieldCheck size={16} />
                    <span>{error}</span>
                  </div>
                )}
              </div>

              <div className="modal-footer">
                <button
                  className="secondary-button"
                  type="button"
                  onClick={closeModal}
                  disabled={loading}
                >
                  Cancel
                </button>

                <button
                  className="primary-button"
                  type="submit"
                  disabled={loading}
                >
                  <Play size={15} />
                  {loading ? "Registering..." : "Create portfolio analysis"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}

function ContextRow({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="context-row">
      <span>{label}</span>
      <strong>{value || "—"}</strong>
    </div>
  );
}