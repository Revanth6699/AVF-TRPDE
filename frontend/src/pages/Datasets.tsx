import {
  CheckCircle2,
  Database,
  Plus,
  RefreshCw,
  Search,
  X,
  AlertCircle,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { createDataset, getDatasets } from "../api/datasets";
import type {
  DatasetListItem,
  DatasetRequest,
} from "../types/api";

type DatasetForm = DatasetRequest;

const emptyForm: DatasetForm = {
  name: "",
  source: "",
  frequency: "",
  assets: [],
  start_timestamp: "",
  end_timestamp: "",
  row_count: 0,
  fingerprint: "",
};

export default function Datasets() {
  const [datasets, setDatasets] = useState<DatasetListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState("");
  const [search, setSearch] = useState("");
  const [showCreate, setShowCreate] = useState(false);
  const [form, setForm] = useState<DatasetForm>(emptyForm);
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState("");

  async function loadDatasets(showRefresh = false) {
    try {
      if (showRefresh) {
        setRefreshing(true);
      } else {
        setLoading(true);
      }

      setError("");

      const response = await getDatasets();
      setDatasets(response.datasets);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to load datasets from the backend.",
      );
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }

  useEffect(() => {
    void loadDatasets();
  }, []);

  const filteredDatasets = useMemo(() => {
    const query = search.trim().toLowerCase();

    if (!query) {
      return datasets;
    }

    return datasets.filter((dataset) => {
      return (
        dataset.name.toLowerCase().includes(query) ||
        dataset.source.toLowerCase().includes(query) ||
        dataset.frequency.toLowerCase().includes(query) ||
        dataset.fingerprint.toLowerCase().includes(query)
      );
    });
  }, [datasets, search]);

  const registeredAssets = datasets.reduce(
    (total, dataset) => total + dataset.asset_count,
    0,
  );

  async function handleCreate(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();

    try {
      setCreating(true);
      setCreateError("");

      const payload: DatasetRequest = {
        ...form,
        assets: form.assets
          .map((asset) => asset.trim())
          .filter(Boolean),
        row_count: Number(form.row_count),
      };

      const response = await createDataset(payload);

      const listItem: DatasetListItem = {
        name: response.name,
        source: response.source,
        frequency: response.frequency,
        asset_count: response.assets.length,
        row_count: response.row_count,
        start_timestamp: response.start_timestamp,
        end_timestamp: response.end_timestamp,
        fingerprint: response.fingerprint,
      };

      setDatasets((current) => [...current, listItem]);

      setForm(emptyForm);
      setShowCreate(false);
    } catch (err) {
      setCreateError(
        err instanceof Error
          ? err.message
          : "Unable to register the dataset.",
      );
    } finally {
      setCreating(false);
    }
  }

  function updateForm<K extends keyof DatasetForm>(
    field: K,
    value: DatasetForm[K],
  ) {
    setForm((current) => ({
      ...current,
      [field]: value,
    }));
  }

  function openCreateModal() {
    setCreateError("");
    setForm(emptyForm);
    setShowCreate(true);
  }

  return (
    <section className="page datasets-page">
      <header className="module-header">
        <div>
          <div className="page-eyebrow">Research data layer</div>

          <h1>Datasets</h1>

          <p className="page-description">
            Register and inspect the market datasets used by the
            AVF-TRPDE research pipeline.
          </p>
        </div>

        <div className="module-header-actions">
          <button
            className="secondary-button"
            type="button"
            onClick={() => void loadDatasets(true)}
            disabled={loading || refreshing}
          >
            <RefreshCw
              size={15}
              className={refreshing ? "spin-icon" : ""}
            />
            Refresh
          </button>

          <button
            className="primary-button"
            type="button"
            onClick={openCreateModal}
          >
            <Plus size={16} />
            Register dataset
          </button>
        </div>
      </header>

      <section className="dataset-stat-grid">
        <StatCard
          label="Registered datasets"
          value={datasets.length}
          icon={<Database size={17} />}
        />

        <StatCard
          label="Validation status"
          value="—"
          icon={<CheckCircle2 size={17} />}
        />

        <StatCard
          label="Registered assets"
          value={registeredAssets}
          icon={<Database size={17} />}
        />

        <StatCard
          label="Backend state"
          value={error ? "Error" : loading ? "Loading" : "Connected"}
          icon={
            error ? (
              <AlertCircle size={17} />
            ) : (
              <CheckCircle2 size={17} />
            )
          }
          tone={error ? "danger" : "success"}
        />
      </section>

      {error && (
        <div className="inline-error">
          <AlertCircle size={16} />
          <span>{error}</span>
        </div>
      )}

      <section className="dataset-workspace surface">
        <div className="dataset-toolbar">
          <div className="dataset-toolbar-title">
            <Database size={16} />

            <div>
              <strong>Research datasets</strong>

              <span>
                {datasets.length} registered{" "}
                {datasets.length === 1 ? "dataset" : "datasets"}
              </span>
            </div>
          </div>

          <label className="dataset-search">
            <Search size={15} />

            <input
              type="search"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search datasets, sources or fingerprints..."
            />
          </label>
        </div>

        {loading ? (
          <DatasetLoading />
        ) : filteredDatasets.length === 0 ? (
          <DatasetEmpty
            hasSearch={Boolean(search.trim())}
            onCreate={openCreateModal}
          />
        ) : (
          <div className="dataset-table-wrapper">
            <table className="dataset-table">
              <thead>
                <tr>
                  <th>Dataset</th>
                  <th>Source</th>
                  <th>Frequency</th>
                  <th>Assets</th>
                  <th>Rows</th>
                  <th>Period</th>
                  <th>Fingerprint</th>
                </tr>
              </thead>

              <tbody>
                {filteredDatasets.map((dataset) => (
                  <tr
                    key={`${dataset.name}-${dataset.fingerprint}`}
                  >
                    <td>
                      <div className="dataset-name-cell">
                        <span className="dataset-icon">
                          <Database size={15} />
                        </span>

                        <div>
                          <strong>{dataset.name}</strong>

                          <span className="dataset-fingerprint">
                            {dataset.fingerprint || "No fingerprint"}
                          </span>
                        </div>
                      </div>
                    </td>

                    <td>
                      {dataset.source || "—"}
                    </td>

                    <td>
                      <span className="frequency-badge">
                        {dataset.frequency || "—"}
                      </span>
                    </td>

                    <td>
                      <span className="asset-count">
                        {dataset.asset_count}{" "}
                        {dataset.asset_count === 1
                          ? "asset"
                          : "assets"}
                      </span>
                    </td>

                    <td className="mono-value">
                      {dataset.row_count.toLocaleString()}
                    </td>

                    <td>
                      <div className="date-range">
                        <span>
                          {formatDate(dataset.start_timestamp)}
                        </span>

                        <span>→</span>

                        <span>
                          {formatDate(dataset.end_timestamp)}
                        </span>
                      </div>
                    </td>

                    <td>
                      <span className="dataset-table-fingerprint">
                        {dataset.fingerprint || "—"}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {showCreate && (
        <div
          className="modal-backdrop"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) {
              setShowCreate(false);
            }
          }}
        >
          <div className="dataset-modal">
            <div className="modal-header">
              <div>
                <div className="page-eyebrow">
                  Dataset metadata
                </div>

                <h2>Register Dataset Metadata</h2>

                <p>
                  Enter the metadata for an existing research
                  dataset. This information is used to register
                  it with the AVF-TRPDE research pipeline.
                </p>
              </div>

              <button
                className="modal-close"
                type="button"
                onClick={() => setShowCreate(false)}
                aria-label="Close"
              >
                <X size={17} />
              </button>
            </div>

            <form onSubmit={handleCreate}>
              <div className="form-grid">
                <FormField
                  label="Dataset name"
                  value={form.name}
                  onChange={(value) =>
                    updateForm("name", value)
                  }
                  placeholder="e.g. NIFTY50"
                  required
                />

                <FormField
                  label="Source"
                  value={form.source}
                  onChange={(value) =>
                    updateForm("source", value)
                  }
                  placeholder="e.g. Alpha Vantage"
                  required
                />

                <FormField
                  label="Frequency"
                  value={form.frequency}
                  onChange={(value) =>
                    updateForm("frequency", value)
                  }
                  placeholder="e.g. daily"
                  required
                />

                <FormField
                  label="Assets / Symbols"
                  value={form.assets.join(", ")}
                  onChange={(value) =>
                    updateForm(
                      "assets",
                      value
                        .split(",")
                        .map((asset) => asset.trim())
                        .filter(Boolean),
                    )
                  }
                  placeholder="e.g. NIFTY50, SPY"
                  required
                />

                <FormField
                  label="Start timestamp"
                  type="datetime-local"
                  value={toDateInputValue(
                    form.start_timestamp,
                  )}
                  onChange={(value) =>
                    updateForm(
                      "start_timestamp",
                      value
                        ? new Date(value).toISOString()
                        : "",
                    )
                  }
                  required
                />

                <FormField
                  label="End timestamp"
                  type="datetime-local"
                  value={toDateInputValue(
                    form.end_timestamp,
                  )}
                  onChange={(value) =>
                    updateForm(
                      "end_timestamp",
                      value
                        ? new Date(value).toISOString()
                        : "",
                    )
                  }
                  required
                />

                <FormField
                  label="Dataset row count"
                  type="number"
                  value={String(form.row_count)}
                  onChange={(value) =>
                    updateForm(
                      "row_count",
                      Number(value),
                    )
                  }
                  min="0"
                  required
                />

                <FormField
                  label="Dataset fingerprint"
                  value={form.fingerprint}
                  onChange={(value) =>
                    updateForm("fingerprint", value)
                  }
                  placeholder="Dataset fingerprint"
                  required
                />
              </div>

              <div className="metadata-note">
                <Database size={15} />

                <span>
                  These fields describe the dataset being
                  registered. The current backend stores the
                  metadata you provide and does not automatically
                  derive these values from a dataset file.
                </span>
              </div>

              {createError && (
                <div className="form-error">
                  <AlertCircle size={15} />
                  {createError}
                </div>
              )}

              <div className="modal-footer">
                <button
                  className="secondary-button"
                  type="button"
                  onClick={() => setShowCreate(false)}
                  disabled={creating}
                >
                  Cancel
                </button>

                <button
                  className="primary-button"
                  type="submit"
                  disabled={creating}
                >
                  {creating ? (
                    <>
                      <RefreshCw
                        size={15}
                        className="spin-icon"
                      />
                      Registering...
                    </>
                  ) : (
                    <>
                      <Plus size={15} />
                      Register metadata
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

function StatCard({
  label,
  value,
  icon,
  tone = "default",
}: {
  label: string;
  value: string | number;
  icon: React.ReactNode;
  tone?: "default" | "success" | "danger";
}) {
  return (
    <div className={`dataset-stat-card stat-${tone}`}>
      <div className="dataset-stat-icon">
        {icon}
      </div>

      <div>
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
    </div>
  );
}

function DatasetLoading() {
  return (
    <div className="dataset-loading">
      {[1, 2, 3, 4].map((item) => (
        <div
          className="dataset-skeleton-row"
          key={item}
        >
          <span />
          <span />
          <span />
          <span />
          <span />
        </div>
      ))}
    </div>
  );
}

function DatasetEmpty({
  hasSearch,
  onCreate,
}: {
  hasSearch: boolean;
  onCreate: () => void;
}) {
  return (
    <div className="dataset-empty">
      <div className="dataset-empty-icon">
        <Database size={23} />
      </div>

      <h2>
        {hasSearch
          ? "No matching datasets"
          : "No datasets registered"}
      </h2>

      <p>
        {hasSearch
          ? "Try a different dataset, source or fingerprint search."
          : "Register research dataset metadata to make it available to the AVF-TRPDE pipeline."}
      </p>

      {!hasSearch && (
        <button
          className="primary-button"
          type="button"
          onClick={onCreate}
        >
          <Plus size={15} />
          Register first dataset
        </button>
      )}
    </div>
  );
}

function FormField({
  label,
  value,
  onChange,
  placeholder,
  type = "text",
  min,
  required = false,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  type?: string;
  min?: string;
  required?: boolean;
}) {
  return (
    <label className="form-field">
      <span>{label}</span>

      <input
        type={type}
        value={value}
        onChange={(event) =>
          onChange(event.target.value)
        }
        placeholder={placeholder}
        min={min}
        required={required}
      />
    </label>
  );
}

function formatDate(value: string | null) {
  if (!value) {
    return "—";
  }

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return new Intl.DateTimeFormat("en-IN", {
    year: "numeric",
    month: "short",
    day: "2-digit",
  }).format(date);
}

function toDateInputValue(value: string) {
  if (!value) {
    return "";
  }

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return "";
  }

  const local = new Date(
    date.getTime() -
      date.getTimezoneOffset() * 60_000,
  );

  return local.toISOString().slice(0, 16);
}