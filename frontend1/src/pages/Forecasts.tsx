import { useEffect, useMemo, useState } from "react";
import { Activity, BarChart3, CalendarRange, Check, Database, Download, LineChart, Play, RefreshCw } from "lucide-react";
import { CartesianGrid, Line, LineChart as RechartsLineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { createForecast, compareForecasts, evaluateForecast, getForecast, type ForecastComparisonResponse, type ForecastEvaluationResponse, type ForecastRequest, type ForecastResponse } from "../api/forecasts";
import { getDatasets } from "../api/datasets";
import { getExperiments, type ExperimentResponse } from "../api/experiments";
import type { DatasetListItem } from "../types/api";

const MODELS = ["EWMA", "GJR-GARCH", "XGBoost", "Regime-XGBoost"];
function errorMessage(error: unknown): string {
  const value = error as { response?: { data?: { detail?: string } }; message?: string };
  return value.response?.data?.detail || value.message || "Forecast request failed.";
}
function toLocalInput(value: string | null | undefined): string {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  const local = new Date(date.getTime() - date.getTimezoneOffset() * 60000);
  return local.toISOString().slice(0, 16);
}
function toIso(value: string): string {
  return new Date(value).toISOString();
}
function fmt(value: number | null | undefined, digits = 5): string {
  return value === null || value === undefined || !Number.isFinite(value) ? "—" : value.toFixed(digits);
}

export default function Forecasts() {
  const [experiments, setExperiments] = useState<ExperimentResponse[]>([]);
  const [datasets, setDatasets] = useState<DatasetListItem[]>([]);
  const [experimentName, setExperimentName] = useState("");
  const [datasetName, setDatasetName] = useState("");
  const [model, setModel] = useState("EWMA");
  const [assets, setAssets] = useState<string[]>([]);
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [comparison, setComparison] = useState<ForecastComparisonResponse | null>(null);
  const [evaluation, setEvaluation] = useState<ForecastEvaluationResponse | null>(null);
  const [busy, setBusy] = useState<"load" | "run" | "evaluate" | "compare" | null>(null);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  useEffect(() => {
    let active = true;
    Promise.all([getExperiments(), getDatasets()]).then(([expResponse, dataResponse]) => {
      if (!active) return;
      setExperiments(expResponse.experiments);
      setDatasets(dataResponse.datasets);
      const saved = sessionStorage.getItem("avf-selected-experiment") || expResponse.experiments[0]?.name || "";
      setExperimentName(saved);
      const selected = expResponse.experiments.find((item) => item.name === saved) ?? expResponse.experiments[0];
      const linkedDataset = String(selected?.parameters?.dataset_name ?? "");
      if (linkedDataset) setDatasetName(linkedDataset);
      const dataset = dataResponse.datasets.find((item) => item.name === linkedDataset);
      if (dataset) {
        setAssets(dataset.assets);
        setStart(toLocalInput(dataset.start_timestamp));
        setEnd(toLocalInput(dataset.end_timestamp));
      }
    }).catch((requestError) => { if (active) setError(errorMessage(requestError)); });
    return () => { active = false; };
  }, []);

  const selectedDataset = datasets.find((item) => item.name === datasetName);
  const selectedExperiment = experiments.find((item) => item.name === experimentName);
  const availableAssets = selectedDataset?.assets ?? [];

  function chooseExperiment(name: string) {
    setExperimentName(name);
    sessionStorage.setItem("avf-selected-experiment", name);
    const experiment = experiments.find((item) => item.name === name);
    const linkedDataset = String(experiment?.parameters?.dataset_name ?? "");
    if (linkedDataset) chooseDataset(linkedDataset);
  }

  function chooseDataset(name: string) {
    setDatasetName(name);
    const dataset = datasets.find((item) => item.name === name);
    if (dataset) {
      setAssets(dataset.assets);
      setStart(toLocalInput(dataset.start_timestamp));
      setEnd(toLocalInput(dataset.end_timestamp));
    } else {
      setAssets([]);
      setStart("");
      setEnd("");
    }
  }

  function buildRequest(): ForecastRequest {
    if (!experimentName || !datasetName || !model || assets.length === 0 || !start || !end) {
      throw new Error("Select an experiment, linked dataset, at least one asset, and a valid date range.");
    }
    if (new Date(start).getTime() > new Date(end).getTime()) throw new Error("Start date must be before end date.");
    return {
      experiment_name: experimentName,
      dataset_name: datasetName,
      model,
      assets,
      start_timestamp: toIso(start),
      end_timestamp: toIso(end),
    };
  }

  async function perform(action: "load" | "run" | "evaluate" | "compare") {
    setBusy(action); setError(""); setNotice("");
    try {
      if (action === "load") {
        if (!experimentName.trim()) throw new Error("Choose a registered experiment.");
        const result = await getForecast(experimentName, model);
        setForecast(result);
        setDatasetName(result.dataset_name);
        setNotice(`Loaded ${result.observation_count.toLocaleString()} stored observations from the backend.`);
      } else {
        const request = buildRequest();
        if (action === "run") {
          const result = await createForecast(request);
          setForecast(result); setComparison(null); setEvaluation(null);
          setNotice(`Forecast generation completed: ${result.observation_count.toLocaleString()} observations.`);
        } else if (action === "evaluate") {
          const result = await evaluateForecast(request);
          setEvaluation(result);
          setNotice("Forecast evaluation completed using the selected experiment, model, assets and target-date range.");
        } else {
          const result = await compareForecasts(request);
          setComparison(result);
          setNotice("Model comparison completed on the requested research scope.");
        }
      }
    } catch (requestError) {
      setError(errorMessage(requestError));
      if (action === "run") setForecast(null);
    } finally {
      setBusy(null);
    }
  }

  const chartData = useMemo(() => (forecast?.forecasts ?? []).map((point) => ({
    date: new Date(point.timestamp).toLocaleDateString(),
    predicted: point.predicted_volatility,
    actual: point.actual_volatility ?? null,
  })), [forecast]);

  return (
    <main className="page forecasts-page">
      <header className="module-header">
        <div>
          <div className="page-eyebrow">OUT-OF-SAMPLE FORECAST EVALUATION</div>
          <h1>Volatility forecasts</h1>
          <p className="page-description">Generate, retrieve and evaluate forecasts using the registered experiment's dataset. The scope below is sent as-is to the backend; no wildcard assets or placeholder dates are used.</p>
        </div>
      </header>

      <section className="surface forecast-scope">
        <div className="forecast-scope-heading"><div><span className="section-kicker">RESEARCH SCOPE</span><h2>Choose what to evaluate</h2></div><span className="forecast-scope-meta"><Database size={14} /> {datasets.length} registered datasets</span></div>
        <div className="forecast-form-grid">
          <label className="forecast-field"><span>Experiment</span><select value={experimentName} onChange={(event) => chooseExperiment(event.target.value)}><option value="">Select experiment</option>{experiments.map((item) => <option key={item.name} value={item.name}>{item.name}</option>)}</select></label>
          <label className="forecast-field"><span>Dataset</span><select value={datasetName} onChange={(event) => chooseDataset(event.target.value)}><option value="">Select dataset</option>{datasets.map((item) => <option key={item.name} value={item.name}>{item.name}</option>)}</select></label>
          <label className="forecast-field"><span>Model</span><select value={model} onChange={(event) => setModel(event.target.value)}>{MODELS.map((item) => <option key={item} value={item}>{item}</option>)}</select></label>
          <label className="forecast-field"><span>Start date</span><input type="datetime-local" value={start} onChange={(event) => setStart(event.target.value)} /></label>
          <label className="forecast-field"><span>End date</span><input type="datetime-local" value={end} onChange={(event) => setEnd(event.target.value)} /></label>
        </div>
        <div className="forecast-assets">
          <div className="forecast-assets-label"><span>Asset universe</span><small>{assets.length} of {availableAssets.length} selected</small></div>
          {availableAssets.length ? <div className="forecast-asset-list">{availableAssets.map((asset) => <label key={asset} className={assets.includes(asset) ? "forecast-asset selected" : "forecast-asset"}><input type="checkbox" checked={assets.includes(asset)} onChange={(event) => setAssets((current) => event.target.checked ? [...current, asset] : current.filter((item) => item !== asset))} /><span>{asset}</span><Check size={13} /></label>)}</div> : <p className="forecast-scope-hint">Choose a dataset that contains registered asset identifiers. If this experiment has no dataset_name parameter, select its dataset manually.</p>}
        </div>
        <div className="forecast-action-row">
          <button className="secondary-button" type="button" onClick={() => void perform("load")} disabled={busy !== null}><RefreshCw size={14} />Load saved forecast</button>
          <button className="primary-button" type="button" onClick={() => void perform("run")} disabled={busy !== null}><Play size={14} />{busy === "run" ? "Generating…" : "Generate forecast"}</button>
          <button className="secondary-button" type="button" onClick={() => void perform("evaluate")} disabled={busy !== null}><Activity size={14} />Evaluate</button>
          <button className="secondary-button" type="button" onClick={() => void perform("compare")} disabled={busy !== null}><BarChart3 size={14} />Compare models</button>
        </div>
        {error && <div className="inline-alert" role="alert">{error}</div>}
        {notice && <div className="forecast-notice" role="status">{notice}</div>}
        {selectedExperiment && <p className="forecast-scope-hint">Experiment dataset binding: {String(selectedExperiment.parameters?.dataset_name ?? "not configured")}.</p>}
      </section>

      {forecast && (
        <section className="forecast-output">
          <div className="forecast-output-heading"><div><span className="section-kicker">BACKEND RESULT</span><h2>{forecast.model} · {forecast.dataset_name}</h2></div><span className="status-pill">{forecast.status}</span></div>
          <div className="forecast-output-metrics">
            <div><span>Observations</span><strong>{forecast.observation_count.toLocaleString()}</strong></div>
            <div><span>Assets</span><strong>{forecast.asset_count.toLocaleString()}</strong></div>
            <div><span>Model</span><strong>{forecast.model}</strong></div>
          </div>
          {chartData.length > 0 ? <div className="forecast-chart-panel">
            <div className="forecast-chart-title"><div><LineChart size={16} /><strong>Predicted vs actual volatility</strong></div><span>Target timestamp</span></div>
            <div className="forecast-chart-container"><ResponsiveContainer width="100%" height="100%"><RechartsLineChart data={chartData} margin={{ top: 12, right: 18, bottom: 4, left: 4 }}><CartesianGrid strokeDasharray="2 5" stroke="var(--border)" /><XAxis dataKey="date" tick={{ fill: "var(--text-muted)", fontSize: 9 }} minTickGap={28} /><YAxis tick={{ fill: "var(--text-muted)", fontSize: 9 }} width={54} /><Tooltip contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 6, color: "var(--text-primary)", fontSize: 11 }} /><Line type="monotone" dataKey="predicted" name="Predicted" stroke="#347c60" strokeWidth={2} dot={false} connectNulls /><Line type="monotone" dataKey="actual" name="Actual" stroke="#b47a45" strokeWidth={1.6} dot={false} connectNulls /></RechartsLineChart></ResponsiveContainer></div>
          </div> : <p className="forecast-scope-hint">No forecast observations were returned.</p>}
          <div className="table-wrap"><table className="data-table"><thead><tr><th>Target timestamp</th><th>Asset</th><th>Predicted volatility</th><th>Actual volatility</th></tr></thead><tbody>{forecast.forecasts.slice(0, 100).map((point, index) => <tr key={point.timestamp + point.asset_id + index}><td>{new Date(point.timestamp).toLocaleString()}</td><td>{point.asset_id}</td><td>{fmt(point.predicted_volatility)}</td><td>{fmt(point.actual_volatility)}</td></tr>)}</tbody></table></div>
          {forecast.forecasts.length > 100 && <p className="forecast-scope-hint">Showing the first 100 of {forecast.forecasts.length.toLocaleString()} observations.</p>}
        </section>
      )}

      {evaluation && <section className="surface forecast-evaluation"><div className="forecast-output-heading"><div><span className="section-kicker">FORECAST METRICS</span><h2>Evaluation result</h2></div><strong>{evaluation.observation_count.toLocaleString()} observations</strong></div><div className="forecast-evaluation-grid">{evaluation.metrics.map((metric, index) => <div key={metric.model + metric.metric + index}><span>{metric.model} · {metric.metric}</span><strong>{fmt(metric.value, 6)}</strong></div>)}</div></section>}

      {comparison && <section className="surface forecast-evaluation"><div className="forecast-output-heading"><div><span className="section-kicker">COMMON OUT-OF-SAMPLE COMPARISON</span><h2>Model comparison</h2></div><strong>{comparison.observation_count.toLocaleString()} observations</strong></div><div className="table-wrap"><table className="data-table"><thead><tr><th>Model</th><th>MAE</th><th>RMSE</th><th>QLIKE</th></tr></thead><tbody>{comparison.models.map((item) => <tr key={item.model}><td>{item.model}</td><td>{fmt(item.mae, 6)}</td><td>{fmt(item.rmse, 6)}</td><td>{fmt(item.qlike, 6)}</td></tr>)}</tbody></table></div></section>}
    </main>
  );
}
