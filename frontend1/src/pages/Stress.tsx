import { Activity, AlertTriangle, RefreshCw, Shield, Zap } from "lucide-react";
import { useState } from "react";
import { getStressResults, runStressAnalysis, type StressRunResponse } from "../api/stress";

function messageFrom(error: unknown): string {
  const value = error as { response?: { data?: { detail?: string } }; message?: string };
  return value.response?.data?.detail || value.message || "Stress analysis request failed.";
}

export default function Stress() {
  const [experimentName, setExperimentName] = useState("");
  const [strategy, setStrategy] = useState("EWMA");
  const [records, setRecords] = useState<StressRunResponse[]>([]);
  const [busy, setBusy] = useState<"run" | "load" | null>(null);
  const [error, setError] = useState("");

  async function run() {
    if (!experimentName.trim() || !strategy.trim()) {
      setError("Choose a completed experiment and portfolio strategy first.");
      return;
    }
    setBusy("run");
    setError("");
    try {
      const result = await runStressAnalysis({
        experiment_name: experimentName.trim(),
        strategy: strategy.trim(),
      });
      setRecords((current) => [result, ...current.filter(
        (item) => !(item.experiment_name === result.experiment_name && item.strategy === result.strategy),
      )]);
    } catch (requestError) {
      setError(messageFrom(requestError));
    } finally {
      setBusy(null);
    }
  }

  async function load() {
    if (!experimentName.trim()) {
      setError("Enter an experiment name to retrieve its saved stress results.");
      return;
    }
    setBusy("load");
    setError("");
    try {
      const result = await getStressResults(experimentName.trim(), strategy.trim() || undefined);
      setRecords(result);
      if (!result.length) setError("No saved stress analysis was found for this experiment and strategy.");
    } catch (requestError) {
      setError(messageFrom(requestError));
    } finally {
      setBusy(null);
    }
  }

  return (
    <section className="page stress-page">
      <header className="module-header">
        <div>
          <div className="page-eyebrow">PORTFOLIO STRESS RESEARCH</div>
          <h1>Stress testing</h1>
          <p className="page-description">
            Run retrospective scenario analysis against a completed portfolio's persisted net-return series.
            These scenarios describe historical portfolio behaviour; they are not forecasts.
          </p>
        </div>
      </header>

      <section className="stress-config surface surface-padding">
        <div className="stress-config-intro">
          <div className="stress-icon"><Shield size={19} /></div>
          <div>
            <h2>Scenario execution</h2>
            <p>Stress analysis requires a completed portfolio and at least 20 usable return observations.</p>
          </div>
        </div>
        <div className="stress-form-grid">
          <label className="field">
            <span>Experiment name</span>
            <input value={experimentName} onChange={(event) => setExperimentName(event.target.value)} placeholder="e.g. btc-volatility-study" />
          </label>
          <label className="field">
            <span>Portfolio strategy</span>
            <select value={strategy} onChange={(event) => setStrategy(event.target.value)}>
              {["Buy & Hold", "EWMA", "GJR-GARCH", "XGBoost", "Regime-XGBoost"].map((item) => <option key={item} value={item}>{item}</option>)}
            </select>
          </label>
        </div>
        <div className="stress-actions">
          <button className="primary-button" type="button" onClick={run} disabled={busy !== null}>
            {busy === "run" ? <RefreshCw className="spin" size={15} /> : <Zap size={15} />}
            Run scenarios
          </button>
          <button className="secondary-button" type="button" onClick={load} disabled={busy !== null}>
            {busy === "load" ? <RefreshCw className="spin" size={15} /> : <Activity size={15} />}
            Load saved results
          </button>
        </div>
        {error && <div className="inline-alert" role="alert"><AlertTriangle size={16} />{error}</div>}
      </section>

      {records.length > 0 ? (
        <section className="stress-results">
          {records.map((record) => (
            <article className="surface stress-result-card" key={record.experiment_name + record.strategy}>
              <div className="stress-result-heading">
                <div>
                  <span className="section-kicker">SCENARIO OUTPUT</span>
                  <h2>{record.experiment_name} <span>/ {record.strategy}</span></h2>
                </div>
                <span className="status-pill">{record.status}</span>
              </div>
              <div className="stress-metrics">
                <div><span>Scenarios</span><strong>{record.scenario_count}</strong></div>
                <div><span>Scenario observations</span><strong>{record.total_observations.toLocaleString()}</strong></div>
                <div><span>Valid scenarios</span><strong>{String(record.summary.valid_scenario_count ?? "—")}</strong></div>
              </div>
              {record.results.length > 0 && (
                <div className="table-wrap">
                  <table className="data-table">
                    <thead><tr>{Object.keys(record.results[0]).map((key) => <th key={key}>{key.replaceAll("_", " ")}</th>)}</tr></thead>
                    <tbody>{record.results.map((row, index) => (
                      <tr key={String(row.scenario ?? row.scenario_name ?? index)}>
                        {Object.keys(record.results[0]).map((key) => <td key={key}>{row[key] === null || row[key] === undefined ? "—" : typeof row[key] === "number" ? row[key].toLocaleString(undefined, { maximumFractionDigits: 5 }) : String(row[key])}</td>)}
                      </tr>
                    ))}</tbody>
                  </table>
                </div>
              )}
              <details className="stress-summary-details">
                <summary>Inspect method and full summary</summary>
                <pre>{JSON.stringify(record.summary, null, 2)}</pre>
              </details>
              <p className="stress-method-note">Method: retrospective descriptive scenario selection on persisted portfolio net returns. No forward-looking probability is implied.</p>
            </article>
          ))}
        </section>
      ) : (
        <div className="empty-workspace surface">
          <div className="empty-workspace-mark"><AlertTriangle size={21} /></div>
          <h2>No stress output loaded</h2>
          <p>Run scenarios after portfolio execution, or retrieve a previously saved stress analysis.</p>
        </div>
      )}
    </section>
  );
}
