import { Activity, ArrowRight, Database, FileText, FlaskConical, LineChart, ShieldCheck, WalletCards } from "lucide-react";
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { getDatasets } from "../api/datasets";
import { getExperiments } from "../api/experiments";
import { getHealth } from "../api/system";

type Counts = { datasets: number | null; experiments: number | null; api: "checking" | "online" | "offline" };

const workbenches = [
  { id: "01", title: "Dataset registry", detail: "Register market histories, inspect coverage and establish the asset universe before modelling.", tag: "INPUT", path: "/datasets", icon: Database },
  { id: "02", title: "Experiment design", detail: "Configure the model set, target, evaluation metrics and walk-forward validation window.", tag: "CONFIGURE", path: "/experiments", icon: FlaskConical },
  { id: "03", title: "Volatility forecasts", detail: "Review out-of-sample predictions and compare EWMA, GJR-GARCH and machine-learning models.", tag: "FORECAST", path: "/forecasts", icon: LineChart },
  { id: "04", title: "Tail-risk analysis", detail: "Inspect Value at Risk, Expected Shortfall estimates and available exception backtests.", tag: "RISK", path: "/risk", icon: ShieldCheck },
  { id: "05", title: "Portfolio research", detail: "Compare portfolio strategies, risk targeting and transaction-cost effects.", tag: "ALLOCATE", path: "/portfolios", icon: WalletCards },
  { id: "06", title: "Stress & reporting", detail: "Run historical stress scenarios and assemble reproducible research outputs.", tag: "OUTPUT", path: "/stress", icon: FileText },
];

export default function Overview() {
  const [counts, setCounts] = useState<Counts>({ datasets: null, experiments: null, api: "checking" });

  useEffect(() => {
    let active = true;
    Promise.allSettled([getHealth(), getDatasets(), getExperiments()]).then(([health, datasets, experiments]) => {
      if (!active) return;
      setCounts({
        api: health.status === "fulfilled" && health.value.status === "ok" ? "online" : "offline",
        datasets: datasets.status === "fulfilled" ? datasets.value.count : null,
        experiments: experiments.status === "fulfilled" ? experiments.value.count : null,
      });
    });
    return () => { active = false; };
  }, []);

  return (
    <main className="page overview-page">
      <header className="terminal-heading">
        <div>
          <div className="terminal-meta">AVF / TRPDE &nbsp;·&nbsp; QUANTITATIVE RESEARCH SYSTEM</div>
          <h1>Market evidence.<br /><span>Measured risk. Better decisions.</span></h1>
          <p>A research environment for adaptive volatility forecasting, tail-risk measurement and portfolio evaluation. The workspace reports persisted backend results; it does not manufacture sample performance figures.</p>
        </div>
        <div className="terminal-heading-actions">
          <Link to="/datasets" className="terminal-button primary"><Database size={14} /> Add dataset</Link>
          <Link to="/experiments" className="terminal-button"><FlaskConical size={14} /> New experiment</Link>
        </div>
      </header>

      <section className="terminal-kpis" aria-label="Live workspace status">
        <article className="terminal-kpi">
          <span className="terminal-kpi-label">API service</span>
          <strong className={`terminal-kpi-value ${counts.api === "online" ? "terminal-state-online" : counts.api === "offline" ? "terminal-state-offline" : ""}`}>{counts.api === "checking" ? "Checking" : counts.api === "online" ? "Online" : "Offline"}</strong>
          <span className="terminal-kpi-note">Live health endpoint</span>
        </article>
        <article className="terminal-kpi">
          <span className="terminal-kpi-label">Registered datasets</span>
          <strong className="terminal-kpi-value">{counts.datasets === null ? "—" : counts.datasets.toLocaleString()}</strong>
          <span className="terminal-kpi-note">Dataset definitions in registry</span>
        </article>
        <article className="terminal-kpi">
          <span className="terminal-kpi-label">Experiments</span>
          <strong className="terminal-kpi-value">{counts.experiments === null ? "—" : counts.experiments.toLocaleString()}</strong>
          <span className="terminal-kpi-note">Saved research configurations</span>
        </article>
        <article className="terminal-kpi">
          <span className="terminal-kpi-label">Forecast models</span>
          <strong className="terminal-kpi-value">04</strong>
          <span className="terminal-kpi-note">EWMA · GJR-GARCH · XGBoost · Regime-XGBoost</span>
        </article>
      </section>

      <section>
        <div className="terminal-section-head">
          <div><div className="section-kicker">RESEARCH WORKBENCH</div><h2>Choose an analysis module</h2></div>
          <p>INPUT → MODEL → VALIDATION → DECISION</p>
        </div>
        <div className="terminal-workbench">
          {workbenches.map((item) => {
            const Icon = item.icon;
            return (
              <Link to={item.path} className="terminal-workbench-row" key={item.id}>
                <span className="terminal-index">{item.id}</span>
                <span className="terminal-workbench-title">{item.title}</span>
                <span className="terminal-workbench-detail">{item.detail}</span>
                <span className="terminal-workbench-tag">{item.tag}</span>
                <ArrowRight className="terminal-arrow" size={15} />
                <span className="sr-only"><Icon size={1} /></span>
              </Link>
            );
          })}
        </div>
      </section>

      <aside className="terminal-footnote">
        <Activity size={17} />
        <div><strong>Research integrity note</strong><p>Forecast and risk statistics are only meaningful with sufficient observations and correctly aligned daily/intraday data. Current validation limitations must remain visible in any report; no placeholder metrics are presented as empirical results.</p></div>
        <Link to="/reports" aria-label="Open research reports"><ArrowRight size={16} /></Link>
      </aside>
    </main>
  );
}
