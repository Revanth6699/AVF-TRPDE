import {
  ArrowRight, Activity, Database, FileText, FlaskConical,
  LineChart, ShieldCheck, WalletCards, Workflow,
} from "lucide-react";
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { getDatasets } from "../api/datasets";
import { getExperiments } from "../api/experiments";
import { getHealth } from "../api/system";

type WorkspaceCounts = { datasets: number | null; experiments: number | null; api: "checking" | "online" | "offline" };

const stages = [
  { number: "01", title: "Register data", detail: "Define the market source, frequency, asset universe and date coverage.", path: "/datasets", icon: Database },
  { number: "02", title: "Configure experiment", detail: "Choose the four supported forecasting models, metrics and walk-forward parameters.", path: "/experiments", icon: FlaskConical },
  { number: "03", title: "Execute research", detail: "Run the registered configuration and persist out-of-sample predictions.", path: "/experiments", icon: Workflow },
  { number: "04", title: "Evaluate forecasts", detail: "Inspect forecast trajectories, MAE, RMSE, QLIKE and model comparisons.", path: "/forecasts", icon: LineChart },
  { number: "05", title: "Measure tail risk", detail: "Generate FHS VaR and ES estimates and run VaR coverage backtests.", path: "/risk", icon: ShieldCheck },
  { number: "06", title: "Test portfolio", detail: "Construct portfolio strategies and evaluate constraints and transaction costs.", path: "/portfolios", icon: WalletCards },
  { number: "07", title: "Stress & report", detail: "Run retrospective scenarios and assemble evidence-based research reports.", path: "/stress", icon: FileText },
];

export default function Overview() {
  const [counts, setCounts] = useState<WorkspaceCounts>({ datasets: null, experiments: null, api: "checking" });

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
      <section className="overview-heading">
        <div>
          <div className="page-eyebrow">QUANTITATIVE RESEARCH WORKSPACE</div>
          <h1>From market data<br /><span>to defensible decisions.</span></h1>
          <p>One chronological research pipeline for volatility forecasting, tail-risk measurement and portfolio analysis. Every result shown here comes from a backend operation—not a simulated dashboard metric.</p>
          <div className="overview-heading-actions">
            <Link to="/datasets" className="primary-button"><Database size={15} /> Start with a dataset <ArrowRight size={15} /></Link>
            <Link to="/experiments" className="secondary-button">Open experiment register</Link>
          </div>
        </div>
        <div className="research-visual" aria-label="Abstract three-dimensional market volatility plot">
          <div className="visual-axis visual-axis-x" />
          <div className="visual-axis visual-axis-y" />
          <div className="visual-axis visual-axis-z" />
          <div className="visual-plane">
            <div className="visual-grid" />
            <div className="visual-ribbon ribbon-back" />
            <div className="visual-ribbon ribbon-front" />
            <span className="visual-point point-a" />
            <span className="visual-point point-b" />
            <span className="visual-point point-c" />
          </div>
          <div className="visual-caption"><span>VOLATILITY SURFACE</span><strong>WALK-FORWARD · OUT-OF-SAMPLE</strong></div>
          <span className="visual-coordinate coordinate-one">σ(t)</span>
          <span className="visual-coordinate coordinate-two">t + 1</span>
        </div>
      </section>

      <section className="overview-status-row" aria-label="Workspace status">
        <article className="overview-status-card">
          <span className="overview-status-label">Backend API</span>
          <strong className={`overview-status-value status-${counts.api}`}>{counts.api === "checking" ? "Checking" : counts.api === "online" ? "Connected" : "Unavailable"}</strong>
          <span className="overview-status-note">Live /health response</span>
        </article>
        <article className="overview-status-card">
          <span className="overview-status-label">Registered datasets</span>
          <strong>{counts.datasets === null ? "—" : counts.datasets.toLocaleString()}</strong>
          <span className="overview-status-note">Persisted dataset definitions</span>
        </article>
        <article className="overview-status-card">
          <span className="overview-status-label">Experiments</span>
          <strong>{counts.experiments === null ? "—" : counts.experiments.toLocaleString()}</strong>
          <span className="overview-status-note">Saved research configurations</span>
        </article>
        <article className="overview-status-card">
          <span className="overview-status-label">Model suite</span>
          <strong>04</strong>
          <span className="overview-status-note">EWMA · GJR-GARCH · XGBoost · Regime-XGBoost</span>
        </article>
      </section>

      <section className="overview-workflow-section">
        <div className="overview-section-heading">
          <div><span className="section-kicker">THE RESEARCH SEQUENCE</span><h2>Work from the root</h2></div>
          <span className="section-context">DATA → MODEL → VALIDATION → DECISION</span>
        </div>
        <div className="workflow-list">
          {stages.map((stage, index) => {
            const Icon = stage.icon;
            return (
              <Link to={stage.path} className="workflow-stage" key={stage.number}>
                <span className="workflow-stage-number">{stage.number}</span>
                <span className="workflow-stage-icon"><Icon size={18} strokeWidth={1.8} /></span>
                <span className="workflow-stage-copy"><strong>{stage.title}</strong><small>{stage.detail}</small></span>
                <ArrowRight className="workflow-stage-arrow" size={17} />
                {index < stages.length - 1 && <span className="workflow-stage-connector" />}
              </Link>
            );
          })}
        </div>
      </section>

      <section className="overview-research-note">
        <div className="research-note-icon"><Activity size={18} /></div>
        <div><strong>Interpretation matters as much as execution.</strong><p>Backtest statistics depend on sample size; 99% VaR exceptions are sparse in short samples. The current real-market validation is BTCUSD-only, and Expected Shortfall does not yet have a separate statistical backtest.</p></div>
        <Link to="/reports" aria-label="Open research reports"><ArrowRight size={17} /></Link>
      </section>
    </main>
  );
}
