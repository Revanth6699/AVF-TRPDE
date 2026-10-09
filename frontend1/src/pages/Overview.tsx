import {
  Activity,
  ArrowUpRight,
  BarChart3,
  BrainCircuit,
  Database,
  FlaskConical,
  Gauge,
  LineChart,
  ShieldAlert,
  WalletCards,
} from "lucide-react";

const models = [
  {
    name: "EWMA",
    description: "Exponentially weighted volatility baseline",
    tone: "cyan",
  },
  {
    name: "GJR-GARCH",
    description: "Asymmetric conditional volatility model",
    tone: "violet",
  },
  {
    name: "XGBoost",
    description: "Machine-learning volatility forecaster",
    tone: "blue",
  },
  {
    name: "Regime-XGBoost",
    description: "Regime-aware ML volatility forecaster",
    tone: "green",
  },
];

const modules = [
  {
    title: "Volatility Forecasting",
    description:
      "Compare baseline, GARCH, ML and regime-aware volatility forecasts.",
    icon: LineChart,
    path: "/forecasts",
  },
  {
    title: "Tail-Risk Engine",
    description:
      "Evaluate filtered historical simulation, VaR, ES and backtests.",
    icon: ShieldAlert,
    path: "/risk",
  },
  {
    title: "Portfolio Decisions",
    description:
      "Analyze volatility-targeted portfolio construction and costs.",
    icon: WalletCards,
    path: "/portfolios",
  },
  {
    title: "Research Reports",
    description:
      "Bring forecasting, risk, portfolio and statistical evidence together.",
    icon: BarChart3,
    path: "/reports",
  },
];

export default function Overview() {
  return (
    <div className="page overview-page">
      <section className="overview-hero">
        <div className="overview-hero-copy">
          <div className="page-eyebrow">Adaptive quantitative research</div>

          <h1>
            Volatility.
            <br />
            <span>Risk.</span> Decisions.
          </h1>

          <p>
            AVF-TRPDE evaluates adaptive volatility forecasting, tail-risk
            behaviour and portfolio decisions through a controlled research
            pipeline.
          </p>

          <div className="overview-actions">
            <a href="/experiments" className="overview-primary-action">
              <FlaskConical size={16} />
              Open experiments
              <ArrowUpRight size={14} />
            </a>

            <a href="/datasets" className="overview-secondary-action">
              <Database size={16} />
              Inspect datasets
            </a>
          </div>
        </div>

        <div className="overview-hero-visual">
          <div className="hero-grid" />

          <div className="volatility-orbit orbit-one" />
          <div className="volatility-orbit orbit-two" />
          <div className="volatility-orbit orbit-three" />

          <div className="hero-core">
            <Activity size={30} strokeWidth={1.4} />
          </div>

          <div className="hero-readout readout-top">
            <span>FORECAST</span>
            <strong>VOLATILITY</strong>
          </div>

          <div className="hero-readout readout-bottom">
            <span>REGIME</span>
            <strong>ADAPTIVE</strong>
          </div>
        </div>
      </section>

      <section className="overview-section">
        <div className="section-heading">
          <div>
            <div className="section-kicker">Research architecture</div>
            <h2>Model layer</h2>
          </div>

          <span className="section-context">
            4 specified forecasting models
          </span>
        </div>

        <div className="model-grid">
          {models.map((model, index) => (
            <article
              className={`model-card model-${model.tone}`}
              key={model.name}
            >
              <div className="model-card-top">
                <span className="model-index">
                  0{index + 1}
                </span>

                <Gauge size={16} />
              </div>

              <h3>{model.name}</h3>

              <p>{model.description}</p>

              <div className="model-card-line">
                <span />
              </div>

              <div className="model-card-footer">
                <span>Research model</span>
                <BrainCircuit size={13} />
              </div>
            </article>
          ))}
        </div>
      </section>

      <section className="overview-section">
        <div className="section-heading">
          <div>
            <div className="section-kicker">Research workflow</div>
            <h2>Analysis modules</h2>
          </div>

          <span className="section-context">
            Forecast → Risk → Portfolio → Report
          </span>
        </div>

        <div className="module-grid">
          {modules.map((module, index) => {
            const Icon = module.icon;

            return (
              <a
                className="module-card"
                href={module.path}
                key={module.title}
              >
                <div className="module-number">
                  0{index + 1}
                </div>

                <div className="module-icon">
                  <Icon size={19} strokeWidth={1.7} />
                </div>

                <div className="module-content">
                  <h3>{module.title}</h3>
                  <p>{module.description}</p>
                </div>

                <ArrowUpRight
                  className="module-arrow"
                  size={17}
                  strokeWidth={1.7}
                />
              </a>
            );
          })}
        </div>
      </section>

      <section className="overview-status">
        <div className="status-panel">
          <div className="status-panel-icon">
            <Activity size={18} />
          </div>

          <div>
            <div className="status-panel-label">
              Research environment
            </div>

            <h3>Awaiting experiment execution</h3>

            <p>
              The interface is connected to the research architecture. Empirical
              outputs will appear here only after the corresponding backend
              research workflow has produced them.
            </p>
          </div>

          <div className="status-panel-state">
            <span />
            Ready
          </div>
        </div>
      </section>
    </div>
  );
}