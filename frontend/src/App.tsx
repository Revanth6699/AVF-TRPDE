import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import AppShell from "./components/layout/AppShell";
import Overview from "./pages/Overview";
import Datasets from "./pages/Datasets";
import Experiments from "./pages/Experiments";

function PlaceholderPage({ title }: { title: string }) {
  return (
    <section className="page">
      <div className="page-header">
        <div className="page-header-copy">
          <div className="page-eyebrow">AVF-TRPDE</div>

          <h1>{title}</h1>

          <p className="page-description">
            This research module is being connected to the AVF-TRPDE backend.
          </p>
        </div>
      </div>

      <div className="surface surface-padding">
        <div className="state">
          <div>
            <h2 className="state-title">Module ready</h2>

            <p className="state-description">
              Research outputs will appear here when the corresponding
              backend workflow produces them.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}

function App() {
  return (
    <BrowserRouter>
      <AppShell>
        <Routes>
          <Route path="/" element={<Overview />} />

          <Route path="/datasets" element={<Datasets />} />

          <Route
            path="/experiments"
            element={<Experiments />}
          />

          <Route
            path="/forecasts"
            element={<PlaceholderPage title="Volatility Forecasts" />}
          />

          <Route
            path="/risk"
            element={<PlaceholderPage title="Tail Risk Engine" />}
          />

          <Route
            path="/portfolios"
            element={<PlaceholderPage title="Portfolio Analysis" />}
          />

          <Route
            path="/reports"
            element={<PlaceholderPage title="Research Reports" />}
          />

          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </AppShell>
    </BrowserRouter>
  );
}

export default App;