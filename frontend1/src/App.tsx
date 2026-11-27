import { lazy, Suspense } from "react";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import AppShell from "./components/layout/AppShell";

const Overview = lazy(() => import("./pages/Overview"));
const Datasets = lazy(() => import("./pages/Datasets"));
const Experiments = lazy(() => import("./pages/Experiments"));
const Forecasts = lazy(() => import("./pages/Forecasts"));
const Risk = lazy(() => import("./pages/Risk"));
const Portfolios = lazy(() => import("./pages/Portfolios"));
const Stress = lazy(() => import("./pages/Stress"));
const Reports = lazy(() => import("./pages/Reports"));

function App() {
  return (
    <BrowserRouter>
      <AppShell>
        <Suspense fallback={<div className="page route-loading" role="status">Loading research module…</div>}>
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/datasets" element={<Datasets />} />
            <Route path="/experiments" element={<Experiments />} />
            <Route path="/forecasts" element={<Forecasts />} />
            <Route path="/risk" element={<Risk />} />
            <Route path="/portfolios" element={<Portfolios />} />
            <Route path="/stress" element={<Stress />} />
            <Route path="/reports" element={<Reports />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </Suspense>
      </AppShell>
    </BrowserRouter>
  );
}

export default App;
