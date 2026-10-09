import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import AppShell from "./components/layout/AppShell";
import Overview from "./pages/Overview";
import Datasets from "./pages/Datasets";
import Experiments from "./pages/Experiments";
import Forecasts from "./pages/Forecasts";
import Risk from "./pages/Risk";
import Portfolios from "./pages/Portfolios";
import Reports from "./pages/Reports";

function App() {
  return (
    <BrowserRouter>
      <AppShell>
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/datasets" element={<Datasets />} />
          <Route path="/experiments" element={<Experiments />} />
          <Route path="/forecasts" element={<Forecasts />} />
          <Route path="/risk" element={<Risk />} />
          <Route path="/portfolios" element={<Portfolios />} />
          <Route path="/reports" element={<Reports />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </AppShell>
    </BrowserRouter>
  );
}

export default App;
