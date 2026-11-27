import { Moon, Sun, Wifi, WifiOff } from "lucide-react";
import { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";
import { getHealth } from "../../api/system";

type Theme = "light" | "dark";
const PAGE_NAMES: Record<string, string> = {
  "/": "Overview",
  "/datasets": "Datasets",
  "/experiments": "Experiments",
  "/forecasts": "Forecasts",
  "/risk": "Risk engine",
  "/portfolios": "Portfolios",
  "/stress": "Stress testing",
  "/reports": "Reports",
};

function getInitialTheme(): Theme {
  try {
    const saved = localStorage.getItem("avf-theme");
    return saved === "dark" ? "dark" : "light";
  } catch {
    return "light";
  }
}

export default function Topbar() {
  const [theme, setTheme] = useState<Theme>(getInitialTheme);
  const [apiState, setApiState] = useState<"checking" | "online" | "offline">("checking");
  const location = useLocation();

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    try { localStorage.setItem("avf-theme", theme); } catch { /* storage may be unavailable */ }
  }, [theme]);

  useEffect(() => {
    let active = true;
    getHealth()
      .then((health) => { if (active) setApiState(health.status === "ok" ? "online" : "offline"); })
      .catch(() => { if (active) setApiState("offline"); });
    return () => { active = false; };
  }, [location.pathname]);

  return (
    <header className="topbar">
      <div className="topbar-location">
        <span className="topbar-location-root">AVF-TRPDE</span>
        <span className="topbar-location-divider">/</span>
        <strong>{PAGE_NAMES[location.pathname] ?? "Workspace"}</strong>
      </div>
      <div className="topbar-actions">
        <div className={`api-connection api-connection-${apiState}`} role="status" aria-live="polite">
          {apiState === "offline" ? <WifiOff size={14} /> : <Wifi size={14} />}
          <span>{apiState === "checking" ? "Checking API" : apiState === "online" ? "API connected" : "API unavailable"}</span>
        </div>
        <span className="topbar-divider" />
        <button className="topbar-icon-button" type="button"
          aria-label={`Switch to ${theme === "light" ? "dark" : "light"} theme`}
          onClick={() => setTheme((current) => current === "light" ? "dark" : "light")}>
          {theme === "light" ? <Moon size={17} /> : <Sun size={17} />}
        </button>
      </div>
    </header>
  );
}
