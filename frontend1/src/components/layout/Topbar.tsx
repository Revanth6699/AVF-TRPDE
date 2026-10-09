import {
  Bell,
  ChevronDown,
  Command,
  Moon,
  Search,
  SlidersHorizontal,
  Sun,
} from "lucide-react";
import { useEffect, useState } from "react";

type Theme = "light" | "dark";

function getInitialTheme(): Theme {
  const saved = localStorage.getItem("avf-theme");

  if (saved === "dark" || saved === "light") {
    return saved;
  }

  return "light";
}

export default function Topbar() {
  const [theme, setTheme] = useState<Theme>(getInitialTheme);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("avf-theme", theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme((current) => (current === "light" ? "dark" : "light"));
  };

  return (
    <header className="topbar">
      <div className="topbar-search">
        <Search size={17} strokeWidth={1.8} />

        <input
          type="text"
          placeholder="Search datasets, experiments, models..."
          aria-label="Search research workspace"
        />

        <div className="search-shortcut">
          <Command size={12} />
          <span>K</span>
        </div>
      </div>

      <div className="topbar-actions">
        <div className="workspace-status">
          <span className="workspace-status-dot" />
          <span>Research</span>
        </div>

        <button
          className="topbar-icon-button"
          type="button"
          aria-label={`Switch to ${
            theme === "light" ? "dark" : "light"
          } theme`}
          onClick={toggleTheme}
        >
          {theme === "light" ? (
            <Moon size={17} strokeWidth={1.8} />
          ) : (
            <Sun size={17} strokeWidth={1.8} />
          )}
        </button>

        <button
          className="topbar-icon-button notification-button"
          type="button"
          aria-label="Notifications"
        >
          <Bell size={17} strokeWidth={1.8} />
          <span className="notification-dot" />
        </button>

        <button
          className="topbar-icon-button"
          type="button"
          aria-label="Workspace settings"
        >
          <SlidersHorizontal size={17} strokeWidth={1.8} />
        </button>

        <div className="topbar-divider" />

        <button className="researcher-profile" type="button">
          <span className="profile-avatar">AV</span>

          <span className="profile-copy">
            <span className="profile-name">Researcher</span>
            <span className="profile-role">AVF-TRPDE</span>
          </span>

          <ChevronDown size={15} strokeWidth={1.8} />
        </button>
      </div>
    </header>
  );
}