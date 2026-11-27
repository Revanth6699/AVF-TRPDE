import {
  Activity,
  BarChart3,
  Database,
  FlaskConical,
  LineChart,
  ShieldAlert,
  WalletCards,
  FileText,
  LayoutDashboard,
  ChevronRight,
  Waves,
} from "lucide-react";
import { NavLink } from "react-router-dom";

const workspaceItems = [
  { label: "Overview", path: "/", icon: LayoutDashboard },
  { label: "Datasets", path: "/datasets", icon: Database },
  { label: "Experiments", path: "/experiments", icon: FlaskConical },
];

const researchItems = [
  { label: "Forecasts", path: "/forecasts", icon: LineChart },
  { label: "Risk engine", path: "/risk", icon: ShieldAlert },
  { label: "Portfolios", path: "/portfolios", icon: WalletCards },
  { label: "Stress testing", path: "/stress", icon: Waves },
  { label: "Reports", path: "/reports", icon: FileText },
];

export default function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="brand-mark"><Activity size={21} strokeWidth={1.9} /></div>
        <div className="brand-copy">
          <div className="brand-name">AVF<span> / </span>TRPDE</div>
          <div className="brand-subtitle">Quantitative research</div>
        </div>
      </div>
      <div className="sidebar-divider" />
      <nav className="sidebar-nav" aria-label="Main navigation">
        <NavSection label="WORKSPACE" items={workspaceItems} />
        <NavSection label="ANALYSIS" items={researchItems} />
      </nav>
      <div className="sidebar-bottom">
        <div className="sidebar-research-note">
          <BarChart3 size={15} />
          <span>Research environment</span>
        </div>
        <div className="sidebar-footer">MODEL · RISK · DECISION</div>
      </div>
    </aside>
  );
}

type Item = {
  label: string;
  path: string;
  icon: React.ComponentType<{ size?: number; strokeWidth?: number }>;
};

function NavSection({ label, items }: { label: string; items: Item[] }) {
  return (
    <div className="nav-section">
      <div className="nav-section-label">{label}</div>
      <div className="nav-list">
        {items.map((item) => (
          <NavLink key={item.path} to={item.path} end={item.path === "/"}
            className={({ isActive }) => `nav-item ${isActive ? "nav-item-active" : ""}`}>
            <span className="nav-item-icon"><item.icon size={17} strokeWidth={1.8} /></span>
            <span className="nav-item-label">{item.label}</span>
            <ChevronRight className="nav-item-arrow" size={14} />
          </NavLink>
        ))}
      </div>
    </div>
  );
}
