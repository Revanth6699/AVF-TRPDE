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
} from "lucide-react";
import { NavLink } from "react-router-dom";

const workspaceItems = [
  {
    label: "Overview",
    path: "/",
    icon: LayoutDashboard,
  },
  {
    label: "Datasets",
    path: "/datasets",
    icon: Database,
  },
  {
    label: "Experiments",
    path: "/experiments",
    icon: FlaskConical,
  },
];

const researchItems = [
  {
    label: "Forecasts",
    path: "/forecasts",
    icon: LineChart,
  },
  {
    label: "Risk Engine",
    path: "/risk",
    icon: ShieldAlert,
  },
  {
    label: "Portfolios",
    path: "/portfolios",
    icon: WalletCards,
  },
  {
    label: "Reports",
    path: "/reports",
    icon: FileText,
  },
];

export default function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="brand-mark">
          <Activity size={22} strokeWidth={2.2} />
        </div>

        <div className="brand-copy">
          <div className="brand-name">AVF-TRPDE</div>
          <div className="brand-subtitle">Research Engine</div>
        </div>
      </div>

      <div className="sidebar-divider" />

      <nav className="sidebar-nav">
        <div className="nav-section">
          <div className="nav-section-label">WORKSPACE</div>

          <div className="nav-list">
            {workspaceItems.map((item) => (
              <NavItem key={item.path} {...item} />
            ))}
          </div>
        </div>

        <div className="nav-section research-section">
          <div className="nav-section-label">RESEARCH</div>

          <div className="nav-list">
            {researchItems.map((item) => (
              <NavItem key={item.path} {...item} />
            ))}
          </div>
        </div>
      </nav>

      <div className="sidebar-bottom">
        <div className="engine-status">
          <span className="status-indicator" />

          <div>
            <div className="status-title">Engine online</div>
            <div className="status-subtitle">API connection ready</div>
          </div>
        </div>

        <div className="sidebar-footer">
          <BarChart3 size={14} />
          <span>Research workspace</span>
        </div>
      </div>
    </aside>
  );
}

type NavItemProps = {
  label: string;
  path: string;
  icon: React.ComponentType<{ size?: number; strokeWidth?: number }>;
};

function NavItem({ label, path, icon: Icon }: NavItemProps) {
  return (
    <NavLink
      to={path}
      end={path === "/"}
      className={({ isActive }) =>
        `nav-item ${isActive ? "nav-item-active" : ""}`
      }
    >
      <span className="nav-item-icon">
        <Icon size={17} strokeWidth={1.8} />
      </span>

      <span className="nav-item-label">{label}</span>

      <ChevronRight
        className="nav-item-arrow"
        size={14}
        strokeWidth={1.8}
      />
    </NavLink>
  );
}