import { NavLink } from "react-router-dom";
import {
  LayoutDashboard,
  Plus,
  Users,
  Bell,
  FileBarChart2,
  BarChart3,
  Settings,
  HeartPulse,
} from "lucide-react";

const NAV_ITEMS = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard },
  { to: "/screening/new/details", label: "New Screening", icon: Plus },
  { to: "/patients", label: "Patients", icon: Users },
  { to: "/alerts", label: "Alerts", icon: Bell },
  { to: "/reports", label: "Reports", icon: FileBarChart2 },
  { to: "/analytics", label: "Community Analytics", icon: BarChart3 },
  { to: "/settings", label: "Settings", icon: Settings },
];

export default function Sidebar() {
  return (
    <aside className="hidden md:flex md:w-[220px] flex-shrink-0 flex-col border-r border-border bg-white h-full">
      <div className="flex items-center gap-2 px-5 py-5">
        <HeartPulse className="text-accent" size={26} />
        <div>
          <div className="text-sm font-medium leading-tight">ILASH Health</div>
          <div className="text-[10px] text-ink-muted leading-tight">
            AI-Powered Screening
          </div>
        </div>
      </div>

      <nav className="flex-1 px-3 flex flex-col gap-1 overflow-y-auto">
        {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === "/"}
            className={({ isActive }) =>
              `flex items-center gap-2.5 rounded-control px-3 py-2 text-sm transition-colors ${
                isActive
                  ? "bg-accent-soft text-accent font-medium"
                  : "text-ink-muted hover:bg-wash hover:text-ink"
              }`
            }
          >
            <Icon size={18} />
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="flex items-center gap-2.5 px-4 py-4 border-t border-border">
        <div className="w-9 h-9 rounded-full bg-accent-soft text-accent flex items-center justify-center text-xs font-medium">
          PS
        </div>
        <div className="text-xs leading-tight">
          <div className="font-medium">Dr. Priya Sharma</div>
          <div className="text-ink-muted">PHC Bengaluru</div>
        </div>
      </div>
    </aside>
  );
}
