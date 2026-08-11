import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Bell, Plus, Search, Eye } from "lucide-react";
import Header from "../components/Header";
import MetricCard from "../components/MetricCard";
import Sparkline from "../components/Sparkline";
import DonutChart from "../components/DonutChart";
import StatusBadge from "../components/StatusBadge";
import { fetchDashboardSummary } from "../api/client";
import { Patient, Status } from "../types";

export default function Dashboard() {
  const [data, setData] = useState<any>(null);

  useEffect(() => {
    fetchDashboardSummary().then(setData);
  }, []);

  const metrics = data?.metrics || {
    total_patients: 1248,
    screenings_today: 42,
    attention_required: 18,
    active_alerts: 5,
  };

  const hrTrend = data?.heart_rate_trend || [70, 74, 72, 78, 75, 80, 78];
  const dist = data?.status_distribution || { normal: 901, attention: 263, high: 84, total: 1248 };
  const recent: Patient[] = data?.recent_screenings || [];
  const riskIndicators = data?.top_risk_indicators || [
    { label: "Stress", cases: 21, color: "#8B5CF6" },
    { label: "High Heart Rate", cases: 12, color: "#E24B4A" },
    { label: "Low SpO2", cases: 9, color: "#378ADD" },
    { label: "Possible Fall Events", cases: 5, color: "#F5A623" },
    { label: "Irregular Heart Rate", cases: 3, color: "#1D9E75" },
  ];

  return (
    <>
      <Header
        title="Good morning, Dr. Sharma 👋"
        subtitle="Here's what's happening with your community today."
        showDate={false}
        actions={
          <>
            <div className="hidden lg:flex items-center gap-2 rounded-control border border-border px-3 py-2 text-xs text-ink-muted w-64">
              <Search size={14} />
              Search patients, screenings...
            </div>
            <button className="text-ink-muted hover:text-ink relative">
              <Bell size={20} />
              <span className="absolute -top-1 -right-1 w-2 h-2 bg-danger-dot rounded-full" />
            </button>
            <Link
              to="/screening/new/details"
              className="flex items-center gap-1.5 bg-gradient-to-r from-accent-light to-accent text-white text-sm font-medium rounded-control px-4 py-2"
            >
              <Plus size={16} /> New Screening
            </Link>
          </>
        }
      />

      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard label="Patients Screened" value={metrics.total_patients.toLocaleString()} trend="12% from last week" />
          <MetricCard label="Screenings Today" value={metrics.screenings_today.toString()} trend="8% from yesterday" />
          <MetricCard label="Attention Required" value={metrics.attention_required.toString()} trend="5% from yesterday" trendUp={false} />
          <MetricCard label="Active Alerts" value={metrics.active_alerts.toString()} trend="2 new today" />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-[1.4fr_1fr] gap-4">
          <div className="bg-white rounded-card border border-border p-5">
            <div className="flex items-center justify-between mb-3">
              <div className="text-sm font-medium">Heart Rate Trend (Community)</div>
              <div className="text-xs text-ink-muted border border-border rounded-control px-2 py-1">
                Last 7 Days
              </div>
            </div>
            <div className="flex items-baseline gap-2 mb-2">
              <span className="text-xl font-semibold">{hrTrend[hrTrend.length - 1] || 78}</span>
              <span className="text-xs text-ink-muted">BPM (10 Aug)</span>
            </div>
            <Sparkline data={hrTrend} color="#6C5CE7" height={90} />
          </div>

          <div className="bg-white rounded-card border border-border p-5">
            <div className="text-sm font-medium mb-3">Screening Status Distribution</div>
            <div className="flex items-center gap-5">
              <DonutChart
                segments={[
                  { value: dist.normal, color: "#1D9E75" },
                  { value: dist.attention, color: "#378ADD" },
                  { value: dist.high, color: "#E24B4A" },
                ]}
                size={110}
                thickness={16}
              />
              <div className="text-xs space-y-1.5">
                <div className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-success-dot" /> Normal {Math.round((dist.normal / (dist.total || 1)) * 100)}% ({dist.normal})
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-attention-dot" /> Attention {Math.round((dist.attention / (dist.total || 1)) * 100)}% ({dist.attention})
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-danger-dot" /> High priority {Math.round((dist.high / (dist.total || 1)) * 100)}% ({dist.high})
                </div>
                <div className="pt-1 text-ink-muted">Total: {dist.total}</div>
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-[1.3fr_1fr] gap-4">
          <div className="bg-white rounded-card border border-border p-5">
            <div className="flex items-center justify-between mb-3">
              <div className="text-sm font-medium">Recent Screenings</div>
              <Link to="/patients" className="text-xs text-accent">View all</Link>
            </div>
            <table className="w-full text-xs">
              <thead>
                <tr className="text-ink-muted text-left">
                  <th className="font-normal pb-2">Patient ID</th>
                  <th className="font-normal pb-2">Name</th>
                  <th className="font-normal pb-2">Age</th>
                  <th className="font-normal pb-2">Screened On</th>
                  <th className="font-normal pb-2">Status</th>
                  <th className="font-normal pb-2"></th>
                </tr>
              </thead>
              <tbody>
                {recent.slice(0, 5).map((p) => (
                  <tr key={p.id} className="border-t border-border">
                    <td className="py-2.5">{p.id}</td>
                    <td className="py-2.5 flex items-center gap-2">
                      <span className="w-6 h-6 rounded-full bg-accent-soft text-accent flex items-center justify-center text-[10px] font-medium">
                        {p.name.split(" ").map((n) => n[0]).join("")}
                      </span>
                      {p.name}
                    </td>
                    <td className="py-2.5">{p.age}</td>
                    <td className="py-2.5">{p.lastScreening}</td>
                    <td className="py-2.5">
                      <StatusBadge status={p.status as Status} />
                    </td>
                    <td className="py-2.5">
                      <Link to={`/patients/${p.id}`}>
                        <Eye size={14} className="text-ink-muted" />
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="bg-white rounded-card border border-border p-5">
            <div className="flex items-center justify-between mb-3">
              <div className="text-sm font-medium">Top Risk Indicators (Today)</div>
              <span className="text-xs text-accent">More</span>
            </div>
            <div className="space-y-3">
              {riskIndicators.map((r: any) => (
                <div key={r.label}>
                  <div className="flex items-center justify-between text-xs mb-1">
                    <span className="flex items-center gap-2">
                      <span
                        className="w-6 h-6 rounded-control flex items-center justify-center"
                        style={{ backgroundColor: `${r.color}1A`, color: r.color }}
                      >
                        ●
                      </span>
                      {r.label}
                    </span>
                    <span className="text-ink-muted">{r.cases} Cases</span>
                  </div>
                  <div className="h-1.5 rounded-full bg-wash overflow-hidden">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${Math.min(r.cases * 4, 100)}%`,
                        backgroundColor: r.color,
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
