import { useState, useEffect } from "react";
import Header from "../components/Header";
import MetricCard from "../components/MetricCard";
import Sparkline from "../components/Sparkline";
import DonutChart from "../components/DonutChart";
import { fetchCommunityAnalytics } from "../api/client";

export default function CommunityAnalytics() {
  const [data, setData] = useState<any>(null);

  useEffect(() => {
    fetchCommunityAnalytics().then(setData);
  }, []);

  const metrics = data?.metrics || {
    total_screened: 1248,
    normal_pct: "72%",
    attention_pct: "21%",
    high_pct: "7%",
  };

  const screeningsTrend = data?.screenings_over_time || [30, 45, 40, 55, 48, 60, 58, 65, 62, 70];
  const genderDist = data?.gender_distribution || { male_pct: 58, female_pct: 42 };
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
        title="Community Analytics"
        subtitle="Overview of community health screening insights."
        actions={
          <div className="text-xs text-ink-muted border border-border rounded-control px-3 py-2">
            Last 30 Days
          </div>
        }
      />

      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard label="Total Screened" value={metrics.total_screened.toLocaleString()} trend="12%" />
          <MetricCard label="Normal" value={`${metrics.normal_pct} (901)`} trend="5%" />
          <MetricCard label="Attention" value={`${metrics.attention_pct} (263)`} trend="3%" trendUp={false} />
          <MetricCard label="High Priority" value={`${metrics.high_pct} (84)`} trend="2%" trendUp={false} />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-[1.4fr_1fr] gap-4">
          <div className="bg-white rounded-card border border-border p-5">
            <div className="text-sm font-medium mb-3">Screenings Over Time</div>
            <Sparkline data={screeningsTrend} color="#6C5CE7" height={100} />
          </div>
          <div className="bg-white rounded-card border border-border p-5">
            <div className="text-sm font-medium mb-3">Gender Distribution</div>
            <div className="flex items-center gap-4">
              <DonutChart
                segments={[
                  { value: genderDist.male_pct, color: "#378ADD" },
                  { value: genderDist.female_pct, color: "#D4537E" },
                ]}
                size={80}
                thickness={12}
              />
              <div className="text-xs space-y-1.5">
                <div className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-[#378ADD]" /> Male {genderDist.male_pct}%
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-[#D4537E]" /> Female {genderDist.female_pct}%
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div className="bg-white rounded-card border border-border p-5">
            <div className="text-sm font-medium mb-3">Top Risk Indicators</div>
            <div className="space-y-2 text-sm">
              {riskIndicators.map((r: any) => (
                <div key={r.label} className="flex justify-between border-t border-border pt-2 first:border-t-0 first:pt-0">
                  <span className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full" style={{ backgroundColor: r.color }} />
                    {r.label}
                  </span>
                  <span className="text-ink-muted">{r.cases} Cases</span>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-white rounded-card border border-border p-5">
            <div className="text-sm font-medium mb-3">Age Distribution</div>
            <div className="flex items-end gap-3 h-20">
              {[30, 55, 80, 60, 40].map((h, i) => (
                <div
                  key={i}
                  className="flex-1 bg-accent-light rounded-t"
                  style={{ height: `${h}%` }}
                />
              ))}
            </div>
            <div className="flex justify-between text-[10px] text-ink-muted mt-1">
              <span>10-30</span><span>31-40</span><span>41-50</span><span>51-60</span><span>60+</span>
            </div>
          </div>

          <div className="bg-white rounded-card border border-border p-5">
            <div className="text-sm font-medium mb-3">SpO2 Distribution</div>
            <DonutChart
              segments={[
                { value: 78, color: "#1D9E75" },
                { value: 14, color: "#378ADD" },
                { value: 8, color: "#E24B4A" },
              ]}
              size={80}
              thickness={12}
            />
            <div className="text-xs space-y-1 mt-3">
              <div className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-success-dot" /> ≥95% Normal 78%
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-attention-dot" /> 92-94% Attention 14%
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-danger-dot" /> &lt;92% Low 8%
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
