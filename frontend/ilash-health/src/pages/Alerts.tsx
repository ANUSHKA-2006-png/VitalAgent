import { useState, useEffect, ReactNode } from "react";
import { AlertTriangle, Activity, HeartPulse, Droplets, CheckCircle2 } from "lucide-react";
import Header from "../components/Header";
import StatusBadge from "../components/StatusBadge";
import { fetchAlerts, resolveAlert, fetchPatients } from "../api/client";
import { AlertItem, Patient, Status } from "../types";

const TABS = ["All Alerts", "High Priority", "Attention", "Resolved"];

const ICONS: Record<string, ReactNode> = {
  "Possible Fall Event": <AlertTriangle size={18} className="text-danger-text" />,
  "Low Oxygen Saturation": <Droplets size={18} className="text-danger-text" />,
  "Elevated Stress Indication": <Activity size={18} className="text-attention-text" />,
  "High Heart Rate Detected": <HeartPulse size={18} className="text-attention-text" />,
  "Screening Completed": <CheckCircle2 size={18} className="text-success-text" />,
};

export default function Alerts() {
  const [tab, setTab] = useState("All Alerts");
  const [alertList, setAlertList] = useState<AlertItem[]>([]);
  const [patientMap, setPatientMap] = useState<Record<string, Patient>>({});

  useEffect(() => {
    fetchAlerts().then(setAlertList);
    fetchPatients().then((list) => {
      const map: Record<string, Patient> = {};
      list.forEach((p) => (map[p.id] = p));
      setPatientMap(map);
    });
  }, []);

  const handleResolve = async (id: string) => {
    await resolveAlert(id);
    setAlertList((prev) =>
      prev.map((a) => (a.id === id ? { ...a, resolved: true } : a))
    );
  };

  const handleMarkAllRead = async () => {
    for (const a of alertList) {
      if (!a.resolved) await resolveAlert(a.id);
    }
    setAlertList((prev) => prev.map((a) => ({ ...a, resolved: true })));
  };

  const filtered = alertList.filter((a) => {
    if (tab === "All Alerts") return true;
    if (tab === "High Priority") return a.severity === "high";
    if (tab === "Attention") return a.severity === "attention";
    if (tab === "Resolved") return a.resolved;
    return true;
  });

  return (
    <>
      <Header
        title="Alerts"
        subtitle="View and manage all alerts."
        actions={
          <button onClick={handleMarkAllRead} className="text-xs text-accent">
            Mark all as read
          </button>
        }
      />

      <div className="flex-1 overflow-y-auto p-6">
        <div className="flex gap-5 border-b border-border mb-4 text-sm text-ink-muted">
          {TABS.map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`pb-2.5 ${
                tab === t ? "text-accent font-medium border-b-2 border-accent" : ""
              }`}
            >
              {t}
            </button>
          ))}
        </div>

        <div className="space-y-2.5">
          {filtered.map((a) => {
            const patient = patientMap[a.patientId];
            return (
              <div
                key={a.id}
                className="bg-white rounded-card border border-border px-4 py-3 flex items-center justify-between gap-4"
              >
                <div className="flex items-start gap-3 min-w-0">
                  {ICONS[a.title] ?? <AlertTriangle size={18} />}
                  <div className="min-w-0">
                    <div className="text-sm font-medium flex items-center gap-2 flex-wrap">
                      {a.title}
                      <StatusBadge status={a.severity as Status} />
                    </div>
                    <div className="text-xs text-ink-muted">
                      Patient: {a.patientId} {patient ? `(${patient.name})` : ""} · {a.description}
                    </div>
                  </div>
                </div>
                <div className="text-right flex-shrink-0">
                  <div className="text-xs text-ink-muted">{a.timestamp}</div>
                  {!a.resolved ? (
                    <button onClick={() => handleResolve(a.id)} className="text-xs text-accent hover:underline">
                      Mark Resolved
                    </button>
                  ) : (
                    <span className="text-xs text-success-text flex items-center gap-1 justify-end">
                      Resolved
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </>
  );
}
