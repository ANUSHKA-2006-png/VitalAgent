import { useState, useEffect, ReactNode } from "react";
import { useParams } from "react-router-dom";
import { HeartPulse, Activity, Droplets, Footprints } from "lucide-react";
import Header from "../components/Header";
import StatusBadge from "../components/StatusBadge";
import { fetchPatient, fetchPatientScreenings } from "../api/client";
import { Patient, Status } from "../types";

const TABS = ["Overview", "History", "Reports", "Notes"];

export default function PatientProfile() {
  const { id } = useParams();
  const [patient, setPatient] = useState<Patient | null>(null);
  const [screenings, setScreenings] = useState<any[]>([]);
  const [tab, setTab] = useState("Overview");

  useEffect(() => {
    if (id) {
      fetchPatient(id).then(setPatient);
      fetchPatientScreenings(id).then(setScreenings);
    }
  }, [id]);

  if (!patient) {
    return (
      <>
        <Header title="Patient Profile" backTo="/patients" />
        <div className="flex-1 p-6 text-sm text-ink-muted">Loading patient profile...</div>
      </>
    );
  }

  const pStatus: Status = patient.status || "normal";

  return (
    <>
      <Header title="Patient Profile" backTo="/patients" />

      <div className="flex-1 overflow-y-auto p-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-11 h-11 rounded-full bg-accent-soft text-accent flex items-center justify-center font-medium">
            {patient.name.split(" ").map((n) => n[0]).join("")}
          </div>
          <div>
            <div className="text-base font-medium flex items-center gap-2">
              {patient.name}
              <StatusBadge status={pStatus} />
            </div>
            <div className="text-xs text-ink-muted">
              {patient.id} · {patient.age} Years · {patient.gender}
            </div>
          </div>
        </div>

        <div className="flex gap-5 border-b border-border mb-5 text-sm text-ink-muted">
          {TABS.map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`pb-2.5 ${
                tab === t
                  ? "text-accent font-medium border-b-2 border-accent"
                  : ""
              }`}
            >
              {t}
            </button>
          ))}
        </div>

        {tab === "Overview" && (
          <div className="grid grid-cols-1 lg:grid-cols-[1.5fr_1fr] gap-5">
            <div className="space-y-4">
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <MiniMetric
                  icon={<HeartPulse size={16} />}
                  label="Heart Rate"
                  value={`${patient.heartRate} BPM`}
                  status={patient.heartRate > 90 ? "high" : patient.heartRate > 80 ? "attention" : "normal"}
                />
                <MiniMetric
                  icon={<Activity size={16} />}
                  label="Stress Level"
                  value={patient.stressLevel}
                  status={patient.stressLevel === "High" ? "high" : patient.stressLevel === "Moderate" ? "attention" : "normal"}
                />
                <MiniMetric
                  icon={<Droplets size={16} />}
                  label="SpO2"
                  value={`${patient.spo2}%`}
                  status={patient.spo2 < 92 ? "high" : patient.spo2 < 95 ? "attention" : "normal"}
                />
                <MiniMetric
                  icon={<Footprints size={16} />}
                  label="Fall Risk"
                  value={patient.fallRisk}
                  status={patient.fallRisk === "High Risk" ? "high" : patient.fallRisk === "Moderate Risk" ? "attention" : "normal"}
                />
              </div>

              <div className="bg-white rounded-card border border-border p-5">
                <div className="text-sm font-medium mb-3">Patient Information</div>
                <table className="w-full text-xs">
                  <tbody>
                    {[
                      ["Age", `${patient.age} Years`],
                      ["Gender", patient.gender],
                      ["Height", `${patient.height} cm`],
                      ["Weight", `${patient.weight} kg`],
                      ["Patient ID", patient.id],
                      ["Phone", patient.phone],
                    ].map(([k, v]) => (
                      <tr key={k} className="border-t border-border first:border-t-0">
                        <td className="py-2 text-ink-muted">{k}</td>
                        <td className="py-2 text-right">{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="bg-wash rounded-card p-5">
              <div className="text-sm font-medium mb-3">
                Latest Screening ({patient.lastScreening})
              </div>
              <div className="space-y-2">
                <LatestRow icon={<HeartPulse size={16} />} label="Heart Rate" value={`${patient.heartRate} BPM`} />
                <LatestRow icon={<Activity size={16} />} label="Stress Level" value={patient.stressLevel} />
                <LatestRow icon={<Droplets size={16} />} label="SpO2" value={`${patient.spo2}%`} />
                <LatestRow icon={<Footprints size={16} />} label="Fall Risk" value={patient.fallRisk} />
              </div>
              <button onClick={() => setTab("History")} className="text-right text-xs text-accent mt-3 block w-full">
                View Full History →
              </button>
            </div>
          </div>
        )}

        {tab === "History" && (
          <div className="bg-white rounded-card border border-border p-5">
            <div className="text-sm font-medium mb-3">Screening History</div>
            {screenings.length > 0 ? (
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-ink-muted text-left border-b border-border">
                    <th className="pb-2">Screening ID</th>
                    <th className="pb-2">Timestamp</th>
                    <th className="pb-2">Heart Rate</th>
                    <th className="pb-2">SpO2</th>
                    <th className="pb-2">Stress</th>
                    <th className="pb-2">Fall Status</th>
                    <th className="pb-2">Overall Status</th>
                  </tr>
                </thead>
                <tbody>
                  {screenings.map((s) => (
                    <tr key={s.id} className="border-t border-border">
                      <td className="py-2.5 font-medium">{s.id}</td>
                      <td className="py-2.5">{s.timestamp}</td>
                      <td className="py-2.5">{s.hr_bpm} BPM</td>
                      <td className="py-2.5">{s.spo2_pct}%</td>
                      <td className="py-2.5">{s.stress_class}</td>
                      <td className="py-2.5">{s.fall_detected}</td>
                      <td className="py-2.5">
                        <StatusBadge status={s.status as Status} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div className="text-xs text-ink-muted py-4">No past screening records logged for this patient yet.</div>
            )}
          </div>
        )}

        {tab !== "Overview" && tab !== "History" && (
          <div className="text-sm text-ink-muted bg-white rounded-card border border-border p-8 text-center">
            {tab} content for {patient.name} will appear here.
          </div>
        )}
      </div>
    </>
  );
}

function MiniMetric({
  icon,
  label,
  value,
  status,
}: {
  icon: ReactNode;
  label: string;
  value: string;
  status: "normal" | "attention" | "high";
}) {
  return (
    <div className="bg-white rounded-card border border-border p-3">
      <div className="flex items-center gap-1.5 text-[11px] text-ink-muted mb-1">
        {icon} {label}
      </div>
      <div className="text-sm font-semibold mb-1">{value}</div>
      <StatusBadge status={status} />
    </div>
  );
}

function LatestRow({
  icon,
  label,
  value,
}: {
  icon: ReactNode;
  label: string;
  value: string;
}) {
  return (
    <div className="bg-white rounded-control px-3 py-2.5 flex items-center justify-between text-sm">
      <span className="flex items-center gap-2 text-ink-muted">
        {icon} {label}
      </span>
      <span className="font-medium">{value}</span>
    </div>
  );
}
