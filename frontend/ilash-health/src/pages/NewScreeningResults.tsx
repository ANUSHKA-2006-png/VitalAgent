import { ReactNode, useState, useEffect } from "react";
import { useNavigate, Link } from "react-router-dom";
import { Download, HeartPulse, Activity, Droplets, Footprints, AlertTriangle } from "lucide-react";
import Header from "../components/Header";
import Stepper from "../components/Stepper";
import StatusBadge from "../components/StatusBadge";
import Sparkline from "../components/Sparkline";
import { generateReport } from "../api/client";
import { Status } from "../types";

export default function NewScreeningResults() {
  const navigate = useNavigate();
  const [resultsData, setResultsData] = useState<any>(null);

  useEffect(() => {
    const cached = sessionStorage.getItem("screening_results");
    if (cached) {
      try {
        setResultsData(JSON.parse(cached));
      } catch (e) {
        console.error("Error parsing screening_results", e);
      }
    }
  }, []);

  const patientId = resultsData?.patient?.id || "P-1042";
  const timestamp = resultsData?.timestamp || "10 Aug 2026, 10:48 AM";
  const metrics = resultsData?.metrics;

  const hrValue = metrics?.heart_rate?.value || "78 BPM";
  const hrStatus: Status = metrics?.heart_rate?.status || "normal";
  const hrTrend = metrics?.heart_rate?.trend || [70, 74, 72, 78, 75, 80, 78];

  const stressValue = metrics?.stress_level?.value || "Moderate";
  const stressStatus: Status = metrics?.stress_level?.status || "attention";
  const stressTrend = metrics?.stress_level?.trend || [40, 55, 48, 60, 52, 58, 54];

  const spo2Value = metrics?.spo2?.value || "97%";
  const spo2Status: Status = metrics?.spo2?.status || "normal";
  const spo2Trend = metrics?.spo2?.trend || [98, 97, 96, 95, 97, 96, 97];

  const fallValue = metrics?.fall_risk?.value || "Low Risk";
  const fallStatus: Status = metrics?.fall_risk?.status || "normal";
  const fallTrend = metrics?.fall_risk?.trend || [2, 1, 2, 1, 1, 2, 1];

  const overallStatus: Status = resultsData?.overall_status || "attention";

  const handleDownload = async () => {
    await generateReport("Individual", patientId);
    alert(`Report generated for patient ${patientId}! Check the Reports page.`);
  };

  return (
    <>
      <Header title="New Screening" backTo="/screening/new/analysis" />
      <Stepper current={4} />
      <div className="flex-1 overflow-y-auto p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="text-xs text-ink-muted">
            Patient ID: <span className="text-ink">{patientId}</span> · Screened on {timestamp}
          </div>
          <button
            onClick={handleDownload}
            className="flex items-center gap-1.5 border border-border rounded-control px-3 py-2 text-sm bg-white hover:bg-wash"
          >
            <Download size={14} /> Download Report
          </button>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
          <ResultCard
            icon={<HeartPulse size={16} />}
            label="Heart Rate"
            value={hrValue}
            status={hrStatus}
            description="MOMENT Model + PPG-DaLiA Predictor."
            trendData={hrTrend}
            trendColor={hrStatus === "high" ? "#E24B4A" : hrStatus === "attention" ? "#378ADD" : "#1D9E75"}
          />
          <ResultCard
            icon={<Activity size={16} />}
            label="Stress Level"
            value={stressValue}
            status={stressStatus}
            description="MOMENT Model + WESAD Finetuned Head."
            trendData={stressTrend}
            trendColor={stressStatus === "high" ? "#E24B4A" : stressStatus === "attention" ? "#378ADD" : "#1D9E75"}
          />
          <ResultCard
            icon={<Droplets size={16} />}
            label="SpO2 (Oxygen Saturation)"
            value={spo2Value}
            status={spo2Status}
            description="MOMENT Model + BIDMC Regressor."
            trendData={spo2Trend}
            trendColor={spo2Status === "high" ? "#E24B4A" : spo2Status === "attention" ? "#378ADD" : "#1D9E75"}
          />
          <ResultCard
            icon={<Footprints size={16} />}
            label="Fall Risk"
            value={fallValue}
            status={fallStatus}
            description="MOMENT Model + UP-Fall Classifier."
            trendData={fallTrend}
            trendColor={fallStatus === "high" ? "#E24B4A" : fallStatus === "attention" ? "#378ADD" : "#1D9E75"}
          />
        </div>

        <div
          className={`${
            overallStatus === "high"
              ? "bg-danger-bg border-danger-text"
              : overallStatus === "attention"
              ? "bg-[#EBF3FB]"
              : "bg-[#EBF7F3]"
          } rounded-card p-5 flex flex-col sm:flex-row sm:items-center justify-between gap-3`}
        >
          <div className="flex items-start gap-2">
            <AlertTriangle
              size={18}
              className={
                overallStatus === "high"
                  ? "text-danger-text mt-0.5"
                  : overallStatus === "attention"
                  ? "text-attention-text mt-0.5"
                  : "text-success-text mt-0.5"
              }
            />
            <div>
              <div
                className={`text-sm font-medium ${
                  overallStatus === "high"
                    ? "text-danger-text"
                    : overallStatus === "attention"
                    ? "text-[#1E5D8C]"
                    : "text-success-text"
                }`}
              >
                Overall Status: {overallStatus === "high" ? "High Priority Risk Detected" : overallStatus === "attention" ? "Attention Required" : "Normal / Healthy"}
              </div>
              <div className="text-xs text-ink-muted">
                {overallStatus !== "normal"
                  ? "One or more physiological parameters need attention. Please review details."
                  : "All parameters are within healthy normal clinical thresholds."}
              </div>
            </div>
          </div>
          <Link
            to="/alerts"
            className={`${
              overallStatus === "high"
                ? "bg-danger-text"
                : overallStatus === "attention"
                ? "bg-[#1E5D8C]"
                : "bg-accent"
            } text-white text-sm font-medium rounded-control px-4 py-2 flex-shrink-0 text-center`}
          >
            Review Alerts
          </Link>
        </div>
      </div>
    </>
  );
}

function ResultCard({
  icon,
  label,
  value,
  status,
  description,
  trendData,
  trendColor,
}: {
  icon: ReactNode;
  label: string;
  value: string;
  status: "normal" | "attention" | "high";
  description: string;
  trendData: number[];
  trendColor: string;
}) {
  return (
    <div className="bg-white rounded-card border border-border p-4">
      <div className="flex items-center gap-1.5 text-xs text-ink-muted mb-2">
        {icon} {label}
      </div>
      <div className="text-xl font-semibold mb-1.5">{value}</div>
      <StatusBadge status={status} />
      <div className="text-[11px] text-ink-muted mt-2 mb-2">{description}</div>
      <Sparkline data={trendData} color={trendColor} height={36} />
    </div>
  );
}
