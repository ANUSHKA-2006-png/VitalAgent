import { ReactNode, useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { UploadCloud, Check, HeartPulse, Activity, Droplets, Footprints } from "lucide-react";
import Header from "../components/Header";
import Stepper from "../components/Stepper";
import { uploadScreeningData } from "../api/client";

interface UploadCardData {
  step: number;
  modalityKey: string;
  title: string;
  subtitle: string;
  icon: ReactNode;
  fileLabel: string;
  detail: string;
}

const CARDS: UploadCardData[] = [
  {
    step: 1,
    modalityKey: "hr",
    title: "Heart Rate Screening",
    subtitle: "PPG + Accelerometer",
    icon: <HeartPulse size={20} />,
    fileLabel: "PPG file",
    detail: "8 seconds / 512 samples detected",
  },
  {
    step: 2,
    modalityKey: "stress",
    title: "Stress Screening",
    subtitle: "WESAD Signals",
    icon: <Activity size={20} />,
    fileLabel: "WESAD file",
    detail: "8 seconds / 512 samples detected",
  },
  {
    step: 3,
    modalityKey: "spo2",
    title: "SpO2 Screening",
    subtitle: "Pulse Oximeter Signal",
    icon: <Droplets size={20} />,
    fileLabel: "SpO2 file",
    detail: "8 seconds / 512 samples detected",
  },
  {
    step: 4,
    modalityKey: "fall",
    title: "Fall Screening",
    subtitle: "Wrist Motion",
    icon: <Footprints size={20} />,
    fileLabel: "Fall file",
    detail: "512 samples detected",
  },
];

export default function NewScreeningUpload() {
  const navigate = useNavigate();
  const fileInputsRef = useRef<Record<string, HTMLInputElement | null>>({});
  const [uploadedFiles, setUploadedFiles] = useState<Record<string, string>>({});

  const handleFileChange = async (modality: string, e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setUploadedFiles((prev) => ({ ...prev, [modality]: file.name }));
      const existingSessionId = sessionStorage.getItem("screening_upload_id") || undefined;
      const res = await uploadScreeningData(file, modality, existingSessionId);
      if (res && (res.session_id || res.upload_id)) {
        sessionStorage.setItem("screening_upload_id", res.session_id || res.upload_id);
      }
    }
  };

  const handleStartAnalysis = async () => {
    if (!sessionStorage.getItem("screening_upload_id")) {
      const res = await uploadScreeningData(undefined, "hr");
      if (res && (res.session_id || res.upload_id)) {
        sessionStorage.setItem("screening_upload_id", res.session_id || res.upload_id);
      }
    }
    navigate("/screening/new/analysis");
  };

  return (
    <>
      <Header title="New Screening" backTo="/screening/new/details" />
      <Stepper current={2} />
      <div className="flex-1 overflow-y-auto p-6">
        <div className="mb-4">
          <div className="text-sm font-medium">Upload Sensor Data</div>
          <div className="text-xs text-ink-muted">
            Upload individual signal files for each screening modality to run targeted ML feature extraction.
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {CARDS.map((c) => (
            <div
              key={c.title}
              className="bg-white rounded-card border border-border p-4 text-center flex flex-col"
            >
              <input
                type="file"
                ref={(el) => (fileInputsRef.current[c.modalityKey] = el)}
                onChange={(e) => handleFileChange(c.modalityKey, e)}
                className="hidden"
                accept=".npy,.csv,.txt,.json"
              />

              <div className="w-7 h-7 rounded-full bg-accent-soft text-accent flex items-center justify-center text-xs font-medium mx-auto mb-2">
                {c.step}
              </div>
              <div className="flex items-center justify-center gap-1.5 text-sm font-medium mb-0.5">
                {c.icon} {c.title}
              </div>
              <div className="text-[11px] text-ink-muted mb-3">{c.subtitle}</div>

              <div
                onClick={() => fileInputsRef.current[c.modalityKey]?.click()}
                className="border border-dashed border-border rounded-control py-6 px-2 text-ink-muted flex-1 flex flex-col items-center justify-center gap-1 cursor-pointer hover:border-accent transition-colors"
              >
                <UploadCloud size={22} className={uploadedFiles[c.modalityKey] ? "text-accent" : ""} />
                <div className="text-xs font-medium truncate max-w-[180px]">
                  {uploadedFiles[c.modalityKey] ? (
                    <span className="text-accent">{uploadedFiles[c.modalityKey]}</span>
                  ) : (
                    `Drag & drop ${c.fileLabel}`
                  )}
                </div>
                <div className="text-[11px] my-1">or</div>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    fileInputsRef.current[c.modalityKey]?.click();
                  }}
                  className="border border-border rounded-control px-3 py-1.5 text-xs bg-white hover:bg-wash"
                >
                  Choose File
                </button>
              </div>

              <div className="flex items-center justify-center gap-1.5 text-[11px] text-success-text mt-2">
                <Check size={12} /> {c.detail}
              </div>
            </div>
          ))}
        </div>

        <div className="text-[11px] text-ink-muted mt-3">
          Each screening requires 8 seconds (512 samples) of sensor data. You can select specific CSV files from <code className="bg-wash px-1 rounded text-ink">sample_csv_inputs/</code> for each modality.
        </div>

        <div className="flex justify-between mt-5">
          <button
            onClick={() => navigate("/screening/new/details")}
            className="border border-border rounded-control px-5 py-2.5 text-sm bg-white"
          >
            ← Back
          </button>
          <button
            onClick={handleStartAnalysis}
            className="bg-gradient-to-r from-accent-light to-accent text-white text-sm font-medium rounded-control px-5 py-2.5"
          >
            Next: Start Analysis →
          </button>
        </div>
      </div>
    </>
  );
}
