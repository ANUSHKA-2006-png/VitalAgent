export type Status = "normal" | "attention" | "high";

export interface Patient {
  id: string;
  name: string;
  age: number;
  gender: "Male" | "Female";
  lastScreening: string;
  status: Status;
  heartRate: number;
  stressLevel: "Low" | "Moderate" | "High";
  fallRisk: "Low Risk" | "Moderate Risk" | "High Risk";
  spo2: number;
  height: number;
  weight: number;
  phone: string;
}

export interface AlertItem {
  id: string;
  title: string;
  description: string;
  severity: Status;
  timestamp: string;
  patientId: string;
  resolved?: boolean;
}

export interface RiskIndicator {
  label: string;
  cases: number;
  color: string;
}

export interface ReportRow {
  name: string;
  type: string;
  generatedOn: string;
  generatedBy: string;
}
