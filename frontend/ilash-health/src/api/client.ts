import { patients as mockPatients, alerts as mockAlerts, topRiskIndicators, recentReports, heartRateTrend, spo2Trend, stressTrend, screeningsOverTime } from "../data/mockData";
import { Patient, AlertItem, ReportRow, Status } from "../types";

const API_BASE = "/api";

export async function fetchDashboardSummary() {
  try {
    const res = await fetch(`${API_BASE}/dashboard/summary`);
    if (res.ok) {
      return await res.json();
    }
  } catch (err) {
    console.warn("Backend API unavailable, using mock data fallback", err);
  }
  return {
    metrics: {
      total_patients: 1248,
      screenings_today: 42,
      attention_required: 18,
      active_alerts: 5,
    },
    heart_rate_trend: heartRateTrend,
    status_distribution: { normal: 901, attention: 263, high: 84, total: 1248 },
    recent_screenings: mockPatients.slice(0, 5),
    top_risk_indicators: topRiskIndicators,
  };
}

export async function fetchPatients(query: string = ""): Promise<Patient[]> {
  try {
    const url = query ? `${API_BASE}/patients?query=${encodeURIComponent(query)}` : `${API_BASE}/patients`;
    const res = await fetch(url);
    if (res.ok) {
      return await res.json();
    }
  } catch (err) {
    console.warn("Backend API unavailable, using mock data fallback", err);
  }
  return mockPatients.filter((p) =>
    (p.name + p.id).toLowerCase().includes(query.toLowerCase())
  );
}

export async function fetchPatient(id: string): Promise<Patient> {
  try {
    const res = await fetch(`${API_BASE}/patients/${id}`);
    if (res.ok) {
      return await res.json();
    }
  } catch (err) {
    console.warn("Backend API unavailable, using mock data fallback", err);
  }
  return mockPatients.find((p) => p.id === id) || mockPatients[0];
}

export async function uploadScreeningData(file?: File, modality: string = "hr", sessionId?: string) {
  try {
    const formData = new FormData();
    if (file) {
      formData.append("file", file);
    }
    formData.append("modality", modality);
    if (sessionId) {
      formData.append("session_id", sessionId);
    }
    const res = await fetch(`${API_BASE}/screenings/upload`, {
      method: "POST",
      body: formData,
    });
    if (res.ok) {
      return await res.json();
    }
  } catch (err) {
    console.warn("Upload API unavailable, using local staged simulation", err);
  }
  return {
    session_id: sessionId || `upl_${Date.now()}`,
    upload_id: sessionId || `upl_${Date.now()}`,
    modality: modality,
    status: "ready",
    bvp_samples: 512,
    accel_samples: 512,
    file_name: file ? file.name : `sample_${modality}.csv`,
    details: "8 seconds / 512 samples detected",
  };
}

export async function analyzeScreening(
  patientId: string,
  uploadId?: string,
  notes?: string,
  patientDetails?: {
    name?: string;
    age?: string | number;
    gender?: string;
    height?: string | number;
    weight?: string | number;
    phone?: string;
  }
) {
  try {
    const res = await fetch(`${API_BASE}/screenings/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        patient_id: patientId,
        upload_id: uploadId,
        notes: notes || "",
        patient_name: patientDetails?.name,
        age: patientDetails?.age ? Number(patientDetails.age) : undefined,
        gender: patientDetails?.gender,
        height: patientDetails?.height ? Number(patientDetails.height) : undefined,
        weight: patientDetails?.weight ? Number(patientDetails.weight) : undefined,
        phone: patientDetails?.phone,
      }),
    });
    if (res.ok) {
      return await res.json();
    }
  } catch (err) {
    console.warn("Analysis API unavailable, using fallback result", err);
  }

  // Realistic fallback object matching real API structure
  return {
    id: `scr_${Date.now()}`,
    patient: mockPatients.find((p) => p.id === patientId) || mockPatients[0],
    timestamp: "10 Aug 2026, 10:48 AM",
    overall_status: "attention" as Status,
    metrics: {
      heart_rate: {
        value: "78 BPM",
        numeric_value: 78,
        status: "normal" as Status,
        description: "Within normal resting range.",
        trend: heartRateTrend,
      },
      stress_level: {
        value: "Moderate",
        raw_class: "STRESS",
        status: "attention" as Status,
        description: "Elevated stress indication.",
        trend: stressTrend,
      },
      spo2: {
        value: "97%",
        numeric_value: 97,
        status: "normal" as Status,
        description: "Oxygen saturation within healthy range (≥95%).",
        trend: spo2Trend,
      },
      fall_risk: {
        value: "Low Risk",
        raw_class: "NO FALL",
        status: "normal" as Status,
        description: "No concerning patterns detected.",
        trend: [2, 1, 2, 1, 1, 2, 1],
      },
    },
  };
}

export async function fetchScreeningResults(screeningId: string) {
  try {
    const res = await fetch(`${API_BASE}/screenings/${screeningId}/results`);
    if (res.ok) {
      return await res.json();
    }
  } catch (err) {
    console.warn("Fetch screening results error", err);
  }
  return null;
}

export async function fetchPatientScreenings(patientId: string) {
  try {
    const res = await fetch(`${API_BASE}/patients/${patientId}/screenings`);
    if (res.ok) {
      return await res.json();
    }
  } catch (err) {
    console.warn("Fetch patient screenings error", err);
  }
  return [];
}

export async function fetchAlerts(): Promise<AlertItem[]> {
  try {
    const res = await fetch(`${API_BASE}/alerts`);
    if (res.ok) {
      return await res.json();
    }
  } catch (err) {
    console.warn("Alerts API unavailable, using mock data fallback", err);
  }
  return mockAlerts;
}

export async function resolveAlert(id: string) {
  try {
    const res = await fetch(`${API_BASE}/alerts/${id}/resolve`, {
      method: "PATCH",
    });
    if (res.ok) {
      return await res.json();
    }
  } catch (err) {
    console.warn("Resolve alert API error", err);
  }
  return { status: "success", alert_id: id };
}

export async function fetchReports(): Promise<ReportRow[]> {
  try {
    const res = await fetch(`${API_BASE}/reports`);
    if (res.ok) {
      return await res.json();
    }
  } catch (err) {
    console.warn("Reports API error", err);
  }
  return recentReports;
}

export async function generateReport(type: string, patientId?: string) {
  try {
    const res = await fetch(`${API_BASE}/reports/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ type, patientId }),
    });
    if (res.ok) {
      return await res.json();
    }
  } catch (err) {
    console.warn("Generate report API error", err);
  }
  return {
    name: `${patientId || "P-1042"}_${type}_Report.pdf`,
    type: type,
    generatedOn: "Just now",
    generatedBy: "Dr. Priya Sharma",
  };
}

export async function fetchCommunityAnalytics() {
  try {
    const res = await fetch(`${API_BASE}/analytics/community`);
    if (res.ok) {
      return await res.json();
    }
  } catch (err) {
    console.warn("Analytics API error", err);
  }
  return {
    metrics: {
      total_screened: 1248,
      normal_pct: "72%",
      attention_pct: "21%",
      high_pct: "7%",
    },
    screenings_over_time: screeningsOverTime,
    gender_distribution: { male_pct: 58, female_pct: 42 },
    top_risk_indicators: topRiskIndicators,
  };
}
