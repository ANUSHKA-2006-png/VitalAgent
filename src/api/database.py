import sqlite3
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "vitalagent.db"

INITIAL_PATIENTS = [
    {
        "id": "P-1042",
        "name": "Ramesh Kumar",
        "age": 54,
        "gender": "Male",
        "lastScreening": "10 Aug 2026",
        "status": "attention",
        "heartRate": 78,
        "stressLevel": "Moderate",
        "fallRisk": "Low Risk",
        "spo2": 97,
        "height": 170,
        "weight": 68,
        "phone": "+91 98765 43210",
    },
    {
        "id": "P-1041",
        "name": "Sunita Devi",
        "age": 43,
        "gender": "Female",
        "lastScreening": "10 Aug 2026",
        "status": "normal",
        "heartRate": 72,
        "stressLevel": "Low",
        "fallRisk": "Low Risk",
        "spo2": 98,
        "height": 158,
        "weight": 61,
        "phone": "+91 98765 11223",
    },
    {
        "id": "P-1040",
        "name": "Mohammed Ali",
        "age": 62,
        "gender": "Male",
        "lastScreening": "10 Aug 2026",
        "status": "attention",
        "heartRate": 88,
        "stressLevel": "Moderate",
        "fallRisk": "Moderate Risk",
        "spo2": 94,
        "height": 165,
        "weight": 74,
        "phone": "+91 98765 33445",
    },
    {
        "id": "P-1039",
        "name": "Lakshmi Nair",
        "age": 48,
        "gender": "Female",
        "lastScreening": "10 Aug 2026",
        "status": "normal",
        "heartRate": 74,
        "stressLevel": "Low",
        "fallRisk": "Low Risk",
        "spo2": 98,
        "height": 160,
        "weight": 58,
        "phone": "+91 98765 55667",
    },
    {
        "id": "P-1038",
        "name": "Suresh Babu",
        "age": 57,
        "gender": "Male",
        "lastScreening": "10 Aug 2026",
        "status": "high",
        "heartRate": 96,
        "stressLevel": "High",
        "fallRisk": "High Risk",
        "spo2": 90,
        "height": 168,
        "weight": 80,
        "phone": "+91 98765 77889",
    },
    {
        "id": "P-1037",
        "name": "Meena Patel",
        "age": 35,
        "gender": "Female",
        "lastScreening": "09 Aug 2026",
        "status": "normal",
        "heartRate": 70,
        "stressLevel": "Low",
        "fallRisk": "Low Risk",
        "spo2": 99,
        "height": 162,
        "weight": 55,
        "phone": "+91 98765 99001",
    },
]

INITIAL_ALERTS = [
    {
        "id": "a1",
        "title": "Possible Fall Event",
        "description": "Movement pattern indicates a possible fall event.",
        "severity": "high",
        "timestamp": "10 minutes ago",
        "patientId": "P-1038",
        "resolved": 0,
    },
    {
        "id": "a2",
        "title": "Low Oxygen Saturation",
        "description": "SpO2 reading dropped below 92% during screening.",
        "severity": "high",
        "timestamp": "18 minutes ago",
        "patientId": "P-1038",
        "resolved": 0,
    },
    {
        "id": "a3",
        "title": "Elevated Stress Indication",
        "description": "Stress level higher than normal range.",
        "severity": "attention",
        "timestamp": "24 minutes ago",
        "patientId": "P-1042",
        "resolved": 0,
    },
    {
        "id": "a4",
        "title": "High Heart Rate Detected",
        "description": "Heart rate above normal range.",
        "severity": "attention",
        "timestamp": "1 hour ago",
        "patientId": "P-1040",
        "resolved": 0,
    },
    {
        "id": "a5",
        "title": "Screening Completed",
        "description": "Screening completed successfully.",
        "severity": "normal",
        "timestamp": "2 hours ago",
        "patientId": "P-1041",
        "resolved": 1,
    },
]

INITIAL_REPORTS = [
    {
        "name": "P-1042_Individual_Report.pdf",
        "type": "Individual",
        "generatedOn": "10 Aug 2026, 10:50 AM",
        "generatedBy": "Dr. Priya Sharma",
    },
    {
        "name": "Community_Summary_Aug2026.pdf",
        "type": "Community",
        "generatedOn": "10 Aug 2026, 09:00 AM",
        "generatedBy": "Dr. Priya Sharma",
    },
    {
        "name": "Alerts_Report_Aug2026.pdf",
        "type": "Alerts",
        "generatedOn": "09 Aug 2026, 08:30 PM",
        "generatedBy": "Dr. Priya Sharma",
    },
]


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        age INTEGER NOT NULL,
        gender TEXT NOT NULL,
        lastScreening TEXT,
        status TEXT NOT NULL,
        heartRate REAL,
        stressLevel TEXT,
        fallRisk TEXT,
        spo2 REAL,
        height REAL,
        weight REAL,
        phone TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS screenings (
        id TEXT PRIMARY KEY,
        patient_id TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        status TEXT NOT NULL,
        hr_bpm REAL NOT NULL,
        spo2_pct REAL NOT NULL,
        stress_class TEXT NOT NULL,
        fall_detected TEXT NOT NULL,
        signal_source TEXT,
        raw_metrics_json TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS alerts (
        id TEXT PRIMARY KEY,
        patientId TEXT NOT NULL,
        title TEXT NOT NULL,
        description TEXT NOT NULL,
        severity TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        resolved INTEGER DEFAULT 0
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        type TEXT NOT NULL,
        generatedOn TEXT NOT NULL,
        generatedBy TEXT NOT NULL
    )
    """)

    conn.commit()

    # Seed patients if empty
    cursor.execute("SELECT COUNT(*) FROM patients")
    if cursor.fetchone()[0] == 0:
        for p in INITIAL_PATIENTS:
            cursor.execute(
                """
                INSERT INTO patients (id, name, age, gender, lastScreening, status, heartRate, stressLevel, fallRisk, spo2, height, weight, phone)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    p["id"],
                    p["name"],
                    p["age"],
                    p["gender"],
                    p["lastScreening"],
                    p["status"],
                    p["heartRate"],
                    p["stressLevel"],
                    p["fallRisk"],
                    p["spo2"],
                    p["height"],
                    p["weight"],
                    p["phone"],
                ),
            )

    # Seed alerts if empty
    cursor.execute("SELECT COUNT(*) FROM alerts")
    if cursor.fetchone()[0] == 0:
        for a in INITIAL_ALERTS:
            cursor.execute(
                """
                INSERT INTO alerts (id, patientId, title, description, severity, timestamp, resolved)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    a["id"],
                    a["patientId"],
                    a["title"],
                    a["description"],
                    a["severity"],
                    a["timestamp"],
                    a["resolved"],
                ),
            )

    # Seed reports if empty
    cursor.execute("SELECT COUNT(*) FROM reports")
    if cursor.fetchone()[0] == 0:
        for r in INITIAL_REPORTS:
            cursor.execute(
                """
                INSERT INTO reports (name, type, generatedOn, generatedBy)
                VALUES (?, ?, ?, ?)
            """,
                (r["name"], r["type"], r["generatedOn"], r["generatedBy"]),
            )

    conn.commit()
    conn.close()


def get_all_patients(query: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    if query:
        q = f"%{query.strip()}%"
        cursor.execute(
            "SELECT * FROM patients WHERE name LIKE ? OR id LIKE ? ORDER BY id DESC",
            (q, q),
        )
    else:
        cursor.execute("SELECT * FROM patients ORDER BY id DESC")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_patient_by_id(patient_id: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients WHERE id = ?", (patient_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def upsert_patient(patient: Dict[str, Any]):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO patients (id, name, age, gender, lastScreening, status, heartRate, stressLevel, fallRisk, spo2, height, weight, phone)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            name=excluded.name,
            age=excluded.age,
            gender=excluded.gender,
            lastScreening=excluded.lastScreening,
            status=excluded.status,
            heartRate=excluded.heartRate,
            stressLevel=excluded.stressLevel,
            fallRisk=excluded.fallRisk,
            spo2=excluded.spo2,
            height=excluded.height,
            weight=excluded.weight,
            phone=excluded.phone
    """,
        (
            patient["id"],
            patient["name"],
            patient["age"],
            patient["gender"],
            patient.get("lastScreening", "Just now"),
            patient.get("status", "normal"),
            patient.get("heartRate", 75),
            patient.get("stressLevel", "Low"),
            patient.get("fallRisk", "Low Risk"),
            patient.get("spo2", 98),
            patient.get("height", 170),
            patient.get("weight", 70),
            patient.get("phone", ""),
        ),
    )
    conn.commit()
    conn.close()


def save_screening_result(screening: Dict[str, Any]):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO screenings (id, patient_id, timestamp, status, hr_bpm, spo2_pct, stress_class, fall_detected, signal_source, raw_metrics_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            screening["id"],
            screening["patient_id"],
            screening["timestamp"],
            screening["status"],
            screening["hr_bpm"],
            screening["spo2_pct"],
            screening["stress_class"],
            screening["fall_detected"],
            screening.get("signal_source", "Wearable Sensor"),
            json.dumps(screening.get("raw_metrics", {})),
        ),
    )

    # Also update patient's latest screening stats
    cursor.execute(
        """
        UPDATE patients SET
            lastScreening=?,
            status=?,
            heartRate=?,
            stressLevel=?,
            fallRisk=?,
            spo2=?
        WHERE id=?
    """,
        (
            screening["timestamp"],
            screening["status"],
            screening["hr_bpm"],
            "High" if screening["stress_class"] == "STRESS" else "Low",
            "High Risk" if screening["fall_detected"] == "FALL DETECTED" else "Low Risk",
            screening["spo2_pct"],
            screening["patient_id"],
        ),
    )

    conn.commit()
    conn.close()


def get_screening_by_id(screening_id: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM screenings WHERE id = ?", (screening_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        d = dict(row)
        if d.get("raw_metrics_json"):
            d["raw_metrics"] = json.loads(d["raw_metrics_json"])
        return d
    return None


def get_screenings_by_patient_id(patient_id: str) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM screenings WHERE patient_id = ? ORDER BY id DESC", (patient_id,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    for r in rows:
        if r.get("raw_metrics_json"):
            r["raw_metrics"] = json.loads(r["raw_metrics_json"])
    return rows


def get_alerts() -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM alerts ORDER BY id DESC")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    for r in rows:
        r["resolved"] = bool(r["resolved"])
    return rows


def add_alert(alert: Dict[str, Any]):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO alerts (id, patientId, title, description, severity, timestamp, resolved)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
        (
            alert["id"],
            alert["patientId"],
            alert["title"],
            alert["description"],
            alert["severity"],
            alert["timestamp"],
            1 if alert.get("resolved") else 0,
        ),
    )
    conn.commit()
    conn.close()


def resolve_alert(alert_id: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE alerts SET resolved = 1 WHERE id = ?", (alert_id,))
    conn.commit()
    conn.close()


def get_reports() -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM reports ORDER BY id DESC")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def add_report(report: Dict[str, Any]):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO reports (name, type, generatedOn, generatedBy)
        VALUES (?, ?, ?, ?)
    """,
        (report["name"], report["type"], report["generatedOn"], report["generatedBy"]),
    )
    conn.commit()
    conn.close()
