# VitalAgent

**AI-Powered Multi-Modal Health Screening System Using MOMENT-1-Large Foundation Model**

VitalAgent is a full-stack clinical dashboard designed for community health workers to perform real-time, multi-modal health screenings. It leverages the MOMENT-1-Large time-series foundation model to extract features from wearable sensors and predict:
- Heart Rate (BPM)
- Blood Oxygen Saturation (SpO2)
- Stress Level
- Fall Detection

## Prerequisites
- Node.js (v18 or higher)
- Python 3.10+
- npm (Node Package Manager)

## Local Setup Instructions

### 1. Backend Setup (FastAPI & ML Inference)

1. Open a terminal and navigate to the project root directory (`VitalAgent`).
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
3. Activate the virtual environment:
   - On Windows:
     ```powershell
     .\.venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```
4. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Start the FastAPI backend server:
   ```bash
   python -m uvicorn src.api.main:app --reload --port 8000
   ```
   *The backend will now be running at `http://localhost:8000`.*

### 2. Frontend Setup (React + Vite)

1. Open a **new** terminal (keep the backend terminal running) and navigate to the frontend directory:
   ```bash
   cd frontend/ilash-health
   ```
2. Install the Node.js dependencies:
   ```bash
   npm install
   ```
3. Start the Vite development server:
   ```bash
   npm run dev
   ```
   *The frontend will typically run at `http://localhost:5173`. Open this URL in your web browser to view the application.*

## Usage Guide

1. **Dashboard:** View overall patient metrics, active alerts, and community trends.
2. **Patient Management:** View existing patient records or search for specific patients.
3. **New Screening Wizard:** Click on "New Screening" to upload patient sensor data.
4. **Sample Data for Testing:** You can use the provided CSV samples located in the `sample_csv_inputs/` directory (e.g., `sample_csv_inputs/hr/hr_sample_01.csv`) to simulate real wearable data uploads for each respective modality.
5. **Results & Alerts:** After the ML analysis completes (~3-5 seconds on CPU), the system will display the predicted vitals alongside color-coded clinical risk badges. Automatic alerts are generated for abnormal values.

## Key Directories
- `src/api/`: FastAPI backend server endpoints and database logic.
- `src/`: Core Python machine learning inference engine (`vitalagent_predict.py`).
- `frontend/ilash-health/`: React frontend application.
- `models/`: Pre-trained task heads (Random Forest models and fine-tuned PyTorch classifier).
- `data/`: Processed datasets and the SQLite database (`vitalagent.db`).
- `sample_csv_inputs/`: Pre-generated 512-sample test windows for all four screening modalities.
