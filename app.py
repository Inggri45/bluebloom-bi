from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import os
import webbrowser
from threading import Timer

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "latihan1.csv")

REQUIRED_COLUMNS = [
    "Student_ID", "Gender", "Age", "Study_Hours", "Sleep_Hours",
    "Attendance", "Exam_Score", "Monthly_Spending", "Screen_Hours",
    "Satisfaction"
]
MODEL_FEATURES = ["Study_Hours", "Attendance", "Sleep_Hours", "Screen_Hours"]


def load_data():
    """Load the provided CSV; create a fallback dataset only if the CSV is missing."""
    if os.path.exists(CSV_PATH):
        data = pd.read_csv(CSV_PATH)
    else:
        rng = np.random.default_rng(42)
        n = 100
        study = rng.integers(4, 18, n)
        attendance = rng.integers(60, 101, n)
        sleep = np.round(rng.uniform(4.5, 9, n), 1)
        screen = np.round(rng.uniform(2, 10, n), 1)
        score = np.clip(
            20 + study * 2.2 + attendance * 0.35 + sleep * 1.8 - screen * 1.1
            + rng.normal(0, 7, n), 0, 100
        ).round().astype(int)
        data = pd.DataFrame({
            "Student_ID": np.arange(1, n + 1),
            "Gender": rng.choice(["F", "M"], n),
            "Age": rng.integers(17, 24, n),
            "Study_Hours": study,
            "Sleep_Hours": sleep,
            "Attendance": attendance,
            "Exam_Score": score,
            "Monthly_Spending": rng.integers(400, 2001, n),
            "Screen_Hours": screen,
            "Satisfaction": rng.integers(1, 11, n)
        })
        data.to_csv(CSV_PATH, index=False)

    missing = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(
            "Kolom CSV belum sesuai. Kolom yang tidak ditemukan: " + ", ".join(missing)
        )

    # Convert numeric columns safely, and remove unusable rows.
    numeric_cols = [c for c in REQUIRED_COLUMNS if c not in ("Gender",)]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data["Gender"] = data["Gender"].astype(str).str.strip().str.upper()
    data = data.dropna(subset=REQUIRED_COLUMNS).copy()
    data["Gender"] = data["Gender"].replace({"FEMALE": "F", "WOMAN": "F", "MALE": "M", "MAN": "M"})
    data = data[data["Gender"].isin(["F", "M"])]
    return data


def fit_ols(data):
    """Fit a small ordinary least squares model using NumPy (no extra ML package)."""
    x = data[MODEL_FEATURES].astype(float).to_numpy()
    y = data["Exam_Score"].astype(float).to_numpy()
    design = np.column_stack([np.ones(len(x)), x])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    predicted = design @ coefficients
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
    rmse = float(np.sqrt(np.mean((y - predicted) ** 2)))
    return coefficients, r2, rmse


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data")
def api_data():
    try:
        data = load_data()
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    gender = request.args.get("gender", "All")
    filtered = data if gender == "All" else data[data["Gender"] == gender]
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    corr = filtered[numeric_cols].corr().replace([np.inf, -np.inf], np.nan).fillna(0)

    # Fit the model on the complete dataset so what-if estimates remain consistent
    # when the dashboard gender filter changes.
    coefficients, r2, rmse = fit_ols(data)
    feature_weights = {
        name: float(value) for name, value in zip(MODEL_FEATURES, coefficients[1:])
    }
    model = {
        "intercept": float(coefficients[0]),
        "coefficients": feature_weights,
        "r2": float(r2),
        "rmse": float(rmse),
        "features": MODEL_FEATURES
    }

    radar_metrics = ["Study_Hours", "Attendance", "Sleep_Hours", "Exam_Score", "Satisfaction"]
    radar = {}
    for g in ["F", "M"]:
        group = data[data["Gender"] == g]
        radar[g] = {
            metric: float(group[metric].mean()) if len(group) else 0
            for metric in radar_metrics
        }

    # Relationship ranking for Exam_Score; keep sign to show direction.
    score_corr = corr["Exam_Score"].drop(labels=["Exam_Score"], errors="ignore")
    score_corr = score_corr.reindex(score_corr.abs().sort_values(ascending=False).index)

    raw = filtered[REQUIRED_COLUMNS].round(2).to_dict(orient="records")
    stats = {
        "average_score": round(float(filtered["Exam_Score"].mean()), 1) if len(filtered) else 0,
        "average_study": round(float(filtered["Study_Hours"].mean()), 1) if len(filtered) else 0,
        "average_attendance": round(float(filtered["Attendance"].mean()), 1) if len(filtered) else 0,
        "average_satisfaction": round(float(filtered["Satisfaction"].mean()), 1) if len(filtered) else 0,
        "average_spending": round(float(filtered["Monthly_Spending"].mean()), 0) if len(filtered) else 0,
        "average_sleep": round(float(filtered["Sleep_Hours"].mean()), 1) if len(filtered) else 0,
        "average_screen": round(float(filtered["Screen_Hours"].mean()), 1) if len(filtered) else 0,
        "female_count": int((filtered["Gender"] == "F").sum()),
        "male_count": int((filtered["Gender"] == "M").sum()),
        "risk_count": int((filtered["Exam_Score"] < 60).sum()),
        "total_students": int(len(filtered)),
        "total_all": int(len(data)),
    }

    return jsonify({
        "raw_data": raw,
        "stats": stats,
        "numeric_cols": numeric_cols,
        "corr_matrix": {
            "labels": corr.columns.tolist(),
            "values": corr.round(3).values.tolist()
        },
        "score_correlations": [
            {"name": str(name), "value": round(float(value), 3)}
            for name, value in score_corr.items()
        ],
        "radar_data": radar,
        "model": model,
        "at_risk_count": stats["risk_count"],
        "total_students": stats["total_students"]
    })


if __name__ == "__main__":
    # Open the dashboard automatically when the server starts.
    Timer(1.2, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(debug=True, use_reloader=False)
