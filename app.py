from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Demo ML model:
# In a production hackathon deployment, replace this synthetic training data
# with a verified landslide dataset for the North Eastern Region (NER).
rng = np.random.default_rng(42)
n = 2500

rainfall = rng.uniform(0, 300, n)          # mm/day
soil_moisture = rng.uniform(10, 100, n)   # %
slope = rng.uniform(0, 60, n)              # degrees
elevation = rng.uniform(0, 3500, n)        # metres
temperature = rng.uniform(5, 40, n)        # °C
past_events = rng.integers(0, 6, n)       # nearby historical events

risk_signal = (
    0.40 * (rainfall / 300) +
    0.25 * (soil_moisture / 100) +
    0.20 * (slope / 60) +
    0.07 * (past_events / 5) +
    0.05 * (elevation / 3500) +
    0.03 * (1 - np.abs(temperature - 24) / 20)
)
probability = np.clip(risk_signal + rng.normal(0, 0.06, n), 0, 1)
y = (probability >= 0.55).astype(int)

X = np.column_stack([
    rainfall, soil_moisture, slope, elevation, temperature, past_events
])

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    random_state=42
)
model.fit(X, y)

LOCATIONS = [
    {"name": "Gangtok", "state": "Sikkim", "lat": 27.3389, "lon": 88.6065},
    {"name": "Aizawl", "state": "Mizoram", "lat": 23.7271, "lon": 92.7176},
    {"name": "Shillong", "state": "Meghalaya", "lat": 25.5788, "lon": 91.8933},
    {"name": "Kohima", "state": "Nagaland", "lat": 25.6751, "lon": 94.1086},
    {"name": "Itanagar", "state": "Arunachal Pradesh", "lat": 27.0844, "lon": 93.6053},
    {"name": "Imphal", "state": "Manipur", "lat": 24.8170, "lon": 93.9368},
    {"name": "Agartala", "state": "Tripura", "lat": 23.8315, "lon": 91.2868},
    {"name": "Guwahati", "state": "Assam", "lat": 26.1445, "lon": 91.7362},
]

def classify(score):
    if score < 25:
        return "LOW", "green", "Conditions are currently relatively stable."
    if score < 50:
        return "MODERATE", "yellow", "Monitor rainfall and terrain conditions."
    if score < 75:
        return "HIGH", "orange", "Avoid risky slopes and follow local advisories."
    return "CRITICAL", "red", "Immediate caution required. Follow official evacuation instructions."

@app.route("/")
def index():
    return render_template("index.html", locations=LOCATIONS)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    try:
        values = [
            float(data["rainfall"]),
            float(data["soil_moisture"]),
            float(data["slope"]),
            float(data["elevation"]),
            float(data["temperature"]),
            float(data["past_events"]),
        ]
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Please provide valid numeric inputs."}), 400

    x = np.array([values])
    model_probability = float(model.predict_proba(x)[0][1])

    # Blend ML probability with transparent domain factors for an explainable demo.
    rain_factor = min(values[0] / 200, 1)
    soil_factor = min(values[1] / 100, 1)
    slope_factor = min(values[2] / 45, 1)
    history_factor = min(values[5] / 5, 1)

    score = round(100 * (
        0.55 * model_probability +
        0.20 * rain_factor +
        0.15 * soil_factor +
        0.07 * slope_factor +
        0.03 * history_factor
    ))
    score = max(0, min(100, score))

    level, color, message = classify(score)
    probability_pct = round(model_probability * 100)

    return jsonify({
        "risk_score": score,
        "probability": probability_pct,
        "level": level,
        "color": color,
        "message": message,
        "inputs": {
            "rainfall": values[0],
            "soil_moisture": values[1],
            "slope": values[2],
            "elevation": values[3],
            "temperature": values[4],
            "past_events": values[5],
        }
    })

@app.route("/api/locations")
def locations():
    return jsonify(LOCATIONS)

if __name__ == "__main__":
    app.run(debug=True)
