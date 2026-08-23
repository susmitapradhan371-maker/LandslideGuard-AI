# LandslideGuard AI

An AI-based early warning and landslide risk monitoring prototype for the
North Eastern Region (NER) of India.

## Problem Statement
**AI-Based Early Warning and Landslide Risk Monitoring System in NER**

## Features
- NER location selector
- AI/ML-based landslide risk prediction
- Risk score from 0–100
- Low / Moderate / High / Critical classification
- Explainable input factors
- NER location map
- Emergency safety guidance
- Simple responsive dashboard

## Run locally

```bash
python -m venv venv
```

Windows:
```bash
venv\Scripts\activate
```

macOS/Linux:
```bash
source venv/bin/activate
```

Install:
```bash
pip install -r requirements.txt
```

Run:
```bash
python app.py
```

Open:
`http://127.0.0.1:5000`

## Important
The included Random Forest model is trained on synthetic demonstration data so
the application works immediately for a hackathon prototype. For a real-world
system, train and validate the model using verified historical landslide,
rainfall, soil-moisture, terrain and geological datasets for NER, and connect
authorized real-time data sources.

## Suggested hackathon upgrades
1. Add live rainfall/weather API.
2. Add satellite/DEM-derived slope and elevation.
3. Use a verified historical landslide dataset.
4. Add SMS/push notification integration.
5. Add an authority/admin dashboard.
6. Add district-level risk heatmaps.
