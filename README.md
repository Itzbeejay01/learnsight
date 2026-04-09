# LearnSight

LearnSight is a Flask web application for predicting student academic performance using multiple machine learning models and explainable AI (SHAP). It provides a web interface, a JSON API, and stores prediction history in a local SQLite database.

## Features

- Predict final grade and risk level from student features
- Supports Random Forest, XGBoost, and LightGBM
- SHAP explanations for transparency and interpretability
- Web UI and REST-style JSON endpoints
- Prediction history persisted to SQLite (`instance/learnsight.sqlite`)

## Getting started

### Prerequisites

- Python 3.9.6
- pip

### Installation

```bash
git clone https://github.com/Itzbeejay01/learnsight.git
cd learnsight

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### Model artifacts (required)

Model artifacts are intentionally not tracked in this repository (see `.gitignore`). Download the latest trained artifacts from Google Drive:

- Google Drive (models): https://drive.google.com/REPLACE_WITH_LINK

Extract/copy the downloaded files into `models/` so the directory contains:

- `best_model.txt`
- `preprocessor.pkl`
- `feature_names.pkl`
- `input_schema.json`
- `random_forest_model.pkl`, `xgboost_model.pkl`, `lightgbm_model.pkl`
- `random_forest_explainer.pkl`, `xgboost_explainer.pkl`, `lightgbm_explainer.pkl`
- Optional: `target_encoder.pkl`

### Run the app

```bash
python app.py
```

Open `http://localhost:5000`.

### Train models (optional)

If you prefer to train locally instead of downloading model artifacts:

```bash
python train_models.py
```

## API

- `POST /api/predict` accepts JSON input using the same feature names as the web form and returns prediction details.
- `GET /api/models` returns information about the loaded models.

Example:

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 21,
    "Gender": "Female",
    "Study_Hours": 18.5,
    "Attendance_Rate": 88.0,
    "Assignment_Completion": 92.0,
    "Discussion_Participation": 75.0,
    "Motivation_Level": "High",
    "Stress_Level": "Medium",
    "Access_to_Resources": "Good",
    "Learning_Style": "Visual",
    "Previous_GPA": 3.4,
    "Midterm_Score": 82.0,
    "Quiz_Scores": 85.0,
    "Library_Visits": 10,
    "Online_Forum_Activity": 65.0
  }'
```

## Project layout

```text
learnsight/
  app.py
  train_models.py
  requirements.txt
  templates/
  static/
  data/
  models/        # downloaded or locally trained artifacts (not committed)
  instance/      # local SQLite DB (not committed)
  reports/
```

## License

No license has been specified yet.

## Contributing

Issues and pull requests are welcome. Please open an issue to discuss significant changes before submitting a PR.
