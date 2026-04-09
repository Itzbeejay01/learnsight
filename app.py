#!/usr/bin/env python3
"""
LearnSight Flask Application
============================
Main web application for the LearnSight student academic performance
prediction system. Provides web interface and API for predictions with
SHAP explanations.

Author: LearnSight Team
Version: 1.0.0
"""

import os
import pickle
import json
import base64
import io
import warnings
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import shap

from flask import Flask, render_template, request, jsonify, session, flash, redirect, url_for, g

# Suppress warnings
warnings.filterwarnings('ignore')

# Flask App Configuration
app = Flask(__name__)
app.secret_key = 'learnsight_secret_key_2026_education_ai'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Paths
MODELS_DIR = 'models'
STATIC_DIR = 'static'
DB_FILENAME = 'learnsight.sqlite'

# Global variables for loaded models
models = {}
preprocessor = None
target_encoder = None
feature_names = []
raw_feature_names = []
input_schema = None
best_model_name = 'XGBoost'
best_model = None
shap_explainer = None

# Feature categories for the form
FEATURE_CATEGORIES = {
    'Demographics': ['Age', 'Gender'],
    'Behavioral Factors': ['Study_Hours', 'Attendance_Rate', 'Assignment_Completion', 'Discussion_Participation'],
    'Psychological Factors': ['Motivation_Level', 'Stress_Level'],
    'Contextual Factors': ['Access_to_Resources', 'Learning_Style'],
    'Academic History': ['Previous_GPA', 'Midterm_Score', 'Quiz_Scores'],
    'Engagement Metrics': ['Library_Visits', 'Online_Forum_Activity']
}

# Feature descriptions for tooltips
FEATURE_DESCRIPTIONS = {
    'Age': 'Student age in years (18-30)',
    'Gender': 'Student gender identity',
    'Study_Hours': 'Average hours studied per week (0-50)',
    'Attendance_Rate': 'Percentage of classes attended (0-100)',
    'Assignment_Completion': 'Percentage of assignments completed on time (0-100)',
    'Discussion_Participation': 'Level of participation in class discussions (0-100)',
    'Motivation_Level': 'Self-reported motivation level (Low/Medium/High)',
    'Stress_Level': 'Self-reported stress level (Low/Medium/High)',
    'Access_to_Resources': 'Quality of access to learning resources (Poor/Fair/Good/Excellent)',
    'Learning_Style': 'Preferred learning style (Visual/Auditory/Kinesthetic/Reading)',
    'Previous_GPA': 'Grade Point Average from previous semester (1.0-4.0)',
    'Midterm_Score': 'Score on midterm examination (0-100)',
    'Quiz_Scores': 'Average score on quizzes (0-100)',
    'Library_Visits': 'Number of library visits per month (0-20)',
    'Online_Forum_Activity': 'Activity level in online learning forums (0-100)'
}

# Categorical feature options
CATEGORICAL_OPTIONS = {
    'Gender': ['Male', 'Female', 'Other'],
    'Motivation_Level': ['Low', 'Medium', 'High'],
    'Stress_Level': ['Low', 'Medium', 'High'],
    'Access_to_Resources': ['Poor', 'Fair', 'Good', 'Excellent'],
    'Learning_Style': ['Visual', 'Auditory', 'Kinesthetic', 'Reading/Writing']
}

# Grade to score mapping for risk calculation
GRADE_SCORES = {
    'A': 95,
    'B': 85,
    'C': 75,
    'D': 65,
    'F': 55
}


def _db_path():
    os.makedirs(app.instance_path, exist_ok=True)
    return os.path.join(app.instance_path, DB_FILENAME)


def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(_db_path())
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(error):
    db = g.pop('db', None)
    if db is not None:
        db.close()


def init_db():
    db = sqlite3.connect(_db_path())
    try:
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                model_name TEXT NOT NULL,
                inputs_json TEXT NOT NULL,
                prediction_label TEXT NOT NULL,
                prediction_index INTEGER NOT NULL,
                confidence REAL NOT NULL,
                probabilities_json TEXT NOT NULL,
                risk_json TEXT NOT NULL,
                shap_json TEXT NOT NULL,
                recommendations_json TEXT NOT NULL
            )
            """
        )
        db.commit()
    finally:
        db.close()


def save_prediction_record(model_name, form_data, prediction_label, prediction_index, confidence, probabilities, risk_level, shap_data, recommendations):
    db = get_db()
    cur = db.execute(
        """
        INSERT INTO predictions (
            created_at, model_name, inputs_json, prediction_label, prediction_index,
            confidence, probabilities_json, risk_json, shap_json, recommendations_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.utcnow().isoformat(),
            model_name,
            json.dumps(form_data),
            prediction_label,
            int(prediction_index),
            float(confidence),
            json.dumps(probabilities),
            json.dumps(risk_level),
            json.dumps(shap_data),
            json.dumps(recommendations),
        ),
    )
    db.commit()
    return int(cur.lastrowid)


def get_prediction_record(prediction_id):
    db = get_db()
    row = db.execute("SELECT * FROM predictions WHERE id = ?", (int(prediction_id),)).fetchone()
    if row is None:
        return None
    rec = dict(row)
    rec["inputs"] = json.loads(rec["inputs_json"])
    rec["probabilities"] = json.loads(rec["probabilities_json"])
    rec["risk_level"] = json.loads(rec["risk_json"])
    rec["shap_data"] = json.loads(rec["shap_json"])
    rec["recommendations"] = json.loads(rec["recommendations_json"])
    return rec


def list_prediction_records(limit=50):
    db = get_db()
    rows = db.execute("SELECT id, created_at, model_name, prediction_label, confidence FROM predictions ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
    return [dict(r) for r in rows]


def load_models():
    """
    Load all trained models and preprocessing objects at startup.
    This ensures models are loaded only once, not per request.
    """
    global models, preprocessor, target_encoder, feature_names, raw_feature_names, input_schema, best_model_name, best_model, shap_explainer
    
    print("\n" + "="*60)
    print("  LearnSight - Loading Models")
    print("="*60)
    
    try:
        # Load best model name
        best_model_file = os.path.join(MODELS_DIR, 'best_model.txt')
        if os.path.exists(best_model_file):
            with open(best_model_file, 'r') as f:
                best_model_name = f.read().strip()
            print(f"  Best model: {best_model_name}")
        
        # Load all models
        model_files = {
            'Random Forest': 'random_forest_model.pkl',
            'XGBoost': 'xgboost_model.pkl',
            'LightGBM': 'lightgbm_model.pkl'
        }
        
        for name, filename in model_files.items():
            model_path = os.path.join(MODELS_DIR, filename)
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    models[name] = pickle.load(f)
                print(f"  Loaded: {name}")
        
        preprocessor_path = os.path.join(MODELS_DIR, 'preprocessor.pkl')
        if os.path.exists(preprocessor_path):
            with open(preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)
            print("  Loaded: Preprocessor")
        
        # Load target encoder
        target_encoder_path = os.path.join(MODELS_DIR, 'target_encoder.pkl')
        if os.path.exists(target_encoder_path):
            with open(target_encoder_path, 'rb') as f:
                target_encoder = pickle.load(f)
            print("  Loaded: Target Encoder")
        
        # Load feature names
        feature_names_path = os.path.join(MODELS_DIR, 'feature_names.pkl')
        if os.path.exists(feature_names_path):
            with open(feature_names_path, 'rb') as f:
                feature_names = pickle.load(f)
            print(f"  Loaded: {len(feature_names)} features")

        schema_path = os.path.join(MODELS_DIR, 'input_schema.json')
        if os.path.exists(schema_path):
            with open(schema_path, 'r') as f:
                input_schema = json.load(f)
            raw_feature_names = [f['name'] for f in input_schema.get('features', [])]
            print(f"  Loaded: Input schema ({len(raw_feature_names)} raw features)")
        
        # Set best model
        if best_model_name in models:
            best_model = models[best_model_name]
            print(f"\n  Active Model: {best_model_name}")
        
        # Load SHAP explainer for best model
        explainer_path = os.path.join(MODELS_DIR, f'{best_model_name.lower().replace(" ", "_")}_explainer.pkl')
        if os.path.exists(explainer_path):
            with open(explainer_path, 'rb') as f:
                shap_explainer = pickle.load(f)
            print("  Loaded: SHAP Explainer")
        else:
            # Create new explainer if not saved
            shap_explainer = shap.TreeExplainer(best_model)
            print("  Created: New SHAP Explainer")
        
        print("="*60 + "\n")
        return True
        
    except Exception as e:
        print(f"  Error loading models: {str(e)}")
        print("  Please run: python train_models.py")
        print("="*60 + "\n")
        return False


def preprocess_input(form_data):
    """
    Preprocess form input data for prediction.
    
    Args:
        form_data: Dictionary of form inputs
        
    Returns:
        Processed DataFrame ready for prediction
    """
    if input_schema and input_schema.get('features'):
        fields = input_schema['features']
        row = {}
        for f in fields:
            name = f['name']
            val = form_data.get(name, None)
            if val is None or val == '':
                row[name] = np.nan
                continue
            if f.get('type') == 'numerical':
                try:
                    row[name] = float(val)
                except Exception:
                    row[name] = np.nan
            else:
                row[name] = str(val)
        raw_df = pd.DataFrame([row])
    else:
        raw_df = pd.DataFrame([form_data])

    if preprocessor is None:
        return raw_df, raw_df.to_numpy()

    X_t = preprocessor.transform(raw_df)
    return raw_df, X_t


def get_risk_level(predicted_grade):
    """
    Determine risk level based on predicted grade.
    
    Args:
        predicted_grade: Predicted letter grade (A, B, C, D, F)
        
    Returns:
        Dictionary with risk level, color, and score
    """
    score = GRADE_SCORES.get(predicted_grade, 70)
    
    if score < 50:
        return {
            'level': 'High Risk',
            'color': 'danger',
            'hex_color': '#dc3545',
            'score': score,
            'icon': 'exclamation-triangle',
            'message': 'This student requires immediate intervention and support.'
        }
    elif score < 70:
        return {
            'level': 'Medium Risk',
            'color': 'warning',
            'hex_color': '#ffc107',
            'score': score,
            'icon': 'exclamation-circle',
            'message': 'This student may benefit from additional support and monitoring.'
        }
    else:
        return {
            'level': 'Low Risk',
            'color': 'success',
            'hex_color': '#28a745',
            'score': score,
            'icon': 'check-circle',
            'message': 'This student is performing well and on track for success.'
        }


def _display_feature_name(name):
    s = str(name)
    if s.startswith("num__"):
        s = s[5:]
    if s.startswith("cat__"):
        s = s[5:]
    s = s.replace("_", " ")
    s = s.replace(" = ", ": ")
    return s


def generate_shap_visualization(X_transformed, prediction_numeric):
    """
    Generate SHAP visualization for the prediction.
    
    Args:
        input_data: Preprocessed input DataFrame
        prediction: Predicted class
        
    Returns:
        Dictionary with SHAP data for visualization
    """
    try:
        shap_values = shap_explainer.shap_values(X_transformed)

        if isinstance(shap_values, list):
            class_list = list(getattr(best_model, "classes_", []))
            class_idx = class_list.index(prediction_numeric) if prediction_numeric in class_list else int(np.argmax(best_model.predict_proba(X_transformed)[0]))
            shap_vals = shap_values[class_idx][0]
            base_value = shap_explainer.expected_value[class_idx] if isinstance(shap_explainer.expected_value, (list, np.ndarray)) else shap_explainer.expected_value
        else:
            shap_vals = shap_values[0]
            base_value = shap_explainer.expected_value if not isinstance(shap_explainer.expected_value, (list, np.ndarray)) else shap_explainer.expected_value[0]

        feats = feature_names if feature_names else [f"f{i}" for i in range(len(shap_vals))]
        feature_importance = []
        for i, feat in enumerate(feats):
            sval = float(shap_vals[i]) if i < len(shap_vals) else 0.0
            feature_importance.append({
                "feature": str(feat),
                "value": sval,
                "display_name": _display_feature_name(feat),
            })

        feature_importance.sort(key=lambda x: abs(x["value"]), reverse=True)
        positive_factors = [f for f in feature_importance if f["value"] > 0][:5]
        negative_factors = [f for f in feature_importance if f["value"] < 0][:5]

        prediction_grade = target_encoder.inverse_transform([prediction_numeric])[0] if target_encoder else str(prediction_numeric)

        return {
            "feature_importance": feature_importance[:10],
            "positive_factors": positive_factors,
            "negative_factors": negative_factors,
            "base_value": float(base_value),
            "prediction": prediction_grade,
        }
        
    except Exception as e:
        print(f"SHAP visualization error: {str(e)}")
        return {
            'feature_importance': [],
            'positive_factors': [],
            'negative_factors': [],
            'error': str(e)
        }


def generate_recommendations(shap_data, risk_level):
    """
    Generate personalized recommendations based on SHAP analysis.
    
    Args:
        shap_data: SHAP explanation data
        risk_level: Risk level dictionary
        
    Returns:
        List of recommendation strings
    """
    recommendations = []
    
    # Analyze negative factors for recommendations
    negative_factors = shap_data.get('negative_factors', [])
    
    factor_recommendations = {
        'Study_Hours': 'Increase weekly study hours and establish a consistent study schedule.',
        'Attendance_Rate': 'Improve class attendance to ensure consistent learning.',
        'Assignment_Completion': 'Focus on completing all assignments on time.',
        'Discussion_Participation': 'Participate more actively in class discussions.',
        'Previous_GPA': 'Review foundational concepts from previous courses.',
        'Midterm_Score': 'Seek additional help to prepare for upcoming examinations.',
        'Quiz_Scores': 'Practice with additional quiz materials to reinforce learning.',
        'Library_Visits': 'Utilize library resources more frequently for studying.',
        'Online_Forum_Activity': 'Engage more in online learning forums and discussions.',
        'Motivation_Level': 'Consider meeting with an academic advisor to boost motivation.',
        'Stress_Level': 'Explore stress management techniques or counseling services.',
        'Access_to_Resources': 'Seek additional learning resources and support materials.'
    }
    
    for factor in negative_factors[:3]:
        feature = factor['feature']
        if feature in factor_recommendations:
            recommendations.append(factor_recommendations[feature])
    
    # Add general recommendations based on risk level
    if risk_level['level'] == 'High Risk':
        recommendations.append('Schedule an immediate meeting with academic advisor.')
        recommendations.append('Consider enrolling in tutoring or study skills workshops.')
    elif risk_level['level'] == 'Medium Risk':
        recommendations.append('Monitor progress closely and check in regularly.')
        recommendations.append('Consider forming study groups with peers.')
    else:
        recommendations.append('Continue current study habits and maintain performance.')
        recommendations.append('Consider mentoring other students who may need help.')
    
    return recommendations


def get_model_comparison_data():
    """
    Get model comparison metrics for display.
    
    Returns:
        Dictionary with model comparison data
    """
    # These would ideally be loaded from training results
    # For now, using placeholder values that would be updated after training
    comparison_data = {
        'models': ['Random Forest', 'XGBoost', 'LightGBM'],
        'metrics': {
            'Accuracy': [0.85, 0.89, 0.87],
            'Precision': [0.84, 0.88, 0.86],
            'Recall': [0.83, 0.87, 0.85],
            'F1-Score': [0.83, 0.87, 0.85]
        },
        'best_model': best_model_name,
        'why_chosen': [
            'Highest overall accuracy on test data',
            'Better handling of feature interactions',
            'Superior performance across all metrics',
            'Robust to overfitting with regularization'
        ]
    }
    
    return comparison_data


# ═══════════════════════════════════════════════════════════════════════════════
# Flask Routes
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    """
    Landing page with system overview.
    """
    if best_model is None:
        flash('Models not loaded. Please run: python train_models.py', 'error')
    
    return render_template(
        'landing.html',
        models_loaded=(best_model is not None),
        active_model=best_model_name,
    )


@app.route('/input')
def input_page():
    if best_model is None:
        flash('Models not loaded. Please run: python train_models.py', 'error')

    return render_template(
        'index.html',
        feature_categories=FEATURE_CATEGORIES,
        feature_descriptions=FEATURE_DESCRIPTIONS,
        categorical_options=CATEGORICAL_OPTIONS,
        models_loaded=(best_model is not None),
        active_model=best_model_name,
    )


@app.route('/design-system')
def design_system():
    return render_template('design_system.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction request from form submission.
    """
    try:
        # Check if models are loaded
        if best_model is None:
            flash('Models not loaded. Please run: python train_models.py', 'error')
            return redirect(url_for('input_page'))
        
        # Get form data
        form_data = request.form.to_dict()
        
        # Validate required fields
        required_fields = []
        for category, fields in FEATURE_CATEGORIES.items():
            required_fields.extend(fields)
        
        missing_fields = [f for f in required_fields if f not in form_data or not form_data[f]]
        if missing_fields:
            flash(f'Missing required fields: {", ".join(missing_fields)}', 'error')
            return render_template(
                'index.html',
                feature_categories=FEATURE_CATEGORIES,
                feature_descriptions=FEATURE_DESCRIPTIONS,
                categorical_options=CATEGORICAL_OPTIONS,
                models_loaded=True,
                form_data=form_data,
                active_model=best_model_name,
            )
        
        # Preprocess input
        raw_df, X_t = preprocess_input(form_data)
        
        # Make prediction (returns numeric class)
        prediction_numeric = best_model.predict(X_t)[0]
        
        # Decode prediction to grade letter
        prediction = target_encoder.inverse_transform([prediction_numeric])[0]
        
        # Get prediction probabilities if available
        if hasattr(best_model, 'predict_proba'):
            probabilities = best_model.predict_proba(X_t)[0]
            prob_dict = {target_encoder.inverse_transform([cls])[0]: float(prob) 
                        for cls, prob in zip(best_model.classes_, probabilities)}
            top_prob = max(prob_dict.values())
        else:
            prob_dict = {prediction: 1.0}
            top_prob = 1.0
        
        # Determine risk level
        risk_level = get_risk_level(prediction)
        
        # Generate SHAP explanation
        shap_data = generate_shap_visualization(X_t, prediction_numeric)
        
        # Generate recommendations
        recommendations = generate_recommendations(shap_data, risk_level)

        prediction_id = save_prediction_record(
            model_name=best_model_name,
            form_data=form_data,
            prediction_label=prediction,
            prediction_index=int(prediction_numeric),
            confidence=float(top_prob),
            probabilities=prob_dict,
            risk_level=risk_level,
            shap_data=shap_data,
            recommendations=recommendations,
        )

        return redirect(url_for('results', prediction_id=prediction_id))
        
    except Exception as e:
        flash(f'LearnSight couldn\'t process your request: {str(e)}', 'error')
        return redirect(url_for('input_page'))


@app.route('/results/<int:prediction_id>')
def results(prediction_id):
    rec = get_prediction_record(prediction_id)
    if rec is None:
        flash('Prediction not found.', 'error')
        return redirect(url_for('history'))

    return render_template(
        'results.html',
        prediction=rec['prediction_label'],
        probability=float(rec['confidence']),
        risk_level=rec['risk_level'],
        shap_data=rec['shap_data'],
        recommendations=rec['recommendations'],
        form_data=rec['inputs'],
        feature_descriptions=FEATURE_DESCRIPTIONS,
        prediction_id=rec['id'],
        created_at=rec['created_at'],
        model_used=rec['model_name'],
        probabilities=rec['probabilities'],
    )


@app.route('/history')
def history():
    records = list_prediction_records(limit=100)
    return render_template('history.html', records=records, models_loaded=(best_model is not None))


@app.route('/compare')
def compare():
    """
    Model comparison page showing performance metrics.
    """
    comparison_data = get_model_comparison_data()
    
    return render_template('comparison.html',
                         comparison=comparison_data,
                         models=list(models.keys()))


@app.route('/about')
def about():
    """
    About page with project information.
    """
    return render_template('about.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    JSON API endpoint for predictions.
    
    Request Body:
        JSON object with student features
        
    Returns:
        JSON with prediction and SHAP values
    """
    try:
        # Check if models are loaded
        if best_model is None:
            return jsonify({
                'success': False,
                'error': 'Models not loaded. Please run: python train_models.py'
            }), 503
        
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Preprocess input
        raw_df, X_t = preprocess_input(data)
        
        # Make prediction (returns numeric class)
        prediction_numeric = best_model.predict(X_t)[0]
        
        # Decode prediction to grade letter
        prediction = target_encoder.inverse_transform([prediction_numeric])[0]
        
        # Get probabilities
        if hasattr(best_model, 'predict_proba'):
            probabilities = best_model.predict_proba(X_t)[0]
            prob_dict = {target_encoder.inverse_transform([cls])[0]: float(prob) 
                        for cls, prob in zip(best_model.classes_, probabilities)}
        else:
            prob_dict = {prediction: 1.0}
        
        # Get risk level
        risk_level = get_risk_level(prediction)
        
        # Generate SHAP explanation (use numeric for SHAP)
        shap_data = generate_shap_visualization(X_t, prediction_numeric)
        
        # Build response
        response = {
            'success': True,
            'prediction': prediction,
            'confidence': float(max(prob_dict.values())),
            'probabilities': prob_dict,
            'risk_level': risk_level,
            'shap_values': shap_data.get('feature_importance', []),
            'top_positive_factors': [
                {'feature': f['feature'], 'contribution': f['value']}
                for f in shap_data.get('positive_factors', [])
            ],
            'top_negative_factors': [
                {'feature': f['feature'], 'contribution': f['value']}
                for f in shap_data.get('negative_factors', [])
            ],
            'model_used': best_model_name,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/models')
def api_models():
    """
    API endpoint to get information about loaded models.
    """
    return jsonify({
        'success': True,
        'models_loaded': list(models.keys()),
        'active_model': best_model_name,
        'features_transformed': feature_names,
        'features_raw': raw_feature_names,
        'feature_categories': FEATURE_CATEGORIES,
        'categorical_options': CATEGORICAL_OPTIONS
    })


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return render_template('landing.html', models_loaded=(best_model is not None), active_model=best_model_name), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


# ═══════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

init_db()
models_loaded = load_models()

if __name__ == '__main__':
    print("  Starting LearnSight server...")
    print("  URL: http://localhost:5000")
    print("  Press Ctrl+C to stop\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
