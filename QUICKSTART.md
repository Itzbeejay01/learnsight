# LearnSight Quick Start Guide

Get started with LearnSight in minutes!

## 🚀 Installation & Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Models

```bash
python train_models.py
```

This will:
- Generate sample student data (or use your own in `data/student_data.csv`)
- Train 3 ML models (Random Forest, XGBoost, LightGBM)
- Evaluate and compare performance
- Save models to `models/` directory

### 3. Run the Application

```bash
python app.py
```

Open your browser and navigate to: **http://localhost:5000**

---

## 📊 Using the Web Interface

### Making a Prediction

1. **Fill in Student Information** across 6 categories:
   - 📊 Demographics (Age, Gender)
   - 📚 Behavioral (Study Hours, Attendance, etc.)
   - 🧠 Psychological (Motivation, Stress)
   - 🌍 Contextual (Resources, Learning Style)
   - 🎓 Academic (GPA, Scores)
   - 💻 Engagement (Library visits, Forum activity)

2. **Click "Analyze Performance"**

3. **View Results**:
   - Predicted Final Grade
   - Risk Level Assessment
   - SHAP Feature Importance
   - Personalized Recommendations

### Sample Student Data

Try this example:

| Feature | Value |
|---------|-------|
| Age | 21 |
| Gender | Female |
| Study Hours | 18.5 |
| Attendance Rate | 88.0 |
| Assignment Completion | 92.0 |
| Discussion Participation | 75.0 |
| Motivation Level | High |
| Stress Level | Medium |
| Access to Resources | Good |
| Learning Style | Visual |
| Previous GPA | 3.4 |
| Midterm Score | 82.0 |
| Quiz Scores | 85.0 |
| Library Visits | 10 |
| Online Forum Activity | 65.0 |

---

## 🔌 Using the API

### Make a Prediction

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

### Response

```json
{
  "success": true,
  "prediction": "B",
  "confidence": 0.85,
  "risk_level": {
    "level": "Low Risk",
    "score": 85
  },
  "model_used": "Random Forest"
}
```

### Get Model Information

```bash
curl http://localhost:5000/api/models
```

---

## 📁 Project Structure

```
learnsight/
├── data/               # Student dataset
├── models/             # Trained ML models
├── static/             # CSS, JS, images
├── templates/          # HTML templates
├── train_models.py     # Training script
├── app.py              # Flask application
├── requirements.txt    # Dependencies
└── README.md           # Full documentation
```

---

## 🛠️ Troubleshooting

### Models Not Found

If you see "Models not loaded":
```bash
python train_models.py
```

### Port Already in Use

Change the port in `app.py`:
```python
app.run(host='0.0.0.0', port=5001)  # Use different port
```

### Missing Dependencies

```bash
pip install -r requirements.txt --upgrade
```

---

## 📞 Support

- **Documentation**: See [README.md](README.md)
- **Issues**: Check GitHub issues
- **Email**: support@learnsight.edu

---

**Powered by LearnSight © 2026** - Empowering Educators with Explainable AI
