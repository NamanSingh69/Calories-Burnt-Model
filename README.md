# Calories Burnt Model — Technical Report

## Architecture Overview

| Component | Technology |
|-----------|-----------|
| Language | Python 3.x |
| ML Framework | XGBoost |
| Preprocessing | Scikit-learn (StandardScaler) |
| Serialized Models | `.pkl` (Pickle) |

### Artifacts
```
calories_burned_xgb_model.pkl  (522 KB)  — Trained XGBoost regressor
scaler.pkl                     (1 KB)    — StandardScaler for feature normalization
Calorie Prediction Model.pdf              — Full project report/study
```

## Study Findings

- **Objective**: Predict calories burned based on physiological inputs (age, weight, heart rate, etc.)
- **Model**: XGBoost Regressor with hyperparameter tuning
- **Preprocessing**: StandardScaler normalization of input features
- **Deployment Verdict**: ❌ **Not deployable on free tier** — No web server, no API. Script/notebook-only workflow.

## Local Setup Guide

```bash
# 1. Navigate to the project
cd "calories burnt model"

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install xgboost scikit-learn pandas numpy

# 4. Load and use the model
python -c "
import pickle
model = pickle.load(open('calories_burned_xgb_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
print('Model loaded successfully')
"
```

## 🔑 API Keys
No API keys required — fully offline ML project.
