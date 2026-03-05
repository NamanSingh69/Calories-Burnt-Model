"""
Calories Burnt Prediction — Web Inference API
Flask wrapper for the pre-trained XGBoost model.
"""
import os
import pickle
import logging
import numpy as np
from flask import Flask, request, jsonify, render_template_string

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Load pre-trained model and scaler at startup
MODEL_DIR = os.path.dirname(__file__)
try:
    with open(os.path.join(MODEL_DIR, 'calories_burned_xgb_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    logger.info("✅ Model and scaler loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    model = None
    scaler = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Calorie Burn Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0a0a0f; --surface: #12121a; --surface-2: #1a1a2e;
            --accent: #f97316; --accent-glow: rgba(249,115,22,0.12);
            --text: #e8e8f0; --text-muted: #8888a0;
            --border: rgba(255,255,255,0.06); --radius: 16px;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', sans-serif; background: var(--bg);
            color: var(--text); min-height: 100vh;
            display: flex; align-items: center; justify-content: center; padding: 2rem;
        }
        .container { width: 100%; max-width: 540px; }
        h1 {
            font-size: 2rem; font-weight: 700;
            background: linear-gradient(135deg, var(--accent), #fb923c);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .subtitle { color: var(--text-muted); font-size: 0.9rem; margin-bottom: 2rem; }
        .card {
            background: var(--surface); border: 1px solid var(--border);
            border-radius: var(--radius); padding: 1.5rem; margin-bottom: 1.5rem;
        }
        .field { margin-bottom: 1rem; }
        .field label {
            display: block; font-size: 0.75rem; font-weight: 500;
            color: var(--text-muted); text-transform: uppercase;
            letter-spacing: 0.05em; margin-bottom: 0.4rem;
        }
        .field input, .field select {
            width: 100%; padding: 0.75rem; background: var(--surface-2);
            border: 1px solid var(--border); border-radius: 10px;
            color: var(--text); font-family: inherit; font-size: 0.9rem;
        }
        .field input:focus, .field select:focus {
            outline: none; border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-glow);
        }
        .row { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
        button {
            width: 100%; padding: 1rem;
            background: linear-gradient(135deg, var(--accent), #ea580c);
            border: none; border-radius: 12px; color: white;
            font-family: inherit; font-size: 1rem; font-weight: 600;
            cursor: pointer; transition: transform 0.1s;
        }
        button:hover { transform: translateY(-1px); box-shadow: 0 8px 30px var(--accent-glow); }
        .result {
            display: none; text-align: center; padding: 2rem;
            background: var(--surface); border: 1px solid var(--accent);
            border-radius: var(--radius); margin-top: 1.5rem;
        }
        .result.show { display: block; }
        .calories { font-size: 3rem; font-weight: 700; color: var(--accent); }
        .cal-label { color: var(--text-muted); font-size: 0.85rem; margin-top: 0.25rem; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔥 Calorie Burn Predictor</h1>
        <p class="subtitle">XGBoost model predicting calories burned during exercise</p>
        <form id="form">
            <div class="card">
                <div class="row">
                    <div class="field">
                        <label>Gender</label>
                        <select id="gender"><option value="male">Male</option><option value="female">Female</option></select>
                    </div>
                    <div class="field">
                        <label>Age</label>
                        <input type="number" id="age" value="28" min="10" max="100">
                    </div>
                </div>
                <div class="row">
                    <div class="field">
                        <label>Height (cm)</label>
                        <input type="number" id="height" value="175" min="100" max="250" step="0.1">
                    </div>
                    <div class="field">
                        <label>Weight (kg)</label>
                        <input type="number" id="weight" value="70" min="30" max="200" step="0.1">
                    </div>
                </div>
                <div class="row">
                    <div class="field">
                        <label>Exercise Duration (min)</label>
                        <input type="number" id="duration" value="30" min="1" max="300">
                    </div>
                    <div class="field">
                        <label>Heart Rate (bpm)</label>
                        <input type="number" id="heart_rate" value="110" min="40" max="220">
                    </div>
                </div>
                <div class="field">
                    <label>Body Temperature (°C)</label>
                    <input type="number" id="body_temp" value="37.0" min="35" max="42" step="0.1">
                </div>
            </div>
            <button type="submit">🔥 Predict Calories</button>
        </form>
        <div class="result" id="result">
            <div class="calories" id="cal-value">—</div>
            <div class="cal-label">Estimated Calories Burned</div>
        </div>
    </div>
    <script>
        document.getElementById('form').addEventListener('submit', async e => {
            e.preventDefault();
            const btn = e.target.querySelector('button');
            btn.disabled = true;
            try {
                const data = {
                    gender: document.getElementById('gender').value,
                    age: +document.getElementById('age').value,
                    height: +document.getElementById('height').value,
                    weight: +document.getElementById('weight').value,
                    duration: +document.getElementById('duration').value,
                    heart_rate: +document.getElementById('heart_rate').value,
                    body_temp: +document.getElementById('body_temp').value,
                };
                const res = await fetch('/api/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                const result = await res.json();
                if (!res.ok) throw new Error(result.error);
                document.getElementById('cal-value').textContent = result.calories_burned.toFixed(1) + ' kcal';
                document.getElementById('result').classList.add('show');
            } catch (err) {
                document.getElementById('cal-value').textContent = err.message;
                document.getElementById('result').classList.add('show');
            } finally { btn.disabled = false; }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict calories burned from exercise parameters."""
    if model is None or scaler is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body required"}), 400

        gender_encoded = 0 if data.get('gender', 'male').lower() == 'male' else 1

        features = np.array([[
            gender_encoded,
            float(data.get('age', 28)),
            float(data.get('height', 175)),
            float(data.get('weight', 70)),
            float(data.get('duration', 30)),
            float(data.get('heart_rate', 110)),
            float(data.get('body_temp', 37.0)),
        ]])

        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]

        return jsonify({
            "calories_burned": round(float(prediction), 1),
            "input": data,
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
