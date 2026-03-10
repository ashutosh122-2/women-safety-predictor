# Quick Start Guide - Fixed Model (90-95% Accuracy)

## ✅ What Was Fixed
- Removed data leakage (`Is_Night_Risk` feature)
- Model now has realistic **93.97% CV / 95.24% Test accuracy**
- Ready for production use

---

## 🚀 Running in VS Code

### 1. Open Project
```bash
# In VS Code terminal (PowerShell)
cd C:\Users\user\Documents\Women_Safety_Project
```

### 2. Activate Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Run the Application
```powershell
# Set the fixed dataset path
$env:CSV_PATH="C:\Users\user\Desktop\synthetic_up_locational_risk_data_fixed.csv"

# Run the Flask app
python backend/app.py
```

### 4. Open in Browser
Visit: http://localhost:5000

---

## 🔧 Alternative: Run Without CSV (Demo Mode)
```powershell
# The app will use fallback demo predictions
python backend/app.py
```

---

## 📊 Test the Model

### Quick Test:
```powershell
python -c "import joblib; import pandas as pd; model = joblib.load('backend/models/model.pkl'); print('✅ Model loaded'); test = pd.DataFrame([{'District': 'Lucknow', 'Latitude': 26.8467, 'Longitude': 80.9462, 'Hour': 22, 'Day_of_Week': 1, 'Is_High_Density_Area': 1}]); pred = model.predict_proba(test); print(f'Lucknow at 10 PM: Safety Score = {int((1-pred[0][1])*100)}/100')"
```

### Evaluate Model Performance:
```powershell
python backend/evaluate_model_fixed.py
```

Expected output:
- Accuracy: ~95%
- Precision: 100%
- Recall: 78.9%

---

## 📁 Files You Need

### Fixed Dataset (Without Is_Night_Risk):
```
C:\Users\user\Desktop\synthetic_up_locational_risk_data_fixed.csv
```

### Model Files:
- `backend/models/model.pkl` - Main model (90-95% accuracy) ✅
- `backend/models/model_fixed_90.pkl` - Backup copy
- `backend/models/metrics.json` - Performance metrics

### Scripts:
- `backend/app.py` - Flask web application
- `backend/train_model_fixed.py` - Retrain model
- `backend/evaluate_model_fixed.py` - Test model accuracy

---

## 🔄 Retrain Model (Optional)

If you want to retrain from scratch:

```powershell
# Default (90-95% accuracy)
python backend/train_model_fixed.py

# Custom hyperparameters
python backend/train_model_fixed.py --max-depth 10 --min-samples-split 15
```

---

## ⚙️ VS Code Launch Configuration

Create `.vscode/launch.json`:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Flask App",
            "type": "python",
            "request": "launch",
            "module": "flask",
            "env": {
                "FLASK_APP": "backend/app.py",
                "FLASK_ENV": "development",
                "CSV_PATH": "C:\\Users\\user\\Desktop\\synthetic_up_locational_risk_data_fixed.csv"
            },
            "args": [
                "run",
                "--no-debugger",
                "--no-reload"
            ],
            "jinja": true,
            "justMyCode": true
        }
    ]
}
```

Then press `F5` to run!

---

## 🧪 Testing Commands

### 1. Check Model Accuracy:
```powershell
python backend/evaluate_model_fixed.py
```

### 2. Analyze Data (Check for leakage):
```powershell
python backend/analyze_leakage.py
```

### 3. Run Tests (if available):
```powershell
pytest
```

---

## 📈 Current Performance

```
✅ Cross-validation Accuracy: 93.97%
✅ Test Set Accuracy: 95.24%
✅ Precision: 100%
✅ Recall: 78.9%
✅ F1 Score: 88.2%

Status: Production Ready 🚀
```

---

## 🐛 Troubleshooting

### Issue: Module not found
```powershell
pip install -r requirements.txt
```

### Issue: Wrong Python interpreter
In VS Code:
1. `Ctrl+Shift+P`
2. Type: "Python: Select Interpreter"
3. Choose: `.venv\Scripts\python.exe`

### Issue: Port 5000 in use
```powershell
# Use different port
$env:PORT=5001
python backend/app.py
```

### Issue: Model file not found
```powershell
# Retrain model
python backend/train_model_fixed.py
```

---

## 📖 Documentation

- **Full Analysis**: `MODEL_ANALYSIS_REPORT.md`
- **Fix Summary**: `MODEL_FIX_SUMMARY.md`
- **This Guide**: `QUICK_START.md`

---

## ✅ Checklist

- [x] Data leakage removed
- [x] Model accuracy: 90-95% ✅
- [x] Fixed dataset created
- [x] App updated to use new model
- [x] Production ready

**You're all set!** 🎉
