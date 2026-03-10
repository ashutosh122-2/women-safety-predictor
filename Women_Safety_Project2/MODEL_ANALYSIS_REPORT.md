# ML Model Analysis Report
## Women Safety Project

---

## ✅ Current Model Status

**Model is working and reading the dataset**: `C:\Users\user\Desktop\synthetic_up_locational_risk_data.csv`

### Model Performance Metrics:
- **Accuracy**: 100% (both CV and test)
- **Precision**: 100%
- **Recall**: 100%
- **F1 Score**: 100%
- **ROC-AUC**: 1.0 (perfect)

---

## ❌ Critical Problem: Data Leakage

### The Issue
The 100% accuracy is **NOT GOOD** - it indicates severe **data leakage** in the dataset.

### Root Cause Analysis

#### 1. **Perfect Feature-Target Correlation**
```
When Is_Night_Risk = 0 → Target_Risk_Y is ALWAYS 0 (1050/1050 cases)
When Is_Night_Risk = 1 → Target_Risk_Y can be 0 or 1 (60% risky)
```

The model learned a simple rule:
- **If not night → 100% safe**
- This is unrealistic for real-world safety prediction

#### 2. **Feature Redundancy**
- `Is_Night_Risk` is directly derived from `Hour`
- Having both features means redundant information
- Model memorizes patterns instead of learning generalizable rules

#### 3. **Location Data Issues**
- Same Latitude/Longitude for entire district
- Examples:
  - Lucknow: (26.8467, 80.9462) for ALL locations
  - Kanpur: (26.4499, 80.3319) for ALL locations
- Lat/Long adds NO value beyond District name

#### 4. **Synthetic Data Limitations**
- Dataset has 1,680 rows with perfect patterns
- All 1,680 feature combinations are unique (no natural variation)
- Risk ratios are artificially consistent:
  - Lucknow, Kanpur, Ghaziabad: 37.5% risk
  - Other districts: 16% risk

---

## 🚨 Why This Won't Work in Production

1. **Oversimplification**: Real-world safety isn't 100% predictable by time of day
2. **No generalization**: Model hasn't learned meaningful patterns
3. **Data mismatch**: Synthetic data doesn't represent real crime patterns
4. **Brittle predictions**: Any deviation from training patterns will fail

### Example Failures:
- Model will predict ALL daytime hours as 100% safe (unrealistic)
- Cannot handle new districts or locations
- Ignores contextual factors (events, weather, specific location characteristics)

---

## ✅ Recommendations

### Immediate Actions

#### 1. **Remove Data Leakage**
Remove the `Is_Night_Risk` feature since it's redundant with `Hour`:
```bash
python backend/train_model.py --csv <dataset> --exclude-loc
```

#### 2. **Get Real Data**
Replace synthetic data with:
- **Actual crime reports** from UP Police
- **Crowd-sourced safety ratings**
- **Incident reports** with timestamps and locations
- **Demographic and infrastructure data**

#### 3. **Add Realistic Features**
- Street lighting availability
- Police station proximity
- Population density (actual, not binary)
- Historical crime rates
- Public transport availability
- Business operating hours

#### 4. **Diverse Location Data**
- Multiple lat/long points per district
- Specific neighborhoods/areas
- Point-of-interest data (hospitals, police stations, etc.)

### Expected Realistic Performance

For a production-ready safety model:
- **Accuracy**: 70-85% (good performance)
- **Precision**: 65-80%
- **Recall**: 70-85%

Lower but realistic metrics mean the model is learning actual patterns, not memorizing data.

---

## 📊 Dataset Summary

```
Total rows: 1,680
Districts: 10
Features: 7
Target distribution:
  - Safe (0): 1,302 (77.5%)
  - Risky (1): 378 (22.5%)

Districts covered:
  High-risk cities (37.5%): Lucknow, Kanpur, Ghaziabad
  Lower-risk cities (16%): Agra, Aligarh, Bareilly, Gorakhpur, Meerut, Prayagraj, Varanasi
```

---

## 🔧 Technical Details

### Files Checked:
- ✅ Model: `backend/models/model.pkl`
- ✅ Dataset: `C:\Users\user\Desktop\synthetic_up_locational_risk_data.csv`
- ✅ Metrics: `backend/models/metrics.json`
- ✅ Training script: `backend/train_model.py`
- ✅ Evaluation script: `backend/evaluate_model.py`

### Model Type:
- **Algorithm**: Random Forest Classifier (300 trees)
- **Preprocessing**: OneHotEncoder for District, passthrough for numeric features
- **Cross-validation**: 5-fold stratified

---

## 📝 Next Steps

1. **Remove `Is_Night_Risk` feature** from training data
2. **Collect or generate better synthetic data** with realistic variance
3. **Add more meaningful features** (see recommendations above)
4. **Retrain model** and expect 70-85% accuracy (realistic range)
5. **Test with real users** to gather feedback
6. **Iterate with actual incident data** when available

---

## Conclusion

**The model is working perfectly from a technical standpoint** - it reads the dataset and makes predictions with 100% accuracy. 

However, **100% accuracy is a red flag**, not a success. It indicates the model memorized artificial patterns in synthetic data rather than learning real-world safety patterns.

**Action Required**: Fix data leakage and use realistic data for production deployment.
