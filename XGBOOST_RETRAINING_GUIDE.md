# XGBoost Retraining Guide - Phase 28

## Overview

Your XGBoost model (B1 brain) needs to be retrained with new features to improve accuracy. This guide shows exactly what to do in your Google Colab notebook.

## New Features Being Added

| Feature | Description | Why It Helps |
|---------|-------------|--------------|
| **volume_ratio** | Current volume / 20-period average | Detects volume spikes (strong moves) |
| **atr** | Average True Range | Measures volatility |
| **bb_position** | Price position within Bollinger Bands | Shows overbought/oversold conditions |
| **hour** | Hour of day (0-23) | Captures time-of-day patterns |
| **volume_sma** | 20-period volume moving average | Baseline for volume comparison |

## Step-by-Step Instructions

### Step 1: Open Your Colab Notebook

1. Go to: https://colab.research.google.com/drive/1GyC3jDt9tgWSSok2_KrxMio_AGwlSMc6
2. Sign in with your Google account
3. You should see your training code

### Step 2: Find the Feature Engineering Section

Look for a cell that says:
```python
# Feature Engineering
# or
# Prepare features
# or
# df['rsi'] = ...
```

### Step 3: Add New Features

**Find this section:**
```python
# Current features (BEFORE)
df['rsi'] = ...
df['adx'] = ...
df['ema'] = ...
```

**Add these new features after the existing ones:**

```python
# ── Phase 28: NEW FEATURES ────────────────────────────────────

# Volume Features
df['volume_sma'] = df['volume'].rolling(20).mean()
df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-9)

# Volatility (ATR)
h, l, c = df["high"], df["low"], df["close"]
tr = pd.concat([
    h - l,
    (h - c.shift(1)).abs(),
    (l - c.shift(1)).abs()
], axis=1).max(axis=1)
df['atr'] = tr.rolling(14).mean()

# Bollinger Bands
bb_sma = df['close'].rolling(20).mean()
bb_std = df['close'].rolling(20).std()
df['bb_upper'] = bb_sma + (bb_std * 2)
df['bb_lower'] = bb_sma - (bb_std * 2)
df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-9)

# Time Features
if hasattr(df.index, 'hour'):
    df['hour'] = df.index.hour
else:
    df['hour'] = 12  # Default to noon if no time index

print("✅ New features added:")
print(f"   - volume_ratio: {df['volume_ratio'].mean():.2f}")
print(f"   - atr: {df['atr'].mean():.2f}")
print(f"   - bb_position: {df['bb_position'].mean():.2f}")
```

### Step 4: Update Feature List

**Find this section:**
```python
# Current features (BEFORE)
features = ['rsi', 'adx', 'ema', 'pattern_score']
```

**Update to:**
```python
# ── Phase 28: UPDATED FEATURE LIST ────────────────────────────
features = [
    'rsi',                    # Existing
    'adx',                    # Existing
    'ema',                    # Existing
    'pattern_score',          # Existing
    'volume_ratio',           # NEW
    'atr',                    # NEW
    'bb_position',            # NEW
    'hour'                    # NEW
]

print(f"✅ Training with {len(features)} features: {features}")
```

### Step 5: Update XGBoost Training

**Find this section:**
```python
# Current training (BEFORE)
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1
)
```

**Update to:**
```python
# ── Phase 28: IMPROVED XGBOOST CONFIGURATION ──────────────────
model = xgb.XGBClassifier(
    n_estimators=150,          # Increased from 100
    max_depth=7,               # Increased from 6
    learning_rate=0.08,        # Slightly reduced for stability
    subsample=0.8,             # Add sampling for regularization
    colsample_bytree=0.8,      # Feature sampling
    reg_alpha=0.1,             # L1 regularization
    reg_lambda=1.0,            # L2 regularization
    random_state=42
)

print("✅ XGBoost model configured with improved hyperparameters")
```

### Step 6: Train the Model

**Find this section:**
```python
# Current training (BEFORE)
model.fit(X_train, y_train)
```

**Update to:**
```python
# ── Phase 28: TRAIN WITH NEW FEATURES ─────────────────────────
print("🔄 Training XGBoost with new features...")
print(f"   Training samples: {len(X_train)}")
print(f"   Features: {len(features)}")

model.fit(
    X_train, 
    y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=20,
    verbose=10
)

print("✅ Training complete!")
```

### Step 7: Evaluate the Model

**Find this section:**
```python
# Current evaluation (BEFORE)
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2%}")
```

**Update to:**
```python
# ── Phase 28: ENHANCED EVALUATION ─────────────────────────────
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print("✅ Model Performance:")
print(f"   Accuracy:  {accuracy:.2%}")
print(f"   Precision: {precision:.2%}")
print(f"   Recall:    {recall:.2%}")
print(f"   F1 Score:  {f1:.2%}")

# Feature importance
print("\n✅ Feature Importance:")
for feat, imp in sorted(zip(features, model.feature_importances_), key=lambda x: x[1], reverse=True):
    print(f"   {feat}: {imp:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\n✅ Confusion Matrix:")
print(f"   True Negatives:  {cm[0,0]}")
print(f"   False Positives: {cm[0,1]}")
print(f"   False Negatives: {cm[1,0]}")
print(f"   True Positives:  {cm[1,1]}")
```

### Step 8: Save the Model

**Find this section:**
```python
# Current saving (BEFORE)
import pickle
pickle.dump(model, open('brain1_xgboost_v4.pkl', 'wb'))
```

**Update to:**
```python
# ── Phase 28: SAVE IMPROVED MODEL ─────────────────────────────
import pickle
import json

# Save model
model_filename = 'brain1_xgboost_v4_phase28.pkl'
pickle.dump(model, open(model_filename, 'wb'))
print(f"✅ Model saved: {model_filename}")

# Save feature list
feature_config = {
    "features": features,
    "n_features": len(features),
    "version": "phase28",
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "trained_at": str(datetime.now())
}

config_filename = 'brain1_config_phase28.json'
with open(config_filename, 'w') as f:
    json.dump(feature_config, f, indent=2)
print(f"✅ Config saved: {config_filename}")

print("\n✅ PHASE 28 TRAINING COMPLETE!")
print(f"   Model: {model_filename}")
print(f"   Config: {config_filename}")
print(f"   Accuracy: {accuracy:.2%}")
```

### Step 9: Download the Files

1. In Colab, click the folder icon on the left
2. Right-click on `brain1_xgboost_v4_phase28.pkl`
3. Select "Download"
4. Right-click on `brain1_config_phase28.json`
5. Select "Download"

### Step 10: Upload to Railway

1. Go to your Railway dashboard: https://railway.app/project/455ad05e-75e3-4b65-9333-c1d10d3374b2
2. Click "Variables"
3. Update:
   - `MODEL_PKL_ID` = (new file ID from Google Drive)
   - `MODEL_JSON_ID` = (new config file ID from Google Drive)
4. Railway will auto-redeploy

## Comparing Old vs New Model

**To see the improvement:**

```python
# After training both models
print("Comparison:")
print(f"Old Model Accuracy: 58.3%")
print(f"New Model Accuracy: {accuracy:.1%}")
print(f"Improvement: {(accuracy - 0.583) * 100:.1f}%")
```

## Expected Improvements

With the new features, you should see:
- **+2-5% accuracy improvement** in signal generation
- **Better detection of volume spikes** (volume_ratio)
- **Volatility-aware predictions** (atr)
- **Overbought/oversold detection** (bb_position)
- **Time-of-day optimization** (hour)

## Troubleshooting

### Error: "KeyError: 'volume'"
- Make sure your data has a 'volume' column
- Check column names: `df.columns`

### Error: "NaN values in features"
- Add this after feature engineering:
```python
df = df.fillna(method='ffill').fillna(method='bfill')
```

### Model accuracy decreased
- This can happen with new features
- Try increasing `n_estimators` to 200
- Reduce `learning_rate` to 0.05

### Can't find Google Drive file ID
- Open the file in Google Drive
- Look at the URL: `https://drive.google.com/file/d/FILE_ID_HERE/view`
- Copy the FILE_ID_HERE part

## Next Steps

1. **Make these changes to your Colab notebook**
2. **Run all cells**
3. **Download the new model files**
4. **Upload to Railway**
5. **Test on demo account**
5. **Proceed to backtesting**

---

**Phase 28 XGBoost Retraining** ✅
