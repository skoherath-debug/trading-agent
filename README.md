# XAU/USD Triple Brain Agent — Railway Deployment

## Setup Steps

### 1. Get Google Drive File IDs
For each file in your Drive trading_agent folder:
- Right click the file → Share → Copy link
- Link: `https://drive.google.com/file/d/FILE_ID/view`
- Copy the FILE_ID part

Files needed:
- brain1_xgboost_v4.pkl
- brain1_features_v4.json
- dynamic_tpsl_config.json

### 2. Set Environment Variables in Railway
Go to Railway → Your Project → Variables → Add:

```
MODEL_PKL_ID      = <file id of brain1_xgboost_v4.pkl>
MODEL_JSON_ID     = <file id of brain1_features_v4.json>
TPSL_JSON_ID      = <file id of dynamic_tpsl_config.json>
TELEGRAM_TOKEN    = 8172828888:AAFWCvtCl1F-Kj5yOv_EFEB9vxL-ir-dD9I
TELEGRAM_CHAT     = 7132630179
GROQ_API_KEY      = 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
RUN_INTERVAL      = 3600
```

### 3. Add trading_agent.py
Copy your trading_agent.py into this folder before deploying.

### 4. Deploy
Push to GitHub → Railway auto-deploys.

### 5. Update dashboard.html
Change API_BASE to your Railway URL:
```js
const API_BASE = "https://your-app.railway.app";
```
