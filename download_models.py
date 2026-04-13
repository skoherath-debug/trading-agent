"""
Downloads model files from Google Drive on first startup.
Files are cached in /app/models/ so they only download once.
"""
import os, gdown, json

MODELS_DIR = "/app/models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Add your Google Drive file IDs here ──────────────────────────────────────
# To get file ID: right-click file in Drive → Share → Copy link
# Link looks like: https://drive.google.com/file/d/FILE_ID_HERE/view
# Copy just the FILE_ID_HERE part

DRIVE_FILES = {
    "brain1_xgboost_v4.pkl":      os.environ.get("MODEL_PKL_ID", ""),
    "brain1_features_v4.json":    os.environ.get("MODEL_JSON_ID", ""),
    "dynamic_tpsl_config.json":   os.environ.get("TPSL_JSON_ID", ""),
}

def download_all():
    all_ok = True
    for filename, file_id in DRIVE_FILES.items():
        dest = os.path.join(MODELS_DIR, filename)
        if os.path.exists(dest):
            print(f"✅ Already exists: {filename}")
            continue
        if not file_id:
            print(f"⚠️  No Drive ID set for {filename} — set env variable")
            all_ok = False
            continue
        try:
            print(f"Downloading {filename}...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, dest, quiet=False)
            print(f"✅ Downloaded: {filename}")
        except Exception as e:
            print(f"❌ Failed to download {filename}: {e}")
            all_ok = False
    return all_ok

if __name__ == "__main__":
    download_all()
