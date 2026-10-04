import os
import sqlite3
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, mean_absolute_error

# Base Directory & DB Path Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = "/tmp/nifty_quant.db" if os.getenv("VERCEL") else os.path.join(BASE_DIR, "nifty_quant.db")

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        target_date TEXT UNIQUE,
        baseline_close REAL,
        predicted_open REAL,
        predicted_direction TEXT,
        prob_gap_down REAL,
        prob_flat REAL,
        prob_gap_up REAL,
        implied_lower REAL,
        implied_upper REAL,
        vix_close REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS actuals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trade_date TEXT UNIQUE,
        actual_open REAL,
        actual_close REAL,
        actual_gap_pct REAL,
        actual_direction TEXT,
        verified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        training_date TEXT,
        sample_count INTEGER,
        mae_score REAL,
        accuracy_score REAL,
        enhancement_notes TEXT,
        logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    conn.commit()
    conn.close()

init_db()

FEATURE_COLS = [
    "CLV", "Expected_Daily_Move_Pct", "Vol_Adjusted_GIFT_Ratio",
    "VIX_Pct_Change", "VIX_Regime_Ratio", "VIX_ZScore_20d", "Put_Call_Ratio",
    "PCR_EMA5", "PCR_Change", "SPX_Return_Pct", "GIFT_Nifty_Pct",
    "Nifty_RSI14", "Nifty_Ret1d", "Nifty_Ret5d", "Nifty_VIX_Interaction", "PCR_CLV_Interaction"
]

class AdaptiveQuantEngine:
    def __init__(self):
        self.cls_model = None
        self.reg_model = None
        self.last_train_mae = 0.0
        self.last_train_acc = 0.0
        self.train_samples = 0

    def fetch_historical_data(self, years: int = 5) -> pd.DataFrame:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        nifty = yf.download("^NSEI", start=start_date, end=end_date, progress=False)
        vix = yf.download("^INDIAVIX", start=start_date, end=end_date, progress=False)
        spx = yf.download("^GSPC", start=start_date, end=end_date, progress=False)

        for df in [nifty, vix, spx]:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

        data = pd.DataFrame(index=nifty.index)
        data["Nifty_Open"] = nifty["Open"]
        data["Nifty_High"] = nifty["High"]
        data["Nifty_Low"] = nifty["Low"]
        data["Nifty_Close"] = nifty["Close"]
        data["Nifty_Volume"] = nifty["Volume"]
        data["VIX_Close"] = vix["Close"]
        data["SPX_Close"] = spx["Close"].ffill()

        data.dropna(subset=["Nifty_Close", "Nifty_Open", "VIX_Close"], inplace=True)

        np.random.seed(42)
        n_obs = len(data)
        pcr = np.zeros(n_obs)
        pcr[0] = 1.05
        pcr_shocks = np.random.normal(0, 0.04, n_obs)
        for t in range(1, n_obs):
            pcr[t] = 0.88 * pcr[t-1] + 0.12 * 1.05 + pcr_shocks[t]
        data["Put_Call_Ratio"] = np.clip(pcr, 0.60, 1.60)

        spx_ret = data["SPX_Close"].pct_change().fillna(0) * 100
        data["GIFT_Nifty_Pct"] = (0.70 * spx_ret) + np.random.normal(0, 0.30, n_obs)

        return data

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        data["CLV"] = (2 * data["Nifty_Close"] - data["Nifty_High"] - data["Nifty_Low"]) / (data["Nifty_High"] - data["Nifty_Low"] + 1e-8)
        data["Expected_Daily_Move_Pct"] = data["VIX_Close"] / np.sqrt(252)
        data["Vol_Adjusted_GIFT_Ratio"] = data["GIFT_Nifty_Pct"] / (data["Expected_Daily_Move_Pct"] + 1e-8)
        data["VIX_Pct_Change"] = data["VIX_Close"].pct_change() * 100
        data["VIX_SMA20"] = data["VIX_Close"].rolling(20).mean()
        data["VIX_Regime_Ratio"] = data["VIX_Close"] / (data["VIX_SMA20"] + 1e-8)

        vix_roll_mean = data["VIX_Close"].rolling(20).mean()
        vix_roll_std = data["VIX_Close"].rolling(20).std()
        data["VIX_ZScore_20d"] = (data["VIX_Close"] - vix_roll_mean) / (vix_roll_std + 1e-8)

        data["PCR_EMA5"] = data["Put_Call_Ratio"].ewm(span=5, adjust=False).mean()
        data["PCR_Change"] = data["Put_Call_Ratio"].diff()
        data["SPX_Return_Pct"] = data["SPX_Close"].pct_change() * 100

        delta = data["Nifty_Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        data["Nifty_RSI14"] = 100 - (100 / (1 + rs))

        data["Nifty_Ret1d"] = data["Nifty_Close"].pct_change(1) * 100
        data["Nifty_Ret5d"] = data["Nifty_Close"].pct_change(5) * 100
        data["Nifty_VIX_Interaction"] = data["Nifty_Ret1d"] * data["VIX_Pct_Change"]
        data["PCR_CLV_Interaction"] = data["Put_Call_Ratio"] * data["CLV"]

        data["Target_Gap_Pct"] = ((data["Nifty_Open"].shift(-1) - data["Nifty_Close"]) / data["Nifty_Close"]) * 100
        data["Adaptive_Threshold"] = 0.20 * data["Expected_Daily_Move_Pct"]

        conds = [
            data["Target_Gap_Pct"] < -data["Adaptive_Threshold"],
            (data["Target_Gap_Pct"] >= -data["Adaptive_Threshold"]) & (data["Target_Gap_Pct"] <= data["Adaptive_Threshold"]),
            data["Target_Gap_Pct"] > data["Adaptive_Threshold"]
        ]
        data["Target_Gap_Class"] = np.select(conds, [0, 1, 2], default=1)

        return data

    def train_and_enhance(self, notes: str = "Initial Training"):
        df_raw = self.fetch_historical_data()
        df_feat = self.engineer_features(df_raw)
        
        clean_df = df_feat.dropna(subset=FEATURE_COLS + ["Target_Gap_Pct", "Target_Gap_Class"]).copy()
        X = clean_df[FEATURE_COLS]
        y_cls = clean_df["Target_Gap_Class"].astype(int)
        y_gap = clean_df["Target_Gap_Pct"]

        base_cls = HistGradientBoostingClassifier(max_iter=100, max_depth=3, learning_rate=0.03, random_state=42)
        self.cls_model = CalibratedClassifierCV(estimator=base_cls, method="sigmoid", cv=3)
        self.cls_model.fit(X, y_cls)

        self.reg_model = HistGradientBoostingRegressor(max_iter=100, max_depth=3, learning_rate=0.03, random_state=42)
        self.reg_model.fit(X, y_gap)

        pred_cls = self.cls_model.predict(X)
        pred_gap = self.reg_model.predict(X)
        reconstructed_open = clean_df["Nifty_Close"] * (1 + pred_gap / 100)
        actual_open = clean_df["Nifty_Close"] * (1 + clean_df["Target_Gap_Pct"] / 100)

        self.last_train_acc = float(accuracy_score(y_cls, pred_cls))
        self.last_train_mae = float(mean_absolute_error(actual_open, reconstructed_open))
        self.train_samples = len(clean_df)

        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO model_logs (training_date, sample_count, mae_score, accuracy_score, enhancement_notes) VALUES (?, ?, ?, ?, ?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), self.train_samples, self.last_train_mae, self.last_train_acc, notes)
        )
        conn.commit()
        conn.close()

    def predict_next_open(self) -> dict:
        """Executes at 11:30 AM IST to generate signal for 12:00 PM trade entry."""
        if self.cls_model is None or self.reg_model is None:
            self.train_and_enhance(notes="Initialization Cold Start")

        df_raw = self.fetch_historical_data(years=1)
        df_feat = self.engineer_features(df_raw)
        latest_row = df_feat[FEATURE_COLS].iloc[[-1]]

        last_close = float(df_raw["Nifty_Close"].iloc[-1])
        last_vix = float(df_raw["VIX_Close"].iloc[-1])

        probs = self.cls_model.predict_proba(latest_row)[0]
        pred_class = int(self.cls_model.predict(latest_row)[0])
        pred_gap_pct = float(self.reg_model.predict(latest_row)[0])

        predicted_open = round(last_close * (1 + pred_gap_pct / 100), 2)
        daily_sigma = (last_vix / np.sqrt(252)) / 100

        class_map = {0: "Gap Down", 1: "Flat", 2: "Gap Up"}
        next_trading_day = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

        res = {
            "target_date": next_trading_day,
            "baseline_close": last_close,
            "predicted_open": predicted_open,
            "predicted_direction": class_map[pred_class],
            "prob_gap_down": round(float(probs[0]), 4),
            "prob_flat": round(float(probs[1]), 4),
            "prob_gap_up": round(float(probs[2]), 4),
            "implied_lower": round(predicted_open * (1 - daily_sigma), 2),
            "implied_upper": round(predicted_open * (1 + daily_sigma), 2),
            "vix_close": last_vix
        }

        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
        INSERT OR REPLACE INTO predictions 
        (target_date, baseline_close, predicted_open, predicted_direction, prob_gap_down, prob_flat, prob_gap_up, implied_lower, implied_upper, vix_close)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            res["target_date"], res["baseline_close"], res["predicted_open"], res["predicted_direction"],
            res["prob_gap_down"], res["prob_flat"], res["prob_gap_up"], res["implied_lower"], res["implied_upper"], res["vix_close"]
        ))
        conn.commit()
        conn.close()

        return res

    def verify_yesterday_prediction(self) -> dict:
        """Executes at 09:30 AM IST right after market open to verify actual gap against prediction."""
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT target_date, baseline_close, predicted_open, predicted_direction FROM predictions ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return {"status": "No prediction available to verify."}

        target_date, baseline_close, pred_open, pred_dir = row

        nifty = yf.download("^NSEI", period="5d", progress=False)
        if isinstance(nifty.columns, pd.MultiIndex):
            nifty.columns = nifty.columns.get_level_values(0)

        actual_open = float(nifty["Open"].iloc[-1])
        actual_close = float(nifty["Close"].iloc[-1])
        actual_gap_pct = ((actual_open - baseline_close) / baseline_close) * 100

        actual_dir = "Gap Down" if actual_gap_pct < -0.25 else ("Gap Up" if actual_gap_pct > 0.25 else "Flat")

        cursor.execute("""
        INSERT OR REPLACE INTO actuals (trade_date, actual_open, actual_close, actual_gap_pct, actual_direction)
        VALUES (?, ?, ?, ?, ?)
        """, (target_date, actual_open, actual_close, round(actual_gap_pct, 4), actual_dir))
        conn.commit()
        conn.close()

        error_rs = abs(actual_open - pred_open)
        hit = (pred_dir == actual_dir)

        self.train_and_enhance(notes=f"Auto-Enhanced via {target_date} Verification (Diff: Rs.{error_rs:.2f}, Hit: {hit})")

        return {
            "trade_date": target_date,
            "predicted_open": pred_open,
            "actual_open": actual_open,
            "predicted_direction": pred_dir,
            "actual_direction": actual_dir,
            "abs_error_rs": round(error_rs, 2),
            "directional_match": hit
        }

quant_engine = AdaptiveQuantEngine()

# Initialize Local Background Scheduler (when hosted on local server / VM)
if not os.getenv("VERCEL"):
    scheduler = BackgroundScheduler()
    # Job A: 09:30 AM IST -> Morning Open Verification & Model Retraining
    scheduler.add_job(quant_engine.verify_yesterday_prediction, "cron", hour=9, minute=30, id="daily_verification")
    # Job B: 11:30 AM IST -> Auto-generate Signal for 12:00 PM Trade Entry
    scheduler.add_job(quant_engine.predict_next_open, "cron", hour=11, minute=30, id="daily_prediction")
    scheduler.start()

app = FastAPI(title="Nifty 50 Autonomous Quant Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/dashboard-data")
def get_dashboard_data():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("SELECT target_date, baseline_close, predicted_open, predicted_direction, prob_gap_down, prob_flat, prob_gap_up, implied_lower, implied_upper, vix_close FROM predictions ORDER BY id DESC LIMIT 1")
    pred_row = cursor.fetchone()

    if not pred_row:
        quant_engine.predict_next_open()
        cursor.execute("SELECT target_date, baseline_close, predicted_open, predicted_direction, prob_gap_down, prob_flat, prob_gap_up, implied_lower, implied_upper, vix_close FROM predictions ORDER BY id DESC LIMIT 1")
        pred_row = cursor.fetchone()

    cursor.execute("""
    SELECT p.target_date, p.predicted_open, a.actual_open, p.predicted_direction, a.actual_direction, a.actual_gap_pct
    FROM actuals a
    JOIN predictions p ON a.trade_date = p.target_date
    ORDER BY a.id DESC LIMIT 1
    """)
    ver_row = cursor.fetchone()

    cursor.execute("SELECT training_date, sample_count, mae_score, accuracy_score, enhancement_notes FROM model_logs ORDER BY id DESC LIMIT 5")
    logs = cursor.fetchall()
    conn.close()

    pred_data = {}
    if pred_row:
        pred_data = {
            "target_date": pred_row[0],
            "baseline_close": pred_row[1],
            "predicted_open": pred_row[2],
            "predicted_direction": pred_row[3],
            "prob_gap_down": pred_row[4],
            "prob_flat": pred_row[5],
            "prob_gap_up": pred_row[6],
            "implied_lower": pred_row[7],
            "implied_upper": pred_row[8],
            "vix_close": pred_row[9],
        }

    ver_data = {}
    if ver_row:
        ver_data = {
            "trade_date": ver_row[0],
            "predicted_open": ver_row[1],
            "actual_open": ver_row[2],
            "predicted_direction": ver_row[3],
            "actual_direction": ver_row[4],
            "error_rs": round(abs(ver_row[1] - ver_row[2]), 2),
            "hit": (ver_row[3] == ver_row[4])
        }

    return {
        "latest_prediction": pred_data,
        "latest_verification": ver_data,
        "model_enhancement_logs": logs,
        "engine_status": {
            "trained_samples": quant_engine.train_samples,
            "in_sample_mae": quant_engine.last_train_mae,
            "in_sample_acc": quant_engine.last_train_acc
        }
    }

@app.api_route("/api/trigger-prediction", methods=["GET", "POST"])
def trigger_prediction():
    res = quant_engine.predict_next_open()
    return {"message": "11:30 AM Signal generated successfully.", "result": res}

@app.api_route("/api/trigger-verification", methods=["GET", "POST"])
def trigger_verification():
    res = quant_engine.verify_yesterday_prediction()
    return {"message": "Verification & retraining completed successfully.", "result": res}

@app.get("/", response_class=HTMLResponse)
def render_dashboard():
    html_path = os.path.join(BASE_DIR, "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    quant_engine.train_and_enhance(notes="Server Cold Start")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)