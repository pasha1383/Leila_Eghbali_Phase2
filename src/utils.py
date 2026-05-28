"""
utils.py — توابع کمکی مشترک برای همه مدل‌ها

این ماژول شامل:
  - بارگذاری داده پردازش‌شده
  - تقسیم train/test
  - ساخت feature/window برای مدل‌های مختلف
  - معیارهای ارزیابی
  - توابع رسم نمودار
  - ذخیره نتایج
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============================================================
# مسیرها
# ============================================================
ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT_DIR, "data", "processed")
FIG_DIR    = os.path.join(ROOT_DIR, "results", "figures")
MET_DIR    = os.path.join(ROOT_DIR, "results", "metrics")
MODEL_DIR  = os.path.join(ROOT_DIR, "models")

# ============================================================
# بارگذاری داده
# ============================================================

def load_data() -> pd.DataFrame:
    """
    داده پردازش‌شده را از data/processed/master_data.csv می‌خواند.
    ایندکس را به PeriodIndex تبدیل می‌کند.
    """
    path = os.path.join(DATA_DIR, "master_data.csv")
    df = pd.read_csv(path, index_col="period")
    df.index = pd.PeriodIndex(df.index, freq="Q")
    df.sort_index(inplace=True)
    return df


def get_train_test(df: pd.DataFrame, test_quarters: int = 16):
    """
    داده را به صورت زمانی (chronological) به train و test تقسیم می‌کند.
    
    Parameters
    ----------
    df : DataFrame با PeriodIndex
    test_quarters : تعداد فصل‌های آزمون (پیش‌فرض: ۱۶ فصل = ۴ سال)
    
    Returns
    -------
    train, test : دو DataFrame
    """
    train = df.iloc[:-test_quarters].copy()
    test  = df.iloc[-test_quarters:].copy()
    return train, test


# ============================================================
# ساخت feature matrix
# ============================================================

FEATURE_COLS = [
    "oil", "gold", "export", "import", "exchange", "liquidity", "cpi",
    "cycle",
    "oil_x_cycle", "gold_x_cycle", "export_x_cycle", "import_x_cycle",
    "exchange_x_cycle", "liquidity_x_cycle", "cpi_x_cycle"
]

# برای مدل‌های tree-based (XGBoost / RF) هدف = log_return
TARGET_RETURN  = "log_return"

# برای مدل‌های سری زمانی و GRU هدف = log_stock_price
TARGET_LEVEL   = "log_stock_price"


def make_supervised_features(df: pd.DataFrame,
                              target: str = TARGET_RETURN,
                              lag: int = 1,
                              add_lags: bool = True):
    """
    Feature matrix برای مدل‌های غیر-sequential (XGBoost, RF, ...).
    
    Parameters
    ----------
    df        : کل DataFrame
    target    : نام ستون هدف
    lag       : تعداد lag برای feature های lag‌دار
    add_lags  : آیا lag های هدف به feature ها اضافه شود؟
    
    Returns
    -------
    X, y, feature_names
    """
    tmp = df[FEATURE_COLS + [target]].copy()
    
    if add_lags:
        for l in range(1, lag + 1):
            tmp[f"{target}_lag{l}"] = tmp[target].shift(l)
    
    tmp.dropna(inplace=True)
    feature_names = [c for c in tmp.columns if c != target]
    X = tmp[feature_names].values
    y = tmp[target].values
    return X, y, feature_names


def make_sequences(series: np.ndarray, window: int = 8):
    """
    داده سری زمانی را به دنباله‌های [X, y] برای GRU/LSTM تبدیل می‌کند.
    
    Parameters
    ----------
    series : آرایه ۱ بعدی (مثلاً log_stock_price)
    window : اندازه پنجره ورودی (فصل)
    
    Returns
    -------
    X : (n_samples, window, 1)
    y : (n_samples,)
    """
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i : i + window])
        y.append(series[i + window])
    return np.array(X)[..., np.newaxis], np.array(y)


# ============================================================
# معیارهای ارزیابی
# ============================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    model_name: str = "model") -> dict:
    """
    RMSE, MAE, MAPE, R² را روی مقیاس اصلی (پس از inverse transform) محاسبه می‌کند.
    """
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    mae   = mean_absolute_error(y_true, y_pred)
    mape  = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    r2    = r2_score(y_true, y_pred)
    
    metrics = {
        "model": model_name,
        "RMSE":  round(rmse, 4),
        "MAE":   round(mae, 4),
        "MAPE":  round(mape, 4),
        "R2":    round(r2, 4)
    }
    return metrics


def save_metrics(metrics: dict, filename: str = "metrics.csv"):
    """نتایج ارزیابی را در results/metrics/ ذخیره می‌کند."""
    os.makedirs(MET_DIR, exist_ok=True)
    path = os.path.join(MET_DIR, filename)
    df = pd.DataFrame([metrics])
    
    # اگر فایل وجود دارد، ردیف جدید اضافه کن
    if os.path.exists(path):
        existing = pd.read_csv(path)
        # جلوگیری از تکرار
        existing = existing[existing["model"] != metrics["model"]]
        df = pd.concat([existing, df], ignore_index=True)
    
    df.to_csv(path, index=False)
    print(f"✓ Metrics saved: {path}")


# ============================================================
# inverse transform
# ============================================================

def reconstruct_from_log_return(log_returns_pred: np.ndarray,
                                 last_known_log_price: float) -> np.ndarray:
    """
    از log_return های پیش‌بینی‌شده، سطح قیمت واقعی را بازسازی می‌کند.
    
    فرمول: log_price[t] = log_price[t-1] + log_return[t]
    سپس: price[t] = exp(log_price[t])
    """
    log_prices = np.zeros(len(log_returns_pred))
    log_prices[0] = last_known_log_price + log_returns_pred[0]
    for i in range(1, len(log_returns_pred)):
        log_prices[i] = log_prices[i - 1] + log_returns_pred[i]
    return np.exp(log_prices)


# ============================================================
# رسم نمودارها
# ============================================================

def plot_predictions(dates, y_true_price: np.ndarray, y_pred_price: np.ndarray,
                     model_name: str, save: bool = True):
    """
    نمودار مقایسه مقادیر واقعی vs پیش‌بینی‌شده (در مقیاس اصلی قیمت).
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f"مدل {model_name} — مقایسه پیش‌بینی و واقعیت", fontsize=14, y=1.01)
    
    # نمودار بالا: قیمت واقعی vs پیش‌بینی
    ax1 = axes[0]
    ax1.plot(dates, y_true_price, label="واقعی", color="#1f77b4", linewidth=2)
    ax1.plot(dates, y_pred_price, label="پیش‌بینی", color="#ff7f0e",
             linewidth=2, linestyle="--")
    ax1.set_title("شاخص قیمت: واقعی در مقابل پیش‌بینی")
    ax1.set_ylabel("شاخص بورس (ریال)")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # نمودار پایین: خطای پیش‌بینی
    ax2 = axes[1]
    errors = y_true_price - y_pred_price
    ax2.bar(range(len(errors)), errors, color=["#d62728" if e < 0 else "#2ca02c"
                                                for e in errors], alpha=0.7)
    ax2.axhline(0, color="black", linewidth=1)
    ax2.set_title("خطای پیش‌بینی (واقعی - پیش‌بینی)")
    ax2.set_ylabel("خطا")
    ax2.set_xlabel("فصل آزمون")
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        os.makedirs(FIG_DIR, exist_ok=True)
        path = os.path.join(FIG_DIR, f"{model_name}_predictions.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"✓ Figure saved: {path}")
    
    plt.show()


def plot_all_models_comparison(results_dict: dict, dates, save: bool = True):
    """
    نمودار مقایسه همه مدل‌ها روی یک محور.
    results_dict = {"XGBoost": (y_true, y_pred), "SARIMAX": ..., ...}
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    
    y_true = list(results_dict.values())[0][0]
    ax.plot(dates, y_true, label="واقعی", color="black",
            linewidth=2.5, zorder=5)
    
    for i, (name, (_, y_pred)) in enumerate(results_dict.items()):
        ax.plot(dates, y_pred, label=name, color=colors[i],
                linewidth=1.8, linestyle="--", alpha=0.85)
    
    ax.set_title("مقایسه همه مدل‌ها — داده آزمون", fontsize=14)
    ax.set_ylabel("شاخص بورس (ریال)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    if save:
        os.makedirs(FIG_DIR, exist_ok=True)
        path = os.path.join(FIG_DIR, "all_models_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"✓ Figure saved: {path}")
    
    plt.show()
