import os
from pathlib import Path
from dotenv import load_dotenv

# 讀取 .env 檔案
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# --- Shioaji API 設定 ---
SHIOAJI_API_KEY = os.getenv("API_KEY", "")
SHIOAJI_SECRET_KEY = os.getenv("SECRET_KEY", "")
SHIOAJI_SIMULATION = os.getenv("SHIOAJI_SIMULATION", "true").lower() == "true"


# --- 交易商品設定 ---
# 台指期近月通常在 Shioaji 的合約代碼為 "TXF" 加上年月，如 "TXF202607"
# 或者是 "TXFR1" 代表近一月 (通常用於連續 K 線撈取，具體依 API 支援為準)
TARGET_SYMBOL = "TXF"
CONTRACT_CODE = "TXFR1" # 預設使用近一月連續月

# --- SMC 策略參數設定 ---
# 5M 大結構參數
SWING_WINDOW_5M = 5       # 5M Swing High/Low 的左右側比較窗口 (左右各 N 根)
# 1M 內部結構參數
SWING_WINDOW_1M = 3       # 1M Swing High/Low 的左右側比較窗口
VOLUME_MA_PERIOD = 20     # 成交量均線週期
VOLUME_MULT = 1.3         # 爆量判定倍數 (目前 K 棒成交量 / 均線成交量 >= VOLUME_MULT)

# --- 回測參數設定 ---
INITIAL_CAPITAL = 1000000.0  # 初始資金：1,000,000 NTD
FUTURES_POINT_VALUE = 200.0  # 台指期大台每點價值：200 NTD (小台為 50，此處預設為大台)
SLIPPAGE_POINTS = 2.0        # 每筆交易單邊滑價點數 (來回共 4 點)
COMMISSION_FEE = 50.0        # 每筆交易單邊手續費 NTD (來回共 100 元)
DEFAULT_RR = 3.1             # 預設風險報酬比 (Risk-Reward Ratio)
MAX_PENDING_BARS = 10        # 限價單觸發後，最多等待幾根 K 棒未成交則取消

# --- Telegram 設定 ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# --- 資料庫/快取設定 ---
DATA_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_cache")
os.makedirs(DATA_CACHE_DIR, exist_ok=True)
