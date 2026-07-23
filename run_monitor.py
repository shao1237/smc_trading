import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from smc_trader.monitor import LiveMonitor
from smc_trader.config import SHIOAJI_API_KEY, SHIOAJI_SECRET_KEY
from smc_trader.logger import get_logger, log_separator

logger = get_logger()

def main():
    parser = argparse.ArgumentParser(description="SMC+SNR 台指期即時監控監測系統")
    parser.add_argument("--mode", type=str, choices=["mock", "shioaji"], default="mock",
                        help="監控模式: mock (模擬實時 Tick) 或 shioaji (真實行情訂閱)")
    parser.add_argument("--api-key", type=str, default=SHIOAJI_API_KEY,
                        help="Shioaji API Key")
    parser.add_argument("--secret-key", type=str, default=SHIOAJI_SECRET_KEY,
                        help="Shioaji Secret Key")
    
    args = parser.parse_args()

    logger.info(f"SMC 監控系統啟動 | mode={args.mode}")
    log_separator()

    # 自動判定是否具備帳密以 Fallback
    mode = args.mode
    if mode == "shioaji" and (not args.api_key or not args.secret_key):
        logger.warning("未設定 Shioaji API Key。將自動切換為 MOCK 即時監控模式！")
        mode = "mock"

    monitor = LiveMonitor(
        mode=mode,
        api_key=args.api_key,
        secret_key=args.secret_key
    )
    
    monitor.run()

if __name__ == "__main__":
    main()
