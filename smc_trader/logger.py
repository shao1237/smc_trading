"""
SMC Trading 統一日誌模組。

- console 輸出：即時觀察執行狀態（含 ANSI 色碼）
- 檔案輸出：logs/monitor_YYYYMMDD.log（純文字，不含 ANSI，可長期追溯）
- Telegram 發送結果、錯誤、訊號觸發全部記錄

使用方式:
    from smc_trader.logger import get_logger
    logger = get_logger()
    logger.info("啟動監控")
    logger.signal("★★ [結構轉換] MSS Bullish 確立！")
    logger.telegram_ok("訊號已發送至 TG 群組")
    logger.telegram_fail("發送失敗: chat not found")
"""

import os
import sys
import logging
import datetime
from pathlib import Path

# ANSI 顏色（僅 console）
C_GREEN = "\033[92m"
C_RED = "\033[91m"
C_YELLOW = "\033[93m"
C_BLUE = "\033[94m"
C_CYAN = "\033[96m"
C_BOLD = "\033[1m"
C_RESET = "\033[0m"

# --- 日誌目錄 ---
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# --- 自訂 Log Level: SIGNAL (25, 介於 INFO 和 WARNING) ---
SIGNAL_LEVEL = 25
logging.addLevelName(SIGNAL_LEVEL, "SIGNAL")


def _signal(self, message, *args, **kwargs):
    if self.isEnabledFor(SIGNAL_LEVEL):
        self._log(SIGNAL_LEVEL, message, args, **kwargs)


logging.Logger.signal = _signal


class AnsiStripFormatter(logging.Formatter):
    """移除 ANSI 色碼，用於檔案輸出"""
    import re
    _ansi_re = re.compile(r'\033\[[0-9;]*m')

    def format(self, record):
        record.msg = self._ansi_re.sub('', str(record.msg))
        return super().format(record)


class ColorConsoleFormatter(logging.Formatter):
    """保留 ANSI 色碼，用於 console 輸出"""

    def format(self, record):
        return super().format(record)


# 全域 logger 實例（避免重複建立 handler）
_logger = None


def get_logger(name: str = "smc_monitor") -> logging.Logger:
    global _logger
    if _logger is not None:
        return _logger

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    today = datetime.datetime.now().strftime("%Y%m%d")
    log_file = LOG_DIR / f"monitor_{today}.log"

    # --- 檔案 handler (純文字，含完整資訊) ---
    file_fmt = AnsiStripFormatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh = logging.FileHandler(log_file, encoding="utf-8", mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(file_fmt)
    logger.addHandler(fh)

    # --- Console handler (含色碼，精簡格式) ---
    # Telegram 相關訊息只寫檔案、不顯示在 console
    class _NoTGFilter(logging.Filter):
        """過濾掉 TG 發送相關訊息，不顯示在 console"""
        def filter(self, record):
            msg = str(record.msg)
            # 訊號觸發發送 TG + Telegram 發送結果 → 只進檔案
            if "發送 Telegram 通知" in msg:
                return False
            if msg.startswith("[Telegram]"):
                return False
            return True

    console_fmt = ColorConsoleFormatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S"
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.addFilter(_NoTGFilter())
    ch.setFormatter(console_fmt)
    logger.addHandler(ch)

    # 啟動分隔線
    logger.info(f"{'=' * 60}")
    logger.info(f"SMC Monitor Logger 啟動 | log file: {log_file}")
    logger.info(f"{'=' * 60}")

    _logger = logger
    return logger


def log_separator(logger=None, char="=", length=60):
    """輸出分隔線"""
    if logger is None:
        logger = get_logger()
    logger.info(char * length)
