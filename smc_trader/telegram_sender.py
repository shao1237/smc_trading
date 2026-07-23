import urllib.request
import urllib.parse
import json
from smc_trader.config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
from smc_trader.logger import get_logger

logger = get_logger()

def send_telegram_notification(text: str) -> bool:
    """
    發送 Telegram 訊息。
    """
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("[Telegram] 未設定 TELEGRAM_TOKEN 或 TELEGRAM_CHAT_ID，跳過發送。")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML"
    }

    try:
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            url,
            data=data,
            headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            response.read()
            logger.info(f"[Telegram] 訊息發送成功 | chat_id={TELEGRAM_CHAT_ID}")
            return True
    except Exception as e:
        logger.error(f"[Telegram] 發送通知失敗: {str(e)}")
        return False
