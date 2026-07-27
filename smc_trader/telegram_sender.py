import urllib.request
import urllib.parse
import json
from smc_trader.config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
from smc_trader.logger import get_logger

logger = get_logger()

def send_telegram_notification(text: str, chat_id: str = None) -> bool:
    """
    發送 Telegram 訊息至指定的 chat_id（若未傳入則預設使用 TELEGRAM_CHAT_ID）。
    """
    target_chat_id = chat_id if chat_id else TELEGRAM_CHAT_ID
    if not TELEGRAM_TOKEN or not target_chat_id:
        logger.warning("[Telegram] 未設定 TELEGRAM_TOKEN 或 chat_id，跳過發送。")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": target_chat_id,
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
            logger.info(f"[Telegram] 訊息發送成功 | chat_id={target_chat_id}")
            return True
    except Exception as e:
        logger.error(f"[Telegram] 發送通知失敗 (chat_id={target_chat_id}): {str(e)}")
        return False
