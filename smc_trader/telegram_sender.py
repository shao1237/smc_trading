import urllib.request
import urllib.parse
import json
from smc_trader.config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

def send_telegram_notification(text: str) -> bool:
    """
    發送 Telegram 訊息。
    """
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        # 未配置則跳過
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
        with urllib.request.urlopen(req, timeout=5) as response:
            response.read()
            return True
    except Exception as e:
        print(f"[Telegram] 發送通知失敗: {str(e)}")
        return False
