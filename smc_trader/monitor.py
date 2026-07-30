import os
import sys
import time
import math
import datetime
import numpy as np
import pandas as pd
import shioaji as sj
from typing import List, Dict, Any, Optional
from smc_trader.config import (
    SHIOAJI_API_KEY, SHIOAJI_SECRET_KEY, SHIOAJI_SIMULATION,
    SWING_WINDOW_5M, SWING_WINDOW_1M, VOLUME_MA_PERIOD, VOLUME_MULT,
    DEFAULT_RR, MAX_SL_POINTS, TELEGRAM_SIGNAL_CHAT_ID, TELEGRAM_SETTLEMENT_CHAT_ID,
    SLIPPAGE_POINTS, COMMISSION_FEE
)
from smc_trader.smc_detector import SMCDetector
from smc_trader.telegram_sender import send_telegram_notification
from smc_trader.logger import get_logger

# ANSI 顏色設定
C_GREEN = "\033[92m"
C_RED = "\033[91m"
C_YELLOW = "\033[93m"
C_BLUE = "\033[94m"
C_CYAN = "\033[96m"
C_BOLD = "\033[1m"
C_RESET = "\033[0m"

# [FIX] 模擬持倉損益計算常數。
# config.py 的 FUTURES_POINT_VALUE(200.0)是給「大台」用的，
# 但本檔案下單/log 訊息用的是「小台 (MTX)」(見 _place_simulated_2stage_order 的 log 文字)，
# 小台每口每點 50 NTD，故獨立定義，不直接沿用 FUTURES_POINT_VALUE。
MINI_POINT_VALUE = 50.0

# config.py 的 SLIPPAGE_POINTS / COMMISSION_FEE 註解皆為「單邊」，來回各要乘以 2。
# 原本 monitor.py 的舊公式用的是「-2.0 點」與「-100 元」，
# 那其實只等於單邊滑價(2點)、和單口單邊成本的 2 倍(=來回100元/口)，
# 且完全沒有依口數縮放 —— 也就是說，就算原本標榜的是「2 口」，
# PnL 數字實際上一直只是照 1 口去算的。這裡統一改成正確依口數縮放。
ROUNDTRIP_SLIPPAGE_PTS = SLIPPAGE_POINTS * 2       # 來回滑價點數
ROUNDTRIP_COMMISSION_PER_LOT = COMMISSION_FEE * 2  # 每口來回手續費 (NTD)

logger = get_logger()

class LiveMonitor:
    """
    SMC+SNR 台指期即時監控器。
    動態接收 (或模擬) 報價，合建成 1M 與 5M K 線，並即時識別 SMC 指標與信號。
    """
    def __init__(self, mode: str = "mock", api_key: str = "", secret_key: str = ""):
        self.mode = mode
        self.api_key = api_key
        self.secret_key = secret_key
        self.detector = SMCDetector(
            swing_window_5m=SWING_WINDOW_5M,
            swing_window_1m=SWING_WINDOW_1M,
            volume_ma_period=VOLUME_MA_PERIOD,
            volume_mult=VOLUME_MULT,
            pullback_buffer_pts=20.0
        )
        
        # 歷史 1M K 線數據庫
        self.history_1m: List[Dict[str, Any]] = []

        # 當前進行中的 1M K 線
        # [FIX] 此屬性曾在 commit 86fb853 被誤刪，導致 on_tick 首次觸發時
        # 直接丟出 AttributeError（且該例外會被 Shioaji 背景執行緒吃掉，
        # 表面上完全看不出來，只會呈現「訂閱成功後就不再有任何 tick 更新」）。
        self.current_bar_1m: Optional[Dict[str, Any]] = None

        # 當前模擬/實盤持倉追蹤 (二階段停利與保本管理)
        self.active_position: Optional[Dict[str, Any]] = None

        # 最近一次收到 tick 的時間，供健康檢查（多久沒收到報價）使用
        self.last_tick_time: Optional[datetime.datetime] = None

        # Shioaji API / 合約物件；於 _run_shioaji() 中實際賦值，
        # 供 _place_simulated_2stage_order() 送出模擬單使用
        self.api: Optional[Any] = None
        self.contract: Optional[Any] = None

        # 預先載入一部分歷史數據以讓 Swing Points 能夠在初始時就能被計算
        self._init_history()

    def _init_history(self):
        """生成或加載初始歷史數據，避免一開始無 Swing 點"""
        logger.info("正在載入初始歷史數據，建立結構基礎...")
        
        # 1. 嘗試從快取數據加載真實數據 (僅在 shioaji 模式或快取存在時)
        from smc_trader.config import DATA_CACHE_DIR
        import glob
        
        cache_files = glob.glob(os.path.join(DATA_CACHE_DIR, "shioaji_TXFR1_*.csv"))
        if cache_files:
            # 找到最新的快取檔案
            latest_cache = max(cache_files, key=os.path.getmtime)
            try:
                df_cache = pd.read_csv(latest_cache)
                df_cache['ts'] = pd.to_datetime(df_cache['ts'])
                # 只取得最後 1000 根 1M K 線作為初始歷史數據
                df_init = df_cache.tail(1000).copy()
                for _, r in df_init.iterrows():
                    self.history_1m.append({
                        'ts': r['ts'],
                        'open': float(r['open']),
                        'high': float(r['high']),
                        'low': float(r['low']),
                        'close': float(r['close']),
                        'volume': int(r['volume'])
                    })
                logger.info(f"成功自本地快取檔案 {os.path.basename(latest_cache)} 載入 {len(self.history_1m)} 根真實 1M K 線歷史數據！")
                return
            except Exception as e:
                logger.warning(f"嘗試自本地快取加載數據時失敗: {str(e)}，將改採隨機數據生成...")
        
        # 2. 隨機生成數據 (Fallback 或者是 mock 模式)
        np.random.seed(42)
        base_price = 20000.0
        now = datetime.datetime.now()
        
        # 預先生成 400 根 1M K 線 (約 80 根 5M K 線，足夠建立穩固的 Swing 結構)
        for i in range(400):
            ts = now - datetime.timedelta(minutes=400 - i)
            # 隨機生成 K 線
            o = base_price + np.random.normal(0, 5.0)
            c = o + np.random.normal(0, 5.0)
            h = max(o, c) + max(0, np.random.exponential(2.0))
            l = min(o, c) - max(0, np.random.exponential(2.0))
            vol = np.random.randint(200, 1000)
            
            # 偶爾注入一些波動，人為製造 Swing 突破點
            if i % 80 == 40: 
                l -= 30.0
            if i % 80 == 70: 
                h += 30.0
                
            self.history_1m.append({
                'ts': ts,
                'open': round(o, 1),
                'high': round(h, 1),
                'low': round(l, 1),
                'close': round(c, 1),
                'volume': vol
            })
            base_price = c
        logger.info(f"初始模擬歷史數據載入完畢，共 {len(self.history_1m)} 根 K 線。")

    def run(self):
        """啟動監控"""
        if self.mode == "shioaji":
            try:
                self._run_shioaji()
            except Exception as e:
                logger.error(f"Shioaji 真實監控啟動失敗: {str(e)}")
                logger.warning(f"將自動轉為模擬即時監控模式運行。")
                self.mode = "mock"
                
        if self.mode == "mock":
            self._run_mock()

    def _run_shioaji(self):
        """使用 Shioaji API 真實訂閱台指期監控"""
        if not self.api_key or not self.secret_key:
            raise ValueError("Shioaji 登入資訊不足，請設定 api_key 與 secret_key")

        api = sj.Shioaji(simulation=SHIOAJI_SIMULATION)
        logger.info(f"{C_CYAN}正在登入 Shioaji API 進行實時監控...{C_RESET}")
        api.login(api_key=self.api_key, secret_key=self.secret_key)
        logger.info(f"{C_GREEN}登入成功！正在取得台指期近月合約...{C_RESET}")
        
        # 註冊委託/成交回報監聽
        api.set_order_callback(self._on_order_callback)
        
        # 尋找當前近月合約
        futures = api.Contracts.Futures.TXF
        try:
            contract = futures["TXFR1"]
        except KeyError:
            contract = getattr(futures, "TXFR1", None)
            
        if contract is None:
            raise ValueError("找不到 TXFR1 期貨合約")

        # [FIX] 原本 api / contract 僅為區域變數，_place_simulated_2stage_order()
        # 內部的 hasattr(self, 'api') 永遠是 False，導致「自動模擬下單」功能
        # 從 commit c78a8b7 導入以來從未真正執行過，只印了 log、發了 Telegram，
        # 沒有任何委託送出。這裡補上綁定。
        self.api = api
        self.contract = contract

        logger.info(f"{C_GREEN}訂閱商品: {contract.code} - {contract.name}{C_RESET}")
        logger.info(f"{C_YELLOW}開始接收即時報價，按下 Ctrl+C 結束監控。{C_RESET}")
        logger.info("=" * 60)

        @api.on_tick_fop_v1()
        def on_tick(exchange, tick):
            # 處理即時 Tick
            # tick 包含：close, volume, datetime 等
            # [FIX] 這裡原本完全沒有 try/except。Shioaji 是在自己的背景執行緒
            # 呼叫這個 callback，任何未被攔截的例外都會被該執行緒吞掉、
            # 不會顯示在畫面或 log 上 —— 這正是這次「畫面停在訂閱成功、
            # 之後完全沒有 tick 更新」卻沒有任何錯誤訊息的根本原因。
            # 加上例外處理後，未來若再出問題，至少 log 檔會留下 traceback。
            try:
                price = float(tick.close)
                vol = int(tick.volume)
                dt = tick.datetime  # 格式通常為 datetime 物件
                self._process_new_tick(price, vol, dt)
            except Exception as e:
                logger.exception(f"處理即時 Tick 時發生例外，本筆 tick 已略過: {e}")

        api.quote.subscribe(contract, quote_type=sj.QuoteType.Tick)

        # 訂閱完成後，立刻先分析並輸出當前歷史基礎點位
        # [FIX] trigger_actions=False：這裡用的是啟動時載入的「快取歷史資料」
        # （很可能是幾天前的舊資料），只用來印出目前結構狀態方便確認基礎是否正確，
        # 不應該用它來判斷訊號、發 Telegram 通知或觸發模擬下單，
        # 否則每次啟動都可能因為快取最後一根舊 K 線恰好符合條件，
        # 送出一筆用「舊時間戳、舊價位」產生的誤導性即時訊號。
        logger.info(f"{C_GREEN}訂閱成功！正在進行初始 SMC 結構運算...{C_RESET}")
        try:
            self._analyze_and_print_state(trigger_actions=False)
        except Exception as e:
            logger.warning(f"初始 SMC 運算提醒: {str(e)}")

        # 保持執行，並定期檢查是否已經一段時間沒收到任何 tick（斷線/清淡時段提醒）
        HEARTBEAT_WARN_SECONDS = 60
        last_warned = False
        try:
            while True:
                time.sleep(1)
                if self.last_tick_time is not None:
                    idle = (datetime.datetime.now() - self.last_tick_time).total_seconds()
                    if idle >= HEARTBEAT_WARN_SECONDS and not last_warned:
                        logger.warning(f"⚠️ 已經 {int(idle)} 秒沒有收到任何即時報價，請確認連線 / 是否為非交易時段。")
                        last_warned = True
                    elif idle < HEARTBEAT_WARN_SECONDS:
                        last_warned = False
        except KeyboardInterrupt:
            logger.info(f"監控已手動終止。")
            api.logout()

    def _on_order_callback(self, stat: sj.OrderState, msg: dict):
        """處理 Shioaji 委託與成交回報"""
        try:
            # [FIX] 原本寫的是 `sj.OrderState.TDeal`，但 Shioaji SDK 的 OrderState
            # 只有 StockOrder / StockDeal / FuturesOrder / FuturesDeal 四種成員，
            # 根本沒有 TDeal，一執行就丟 AttributeError（被下面的 except 悄悄吃掉，
            # 只留一行 error log）。期貨成交要用 FuturesDeal，這正是
            # 「平倉戰報那個 Telegram 群組完全沒訊息」的 root cause：
            # 每一筆真實成交回報都在這裡失敗，_process_settlement_deal 從未被呼叫過。
            if stat == sj.OrderState.FuturesDeal:
                action = msg.get('action')
                price = msg.get('price')
                qty = msg.get('quantity')
                logger.info(f"⚡ [成交回報] Action: {action}, 價格: {price}, 口數: {qty}")
                
                pos = self.active_position
                if not pos:
                    return
                
                # 若為進場單，記錄真實成交均價（為簡化，直接取第一筆成交價或更新均價）
                if 'real_entry_price' not in pos:
                    pos['real_entry_price'] = float(price)
                    
                    # 發送真實開倉戰報 (因模擬/真實單皆會觸發 callback，此處為真正確立持倉的時機)
                    direction = pos.get('direction', 'UNKNOWN')
                    dir_label = "多單市價買" if direction == "LONG" else "空單市價賣"
                    dt_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    e_int = int(round(float(price)))
                    sl_int = int(round(pos.get('sl', 0)))
                    tp_int = int(round(pos.get('tp1', 0)))
                    lots = pos.get('lots', 2)
                    
                    from smc_trader.telegram_bot import send_telegram_notification, TELEGRAM_SETTLEMENT_CHAT_ID
                    open_msg = (
                        f"🆕 <b>[SMC 真實開倉戰報]</b>\n"
                        f"━━━━━━━━━━━━━━━━━━\n"
                        f"<b>成交時間</b>：{dt_str}\n"
                        f"<b>交易方向</b>：{direction} ({dir_label})\n"
                        f"<b>成交價格</b>：<code>{e_int}</code>\n"
                        f"<b>設定 SL</b>：<code>{sl_int}</code>｜<b>目標 TP1</b>：<code>{tp_int}</code>\n"
                        f"<b>持倉部位</b>：{lots} 口"
                    )
                    send_telegram_notification(open_msg, chat_id=TELEGRAM_SETTLEMENT_CHAT_ID)
                
                # 若有正在等待的平倉單
                if 'pending_close' in pos:
                    close_info = pos['pending_close']
                    exit_price = float(price)
                    close_lots = int(qty)
                    
                    close_info['filled_lots'] = close_info.get('filled_lots', 0) + close_lots
                    close_info['total_value'] = close_info.get('total_value', 0.0) + exit_price * close_lots
                    
                    # 若已全數成交
                    if close_info['filled_lots'] >= close_info['req_lots']:
                        avg_exit = close_info['total_value'] / close_info['filled_lots']
                        self._process_settlement_deal(avg_exit, close_info)
                        del pos['pending_close']
        except Exception as e:
            logger.error(f"處理成交回報失敗: {e}")

    def _process_settlement_deal(self, exit_price: float, close_info: dict):
        """處理平倉完全成交後的損益計算與 Telegram 通知"""
        pos = self.active_position
        if not pos: return
        
        dt_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        dir_label = "多單 (LONG)" if pos['direction'] == 'LONG' else "空單 (SHORT)"
        entry_p = pos.get('real_entry_price', pos['entry_price'])
        req_lots = close_info['req_lots']
        stage_type = close_info['stage_type'] # 'SL', 'TP1', 'FINAL', 'REVERSE'
        reason = close_info['reason']
        
        net_pts, net_pnl = self._calc_pnl(entry_p, exit_price, pos['direction'], req_lots)
        
        # 取得實時餘額
        balance_str = "無法取得"
        if self.api is not None:
            try:
                fa = self.api.futopt_account
                if fa is not None:
                    margin = self.api.margin()
                    balance_str = f"{margin.equity:+,.0f} NTD"
            except Exception as e:
                logger.warning(f"取得餘額失敗: {e}")
                
        # 根據階段發送對應戰報
        if stage_type == 'SL':
            msg = (
                f"🛑 <b>[SMC 真實平倉戰報 - 停損出場]</b>\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"<b>交易標的</b>：台指期近月 (TXFR1)\n"
                f"<b>交易方向</b>：{dir_label}\n"
                f"<b>平倉時間</b>：{dt_str}\n"
                f"<b>平倉原因</b>：{reason}\n\n"
                f"<b>真實進場價</b>：<code>{int(round(entry_p))}</code>\n"
                f"<b>真實平倉價</b>：<code>{int(round(exit_price))}</code>\n"
                f"<b>平倉部位</b>：{req_lots} 口\n"
                f"<b>本次盈虧</b>：<code>{net_pnl:+,.0f} NTD</code> ({net_pts:+.1f} 點)\n\n"
                f"💰 <b>帳戶實時餘額</b>：<code>{balance_str}</code>"
            )
            logger.signal(f"平倉戰報 (停損): {net_pnl:+,.0f} NTD")
            send_telegram_notification(msg, chat_id=TELEGRAM_SETTLEMENT_CHAT_ID)
            self.active_position = None
            
        elif stage_type == 'TP1':
            pos['stage'] = 2
            pos['sl'] = pos['entry_price'] # 保本
            pos['pnl_stage1'] = net_pnl
            pos['remaining_lots'] = pos.get('lots', 2) - req_lots
            
            msg = (
                f"🎉 <b>[SMC 真實平倉戰報 - 部分停利]</b>\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"<b>交易標的</b>：台指期近月 (TXFR1)\n"
                f"<b>交易方向</b>：{dir_label}\n"
                f"<b>平倉時間</b>：{dt_str}\n"
                f"<b>平倉原因</b>：{reason}\n\n"
                f"<b>真實進場價</b>：<code>{int(round(entry_p))}</code>\n"
                f"<b>真實平倉價</b>：<code>{int(round(exit_price))}</code>\n"
                f"<b>平倉部位</b>：{req_lots} 口 (落袋為安)\n"
                f"<b>本次盈虧</b>：<code>{net_pnl:+,.0f} NTD</code>\n\n"
                f"🛡️ <b>風控狀態</b>：剩餘 {pos['remaining_lots']} 口停損已移至保本\n"
                f"💰 <b>帳戶實時餘額</b>：<code>{balance_str}</code>"
            )
            logger.signal(f"平倉戰報 (TP1): {net_pnl:+,.0f} NTD")
            send_telegram_notification(msg, chat_id=TELEGRAM_SETTLEMENT_CHAT_ID)
            
        elif stage_type == 'FINAL' or stage_type == 'REVERSE':
            tot_pnl = pos.get('pnl_stage1', 0.0) + net_pnl
            msg = (
                f"🎯 <b>[SMC 真實平倉戰報 - 最終結算]</b>\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"<b>交易標的</b>：台指期近月 (TXFR1)\n"
                f"<b>交易方向</b>：{dir_label}\n"
                f"<b>平倉時間</b>：{dt_str}\n"
                f"<b>平倉原因</b>：{reason}\n\n"
                f"<b>真實進場價</b>：<code>{int(round(entry_p))}</code>\n"
                f"<b>真實平倉價</b>：<code>{int(round(exit_price))}</code>\n"
                f"<b>平倉部位</b>：{req_lots} 口 (剩餘全部)\n"
                f"<b>後半段損益</b>：<code>{net_pnl:+,.0f} NTD</code>\n\n"
                f"🏆 <b>該單總損益</b>：<code>{tot_pnl:+,.0f} NTD</code>\n"
                f"💰 <b>帳戶實時餘額</b>：<code>{balance_str}</code>"
            )
            logger.signal(f"平倉戰報 (最終): {tot_pnl:+,.0f} NTD")
            send_telegram_notification(msg, chat_id=TELEGRAM_SETTLEMENT_CHAT_ID)
            
            if stage_type == 'FINAL' or (stage_type == 'REVERSE' and 'pending_reverse_order' not in pos):
                 self.active_position = None

    def _run_mock(self):
        """模擬實時監控"""
        logger.info(f"{C_CYAN}啟動模擬實時監控 (1M K線加速為每 10 秒一根)...{C_RESET}")
        logger.info(f"{C_YELLOW}開始接收模擬報價，按下 Ctrl+C 結束監控。{C_RESET}")
        logger.info("=" * 70)
        
        try:
            while True:
                # 模擬 Tick 更新 (每 2 秒一次)
                dt = datetime.datetime.now()
                # 隨機產生 Tick 價格
                tick_change = np.random.normal(0, 2.0)
                price = round(last_price + tick_change, 1)
                vol = np.random.randint(10, 50)
                
                self._process_new_tick(price, vol, dt)
                
                last_price = price
                time.sleep(2)
                
        except KeyboardInterrupt:
            logger.info(f"模擬監控已手動終止。")

    def _process_new_tick(self, price: float, vol: int, dt: Any):
        """處理傳入的即時報價並合建成 K 線"""
        # 確保 dt 統一轉為 datetime 物件
        if isinstance(dt, str):
            dt = pd.to_datetime(dt).to_pydatetime()
        elif hasattr(dt, 'to_pydatetime'):
            dt = dt.to_pydatetime()
        elif not isinstance(dt, datetime.datetime):
            dt = datetime.datetime.now()
            
        # 去除時區屬性以利安全計算時間差
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)

        # 供主迴圈健康檢查使用：只要有進到這裡就代表確實收到報價
        self.last_tick_time = datetime.datetime.now()

        # [FIX] 防止斷線重連或延遲補送時，收到「時間戳早於目前這根K棒起始時間」的
        # 亂序 tick。若不擋掉，append 進 history_1m 後 ts 不再單調遞增，
        # 之後 _analyze_and_print_state() 裡的 df_1m.resample(on='ts') 會直接
        # 拋出 ValueError（而這個例外目前只在 on_tick 內被記錄、不會讓程式當掉，
        # 但該筆之後的分析會整個失敗）。
        if self.current_bar_1m is not None and dt < self.current_bar_1m['ts']:
            logger.warning(f"⚠️ 收到時間倒退的 tick（{dt.strftime('%H:%M:%S.%f')} 早於目前 K 線 {self.current_bar_1m['ts'].strftime('%H:%M:%S.%f')}），已略過。")
            return

        # 模擬模式下，1M K 線加速為 10 秒
        time_step = 10 if self.mode == "mock" else 60
        
        # 判定是否需要換新的一根 K 線
        if self.current_bar_1m is None:
            self.current_bar_1m = {
                'ts': dt,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': vol
            }
            logger.info(f"⚡ [開始新 K 線] {dt.strftime('%H:%M:%S')} | 成交價: {C_BOLD}{price}{C_RESET} | 量: {vol}")
        else:
            cb = self.current_bar_1m
            # 檢查時間差是否達到一個 K 線週期 (60秒或分鐘換棒)
            time_diff = (dt - cb['ts']).total_seconds()
            
            # 若已經跨到新的一分鐘 (例如 19:32 跨到 19:33) 或滿 60 秒
            is_new_minute = (dt.minute != cb['ts'].minute) or (time_diff >= time_step)
            
            if not is_new_minute:
                # 更新當前 K 線
                cb['high'] = max(cb['high'], price)
                cb['low'] = min(cb['low'], price)
                cb['close'] = price
                cb['volume'] += vol
                # 印出即時報價滾動 log
                ts_str = dt.strftime('%H:%M:%S')
                print(f"\r⚡ [即時報價] {ts_str} | 現價: {price:.1f} | 單量: {vol} | K線 (高:{cb['high']:.1f} 低:{cb['low']:.1f})", end="", flush=True)
            else:
                print("\n") # 換行
                # 將當前 K 線歸檔到歷史中
                self.history_1m.append(cb)
                if len(self.history_1m) > 2000:
                    self.history_1m.pop(0)
                
                # 開啟新的一根 K 線
                self.current_bar_1m = {
                    'ts': dt,
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': vol
                }
                
                # 換棒時，進行最新 SMC 特徵檢測並輸出螢幕與 Telegram！
                self._analyze_and_print_state()

    def _analyze_and_print_state(self, trigger_actions: bool = True):
        """對當前歷史 K 線數據進行 SMC 特徵辨識，並精美輸出

        Args:
            trigger_actions: 是否允許本次分析觸發持倉結算、Telegram 訊號通知與模擬下單。
                訂閱成功後、尚未收到任何真實 tick 前的「初始結構運算」應傳入 False，
                因為那時 history_1m 只有啟動時載入的舊快取資料，不該被當成即時訊號來源。
        """
        df_1m = pd.DataFrame(self.history_1m)
        
        rule = '50s' if self.mode == "mock" else '5min'
        df_5m = df_1m.resample(rule, on='ts').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna().reset_index()

        # 特徵檢測
        df_5m_proc = self.detector.process_5m_structure(df_5m)
        df_1m_proc = self.detector.process_1m_signals(df_1m, df_5m_proc)

        # 取得最後一根 K 線的狀態與指標
        last_bar = df_1m_proc.iloc[-1]
        ts_str = last_bar['ts'].strftime('%H:%M:%S')
        price = last_bar['close']
        trend_5m = last_bar['trend_5m']

        # 即時追蹤持倉是否平倉並發送 Telegram 戰報（初始運算不觸發）
        if trigger_actions:
            self._check_position_settlement(last_bar)

        # 檢測是否有特殊信號
        has_signal = False
        signal_level = 0  # 1=Sweep, 2=MSS, 3=CISD（★ 星級）
        signal_tg_name = ""
        signal_str = f"{C_BOLD}無特殊信號{C_RESET}"
        
        if last_bar['sweep_low']:
            signal_str = f"{C_GREEN}{C_BOLD}★ [流動性掠奪] Sweep Low 形成！(買方防守){C_RESET}"
            signal_tg_name = "★ [流動性掠奪] Sweep Low 形成 (多頭訊號)！"
            has_signal = True
            signal_level = 1
        elif last_bar['sweep_high']:
            signal_str = f"{C_RED}{C_BOLD}★ [流動性掠奪] Sweep High 形成！(賣方防守){C_RESET}"
            signal_tg_name = "★ [流動性掠奪] Sweep High 形成 (空頭訊號)！"
            has_signal = True
            signal_level = 1
        elif last_bar['mss_bullish']:
            signal_str = f"{C_GREEN}{C_BOLD}★★ [結構轉換] MSS Bullish 確立！趨勢轉多{C_RESET}"
            signal_tg_name = "★★ [結構轉換] MSS Bullish 確立 (多頭訊號)！"
            has_signal = True
            signal_level = 2
        elif last_bar['mss_bearish']:
            signal_str = f"{C_RED}{C_BOLD}★★ [結構轉換] MSS Bearish 確立！趨勢轉空{C_RESET}"
            signal_tg_name = "★★ [結構轉換] MSS Bearish 確立 (空頭訊號)！"
            has_signal = True
            signal_level = 2
        elif last_bar['cisd_bullish']:
            signal_str = f"{C_GREEN}{C_BOLD}★★★ [價格交付改變] CISD Bullish 爆量突破對立 OB！{C_RESET}"
            signal_tg_name = "★★★ [價格交付改變] CISD Bullish 爆量突破 (多頭訊號)！"
            has_signal = True
            signal_level = 3
        elif last_bar['cisd_bearish']:
            signal_str = f"{C_RED}{C_BOLD}★★★ [價格交付改變] CISD Bearish 爆量跌破對立 OB！{C_RESET}"
            signal_tg_name = "★★★ [價格交付改變] CISD Bearish 爆量跌破 (空頭訊號)！"
            has_signal = True
            signal_level = 3

        # 取得當前 OB / FVG 區間
        bull_ob = f"[{last_bar['bullish_ob_low']} - {last_bar['bullish_ob_high']}]" if not np.isnan(last_bar['bullish_ob_low']) else "無"
        bear_ob = f"[{last_bar['bearish_ob_low']} - {last_bar['bearish_ob_high']}]" if not np.isnan(last_bar['bearish_ob_low']) else "無"
        
        bull_fvg = f"[{last_bar['bullish_fvg_low']} - {last_bar['bullish_fvg_high']}]" if not np.isnan(last_bar['bullish_fvg_low']) else "無"
        bear_fvg = f"[{last_bar['bearish_fvg_low']} - {last_bar['bearish_fvg_high']}]" if not np.isnan(last_bar['bearish_fvg_low']) else "無"

        # 輸出格式
        trend_color = C_GREEN if trend_5m == "BULLISH" else (C_RED if trend_5m == "BEARISH" else C_RESET)
        
        logger.info(f"[{ts_str}] 價格: {C_BOLD}{price}{C_RESET} | 大結構趨勢 (5M): {trend_color}{C_BOLD}{trend_5m}{C_RESET}")
        logger.info(f"       即時信號 : {signal_str}")
        logger.info(f"       多頭 OB  : {C_GREEN}{bull_ob}{C_RESET} | 空頭 OB  : {C_RED}{bear_ob}{C_RESET}")
        logger.info(f"       多頭 FVG : {C_GREEN}{bull_fvg}{C_RESET} | 空頭 FVG : {C_RED}{bear_fvg}{C_RESET}")
        logger.info("-" * 70)

        # 發送 Telegram 通知（初始運算只印出結構狀態，不觸發通知/下單）
        if has_signal and not trigger_actions:
            logger.info(f"       （初始結構運算偵測到訊號條件，但因使用舊快取資料，已略過通知與下單）")

        if has_signal and trigger_actions:
            dt_str = last_bar['ts'].strftime('%Y-%m-%d %H:%M:%S')
            
            # 動態計算交易建議與進場、停損、二階段停利價格
            is_bullish_signal = "bullish" in signal_tg_name.lower() or "sweep low" in signal_tg_name.lower() or "mss bullish" in signal_tg_name.lower() or "cisd bullish" in signal_tg_name.lower()
            is_bearish_signal = "bearish" in signal_tg_name.lower() or "sweep high" in signal_tg_name.lower() or "mss bearish" in signal_tg_name.lower() or "cisd bearish" in signal_tg_name.lower()
            
            # 檢查過濾條件 (0.9x ATR 波動濾網)
            is_volatile = last_bar.get('is_volatile', True)

            if is_volatile:
                if is_bullish_signal:
                    ob_low = last_bar['bullish_ob_low']
                    
                    entry_price = price  # Combo #11: OB未失效即刻市價敲進
                    sl_price = ob_low if not np.isnan(ob_low) else (last_bar['confirmed_sl_1m'] if not np.isnan(last_bar['confirmed_sl_1m']) else last_bar['low'] - 15.0)
                    
                    if not np.isnan(entry_price) and not np.isnan(sl_price) and price > sl_price:
                        sl_points = entry_price - sl_price
                        if sl_points <= 5.0:
                            sl_price = entry_price - 20.0
                            sl_points = 20.0
                        elif sl_points > MAX_SL_POINTS:
                            sl_price = entry_price - MAX_SL_POINTS
                            sl_points = MAX_SL_POINTS
                        
                        # TP1 @ 3.0x RR 二階段分批停利第一目標 (經大數據回測驗證最優)
                        tp1_price = entry_price + sl_points * 3.0
                        
                        logger.signal(f"訊號觸發，發送 Telegram 即時監控通知 (ID: {TELEGRAM_SIGNAL_CHAT_ID}): {signal_tg_name}")
                        self._handle_signal_action("LONG", signal_level, signal_tg_name, entry_price, sl_price, tp1_price, price, dt_str, trend_5m)
                    else:
                        logger.info(f"       🚫 [訊號過濾] 多頭條件不足: entry_price={entry_price}, sl_price={sl_price}, 現價={price} (現價需大於SL)")
                        
                elif is_bearish_signal:
                    ob_high = last_bar['bearish_ob_high']
                    
                    entry_price = price  # Combo #11: OB未失效即刻市價敲進
                    sl_price = ob_high if not np.isnan(ob_high) else (last_bar['confirmed_sh_1m'] if not np.isnan(last_bar['confirmed_sh_1m']) else last_bar['high'] + 15.0)
                    
                    if not np.isnan(entry_price) and not np.isnan(sl_price) and price < sl_price:
                        sl_points = sl_price - entry_price
                        if sl_points <= 5.0:
                            sl_price = entry_price + 20.0
                            sl_points = 20.0
                        elif sl_points > MAX_SL_POINTS:
                            sl_price = entry_price + MAX_SL_POINTS
                            sl_points = MAX_SL_POINTS
                        
                        # TP1 @ 3.0x RR 二階段分批停利第一目標 (經大數據回測驗證最優)
                        tp1_price = entry_price - sl_points * 3.0
                        
                        logger.signal(f"訊號觸發，發送 Telegram 即時監控通知 (ID: {TELEGRAM_SIGNAL_CHAT_ID}): {signal_tg_name}")
                        self._handle_signal_action("SHORT", signal_level, signal_tg_name, entry_price, sl_price, tp1_price, price, dt_str, trend_5m)
                    else:
                        logger.info(f"       🚫 [訊號過濾] 空頭條件不足: entry_price={entry_price}, sl_price={sl_price}, 現價={price} (現價需小於SL)")
            else:
                logger.info(f"       🚫 [訊號過濾] 波動率不足 (is_volatile=False)，略過本次訊號")

    def _handle_signal_action(self, direction: str, signal_level: int, signal_tg_name: str,
                               entry_price: float, sl_price: float, tp1_price: float,
                               price: float, dt_str: str, trend_5m: str):
        """
        依「目前持倉狀態」與「新訊號星級」決定動作：
        - 沒有持倉：正常開新倉（2 口 / 2 階段停利，維持原邏輯）
        - 已有持倉，且持倉是 1★ 訊號開的、本次新訊號是 2★：允許覆蓋
            - 同方向：加碼 1 口，套用新訊號的 SL/TP1
            - 反方向：先以目前市價強制平倉，再反向開 1 口新倉
        - 其他所有情況（含 3★ 想蓋 1★、任何訊號想蓋 2★/3★ 持倉）：
          已持倉就不開新倉，Telegram 顯示「已持倉不開新倉」
        """
        pos = self.active_position
        p_int = int(round(price))
        e_int = int(round(entry_price))
        sl_int = int(round(sl_price))
        tp_int = int(round(tp1_price))
        dir_label = "多單市價買" if direction == "LONG" else "空單市價賣"

        # 1. 無論如何，所有觸發的訊號都要純粹地推播到「訊號來源地」(SIGNAL_CHAT_ID)
        pure_signal_text = (
            f"觸發時間：{dt_str}\n"
            f"最新價格：{p_int}\n"
            f"大結構趨勢 (5M)：{trend_5m}\n"
            f"🚨 訊號類型：{signal_tg_name}\n"
            f"💡 {dir_label}：{e_int} | SL：{sl_int}  | TP：{tp_int}"
        )
        send_telegram_notification(pure_signal_text, chat_id=TELEGRAM_SIGNAL_CHAT_ID)

        if pos is None:
            # 保護機制：若現價距離掛單價太遠（例如跳空造成），為避免 Shioaji 模擬交易所 bug（會直接以市價異常成交），略過下單
            # 此為舊版限價進場的保護，既然改為市價進場，此保護機制可暫時保留以防極端跳空，但可放寬
            if abs(price - entry_price) > 300:
                skip_msg = f"⏸️ [極端跳空過濾] 現價 {p_int} 距離理論邊界過遠（>300點），略過本次下單。"
                logger.info(skip_msg)
                send_telegram_notification(skip_msg, chat_id=TELEGRAM_SETTLEMENT_CHAT_ID)
                return
                
            # 目前無持倉，正常開新倉 (已在上面發送純訊號，真實成交戰報將交由 callback 處理)
            self._place_simulated_2stage_order(direction, entry_price, sl_price, tp1_price, signal_level=signal_level, lots=2)
            return

        # 已有持倉：唯一允許覆蓋的組合是「目前持倉為 1★ 開倉、且本次新訊號為 2★」
        if pos.get('signal_level') == 1 and signal_level == 2:
            if direction == pos['direction']:
                self._add_on_position(direction, signal_tg_name, entry_price, sl_price, tp1_price, dt_str)
            else:
                self._reverse_position(direction, signal_tg_name, entry_price, sl_price, tp1_price, dt_str, price)
            return

        # 其他所有情況：已持倉就不開新倉
        logger.info(f"⏸️ 已持倉中（{pos['direction']}／{pos.get('signal_level', '?')}★開倉），偵測到 {signal_level}★ 新訊號但不符合覆蓋條件，略過本次訊號。")
        skip_text = (
            f"觸發時間：{dt_str}\n"
            f"最新價格：{p_int}\n"
            f"🚨 訊號類型：{signal_tg_name}\n"
            f"⏸️ 已持倉不開新倉"
        )
        send_telegram_notification(skip_text, chat_id=TELEGRAM_SETTLEMENT_CHAT_ID)

    def _add_on_position(self, direction: str, signal_tg_name: str, entry_price: float,
                          sl_price: float, tp1_price: float, dt_str: str):
        """2★ 訊號同方向覆蓋 1★ 持倉：加碼 1 口，套用新訊號的 SL/TP1。"""
        pos = self.active_position
        add_lots = 1

        if pos.get('stage') == 2:
            # 原持倉已過 TP1、剩餘部位保本續抱中：把加碼口數併入剩餘部位，
            # 並用新訊號的 SL/TP1 重新展開一輪停利週期（stage 重設為 1）。
            # 先前 TP1 那筆已落袋的損益 (pnl_stage1) 予以保留，最終結算時會一併加總。
            old_lots = pos.get('remaining_lots', pos.get('lots', 2))
            pos['stage'] = 1
            pos.pop('remaining_lots', None)
        else:
            old_lots = pos.get('lots', 2)

        new_lots = old_lots + add_lots
        preview_close = math.ceil(new_lots / 2)
        preview_remain = new_lots - preview_close

        e_int = int(round(entry_price))
        sl_int = int(round(sl_price))
        tp_int = int(round(tp1_price))

        logger.signal(f"🔼 [2★訊號加倉] {direction} 同方向加碼 {add_lots} 口（{old_lots}→{new_lots} 口），套用新 SL/TP1")

        pos['lots'] = new_lots
        pos['sl'] = sl_price
        pos['tp1'] = tp1_price
        pos['signal_level'] = 2  # 持倉升級為 2★

        remain_note = f"下次 TP1 將平 {preview_close} 口、留 {preview_remain} 口續抱" if preview_remain > 0 else f"下次 TP1 將全部 {preview_close} 口出場"
        msg = (
            f"🔼 <b>[SMC 加倉通知 - 2★訊號覆蓋1★持倉]</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"<b>觸發時間</b>：{dt_str}\n"
            f"<b>訊號類型</b>：{signal_tg_name}\n"
            f"<b>加碼部位</b>：+{add_lots} 口（{old_lots} → {new_lots} 口）\n"
            f"<b>更新 SL</b>：<code>{sl_int}</code>｜<b>更新 TP1</b>：<code>{tp_int}</code>\n"
            f"<b>停利規劃</b>：{remain_note}"
        )
        send_telegram_notification(msg, chat_id=TELEGRAM_SETTLEMENT_CHAT_ID)

        if self.api is not None and self.contract is not None:
            try:
                act = sj.Action.Buy if direction == "LONG" else sj.Action.Sell
                order = self.api.Order(
                    action=act,
                    price=0,
                    quantity=add_lots,
                    order_type=sj.OrderType.ROD,
                    price_type=sj.FuturesPriceType.MKT if hasattr(sj, 'FuturesPriceType') else sj.StockPriceType.MKT
                )
                trade = self.api.place_order(self.contract, order)
                
                # 記錄此加碼單，供 Callback 配對
                if trade and trade.order:
                    # 由於尚未取得真實均價，先將 pending_trade 放進 active_position
                    if 'pending_trades' not in pos:
                        pos['pending_trades'] = []
                    pos['pending_trades'].append({
                        'trade': trade,
                        'purpose': 'add_on',
                        'req_lots': add_lots
                    })
                
                logger.signal(f"✅ [Shioaji 加碼下單成功] 交易單號等待回報...")
            except Exception as e:
                logger.error(f"❌ [Shioaji 加碼下單失敗]: {str(e)}")
        else:
            # 模擬模式：直接觸發加碼成交回報
            msg = {
                'action': 'Buy' if direction == "LONG" else 'Sell',
                'price': entry_price,
                'quantity': add_lots
            }
            logger.info(f"🔧 [模擬模式] 自動觸發加碼成交回報...")
            self._on_order_callback(sj.OrderState.FuturesDeal if hasattr(sj, 'OrderState') else 10, msg)

    def _reverse_position(self, new_direction: str, signal_tg_name: str, entry_price: float,
                           sl_price: float, tp1_price: float, dt_str: str, current_price: float):
        """2★ 訊號反方向覆蓋 1★ 持倉：先以目前市價強制平倉舊部位，再反向開 1 口新倉。"""
        pos = self.active_position
        lots = pos.get('remaining_lots', pos.get('lots', 2)) if pos.get('stage') == 2 else pos.get('lots', 2)

        # 改用實際送單平倉，待成交後再反手
        pos['pending_reverse_order'] = {
            'direction': new_direction,
            'entry_price': entry_price,
            'sl_price': sl_price,
            'tp1_price': tp1_price,
        }
        self._send_close_order(lots, 'REVERSE', f"🔄 偵測到反方向 2★ 訊號（{signal_tg_name}），強制平倉反手")

    def _place_simulated_2stage_order(self, direction: str, entry_price: float, sl_price: float,
                                       tp1_price: float, signal_level: int = 1, lots: int = 2):
        """透過 Shioaji 模擬環境自動送出限價進場與二階段分批停利委託

        Args:
            signal_level: 開倉訊號的星級（1=Sweep, 2=MSS, 3=CISD），記錄在持倉上供後續覆蓋規則判斷。
            lots: 下單口數，預設 2 口（標準 2 階段停利模型）；反向覆蓋開倉時為 1 口。
        """
        e_int = int(round(entry_price))
        sl_int = int(round(sl_price))
        tp_int = int(round(tp1_price))
        
        logger.info(f"🤖 [Shioaji 模擬下單觸發] 自動發起 {lots} 口小台 {direction} 委託 | 限價: {e_int}, SL: {sl_int}, TP1: {tp_int}")
        
        # 設定即時持倉追蹤 (二階段停利)
        self.active_position = {
            'direction': direction,
            'entry_price': entry_price,
            'sl': sl_price,
            'tp1': tp1_price,
            'stage': 1,
            'signal_level': signal_level,
            'lots': lots
        }
        
        # [FIX] 原判斷式為 `hasattr(self, 'api') and ... and self.contract is None`：
        # self.api / self.contract 從未在別處被賦值，hasattr 永遠是 False，
        # 且就算賦值了，`self.contract is None` 這個條件邏輯也是反的
        # （應該是「有合約才下單」而不是「沒合約才下單」）。
        # 這裡改為 self.api / self.contract 已在 _run_shioaji() 中綁定，
        # 且合約存在時才真正送出委託。
        if self.api is not None and self.contract is not None:
            try:
                act = sj.Action.Buy if direction == "LONG" else sj.Action.Sell
                order = self.api.Order(
                    action=act,
                    price=0, # 市價單不指定價格
                    quantity=lots,  # 依 signal_level/覆蓋規則決定口數
                    order_type=sj.OrderType.ROD, # 市價單(MKT) 在 Shioaji 也可以搭配 ROD
                    price_type=sj.FuturesPriceType.MKT if hasattr(sj, 'FuturesPriceType') else sj.StockPriceType.MKT
                )
                trade = self.api.place_order(self.contract, order)
                
                if trade and trade.order:
                    if 'pending_trades' not in self.active_position:
                        self.active_position['pending_trades'] = []
                    self.active_position['pending_trades'].append({
                        'trade': trade,
                        'purpose': 'entry',
                        'req_lots': lots
                    })
                
                logger.signal(f"✅ [Shioaji 模擬帳戶進場委託已送出] 等待成交回報...")
            except Exception as e:
                logger.error(f"❌ [Shioaji 模擬帳戶下單失敗]: {str(e)}")
        else:
            # 模擬模式：直接觸發進場成交回報
            msg = {
                'action': 'Buy' if direction == "LONG" else 'Sell',
                'price': entry_price,
                'quantity': lots
            }
            logger.info(f"🔧 [模擬模式] 自動觸發進場成交回報...")
            self._on_order_callback(sj.OrderState.FuturesDeal if hasattr(sj, 'OrderState') else 10, msg)

    def _calc_pnl(self, entry_price: float, exit_price: float, direction: str, lots: int) -> "tuple[float, float]":
        """依離場價格與口數，計算扣除滑價與手續費後的淨點數與淨損益(NTD)。

        [FIX] 舊版公式（`net_pts*50 - 100`）完全沒有依口數縮放，就算文字寫「2 口」，
        算出來的錢其實只等於 1 口的損益。這裡統一改用小台每口 50 NTD 點值，
        並依 config.py 的 SLIPPAGE_POINTS / COMMISSION_FEE（單邊）換算成來回成本，
        正確依 `lots` 縮放。
        """
        dir_mult = 1 if direction == 'LONG' else -1
        net_pts = (exit_price - entry_price) * dir_mult - ROUNDTRIP_SLIPPAGE_PTS
        net_pnl = net_pts * MINI_POINT_VALUE * lots - ROUNDTRIP_COMMISSION_PER_LOT * lots
        return net_pts, net_pnl

    def _send_close_order(self, lots: int, stage_type: str, reason: str):
        pos = self.active_position
        if not pos: return
        
        pos['pending_close'] = {
            'req_lots': lots,
            'filled_lots': 0,
            'total_value': 0.0,
            'stage_type': stage_type,
            'reason': reason
        }
        
        if self.api is not None and self.contract is not None:
            try:
                # 平倉單：多單賣出，空單買進
                act = sj.Action.Sell if pos['direction'] == "LONG" else sj.Action.Buy
                order = self.api.Order(
                    action=act,
                    price=0,
                    quantity=lots,
                    order_type=sj.OrderType.ROD,  # Shioaji 市價有時也可使用 ROD
                    price_type=sj.FuturesPriceType.MKT if hasattr(sj, 'FuturesPriceType') else sj.StockPriceType.MKT
                )
                trade = self.api.place_order(self.contract, order)
                logger.signal(f"✅ [Shioaji 平倉委託已送出] 等待成交回報... (原因: {reason})")
            except Exception as e:
                logger.error(f"❌ [Shioaji 平倉單下單失敗]: {str(e)}")
        else:
            # 模擬模式 (無真實 API)：直接模擬成交回報
            if pos['direction'] == 'LONG':
                # 多單平倉 (賣出) 取買價 (此處以當前收盤價模擬)
                sim_price = pos.get('tp1') if stage_type == 'TP1' else pos.get('sl') if stage_type == 'SL' else pos['entry_price'] # 這裡只是一個大概，實戰會由真實價位取代
            else:
                sim_price = pos.get('tp1') if stage_type == 'TP1' else pos.get('sl') if stage_type == 'SL' else pos['entry_price']
            
            # 使用當前時間模擬成交，或者因為沒有最新 K 線價格，簡單傳入 0 讓 _process_settlement_deal 使用上次理論價格...
            # 其實最好的方法是直接傳遞一個假的 msg 到 _on_order_callback
            msg = {
                'action': 'Sell' if pos['direction'] == 'LONG' else 'Buy',
                'price': sim_price,
                'quantity': lots
            }
            logger.info(f"🔧 [模擬模式] 自動觸發平倉成交回報...")
            self._on_order_callback(sj.OrderState.FuturesDeal if hasattr(sj, 'OrderState') else 10, msg)

    def _check_position_settlement(self, last_bar: Dict[str, Any]):
        """即時檢查持倉是否到達 TP1 (3.0x RR)、保本點 SL、或反轉訊號，並自動送出平倉委託"""
        pos = self.active_position
        if pos is None:
            return

        # 若已送出平倉委託且等待回報中，避免重複觸發
        if 'pending_close' in pos:
            return

        price = last_bar['close']
        high = last_bar['high']
        low = last_bar['low']
        
        total_lots = pos.get('lots', 2)
        
        # Stage 1: 尚未到達 TP1 (持有 100% 部位)
        if pos['stage'] == 1:
            is_sl_hit = (low <= pos['sl']) if pos['direction'] == 'LONG' else (high >= pos['sl'])
            if is_sl_hit:
                self._send_close_order(total_lots, 'SL', "🛑 觸及 SL 停損價")
                return
                
            is_tp1_hit = (high >= pos['tp1']) if pos['direction'] == 'LONG' else (low <= pos['tp1'])
            if is_tp1_hit:
                close_lots = math.ceil(total_lots / 2)   # 鎖利優先：平掉「一半以上」
                remain_lots = total_lots - close_lots
                stage = 'FINAL' if remain_lots <= 0 else 'TP1'
                reason = f"🎯 到達 TP1 第一目標 (3.0x RR)" + ("，全部出場" if remain_lots <= 0 else "，部分停利")
                self._send_close_order(close_lots, stage, reason)
                return

        # Stage 2: 已於 TP1 平掉一部分且保本 (持有 pos['remaining_lots'] 口)
        elif pos['stage'] == 2:
            closed = False
            reason = ""
            
            if pos['direction'] == 'LONG':
                if low <= pos['sl']:
                    reason, closed = "🛑 觸及進場保本價", True
                elif last_bar['mss_bearish'] or last_bar['sweep_high']:
                    reason, closed = "🚨 檢測到反向空頭 SMC 訊號", True
            else: # SHORT
                if high >= pos['sl']:
                    reason, closed = "🛑 觸及進場保本價", True
                elif last_bar['mss_bullish'] or last_bar['sweep_low']:
                    reason, closed = "🚨 檢測到反向多頭 SMC 訊號", True
                    
            if closed:
                remain_lots = pos.get('remaining_lots', total_lots - math.ceil(total_lots / 2))
                self._send_close_order(remain_lots, 'FINAL', reason)