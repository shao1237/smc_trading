import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
import shioaji as sj
from typing import List, Dict, Any, Optional
from smc_trader.config import (
    SHIOAJI_API_KEY, SHIOAJI_SECRET_KEY, SHIOAJI_SIMULATION,
    SWING_WINDOW_5M, SWING_WINDOW_1M, VOLUME_MA_PERIOD, VOLUME_MULT,
    DEFAULT_RR, MAX_SL_POINTS, TELEGRAM_SIGNAL_CHAT_ID, TELEGRAM_SETTLEMENT_CHAT_ID
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
        
        # 當前模擬/實盤持倉追蹤 (二階段停利與保本管理)
        self.active_position: Optional[Dict[str, Any]] = None
        
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
        
        # 尋找當前近月合約
        futures = api.Contracts.Futures.TXF
        try:
            contract = futures["TXFR1"]
        except KeyError:
            contract = getattr(futures, "TXFR1", None)
            
        if contract is None:
            raise ValueError("找不到 TXFR1 期貨合約")
        
        logger.info(f"{C_GREEN}訂閱商品: {contract.code} - {contract.name}{C_RESET}")
        logger.info(f"{C_YELLOW}開始接收即時報價，按下 Ctrl+C 結束監控。{C_RESET}")
        logger.info("=" * 60)

        @api.on_tick_fop_v1()
        def on_tick(exchange, tick):
            # 處理即時 Tick
            # tick 包含：close, volume, datetime 等
            price = float(tick.close)
            vol = int(tick.volume)
            dt = tick.datetime # 格式通常為 datetime 物件
            
            self._process_new_tick(price, vol, dt)

        api.quote.subscribe(contract, quote_type=sj.constant.QuoteType.Tick)
        
        # 訂閱完成後，立刻先分析並輸出當前歷史基礎點位
        logger.info(f"{C_GREEN}訂閱成功！正在進行初始 SMC 結構運算...{C_RESET}")
        try:
            self._analyze_and_print_state()
        except Exception as e:
            logger.warning(f"初始 SMC 運算提醒: {str(e)}")

        # 保持執行
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info(f"監控已手動終止。")
            api.logout()

    def _run_mock(self):
        """模擬實時監控"""
        logger.info(f"{C_CYAN}啟動模擬實時監控 (1M K線加速為每 10 秒一根)...{C_RESET}")
        logger.info(f"{C_YELLOW}開始接收模擬報價，按下 Ctrl+C 結束監控。{C_RESET}")
        logger.info("=" * 70)
        
        last_price = self.history_1m[-1]['close']
        
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

    def _analyze_and_print_state(self):
        """對當前歷史 K 線數據進行 SMC 特徵辨識，並精美輸出"""
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

        # 即時追蹤持倉是否平倉並發送 Telegram 戰報
        self._check_position_settlement(last_bar)

        # 檢測是否有特殊信號
        has_signal = False
        signal_tg_name = ""
        signal_str = f"{C_BOLD}無特殊信號{C_RESET}"
        
        if last_bar['sweep_low']:
            signal_str = f"{C_GREEN}{C_BOLD}★ [流動性掠奪] Sweep Low 形成！(買方防守){C_RESET}"
            signal_tg_name = "★ [流動性掠奪] Sweep Low 形成 (多頭訊號)！"
            has_signal = True
        elif last_bar['sweep_high']:
            signal_str = f"{C_RED}{C_BOLD}★ [流動性掠奪] Sweep High 形成！(賣方防守){C_RESET}"
            signal_tg_name = "★ [流動性掠奪] Sweep High 形成 (空頭訊號)！"
            has_signal = True
        elif last_bar['mss_bullish']:
            signal_str = f"{C_GREEN}{C_BOLD}★★ [結構轉換] MSS Bullish 確立！趨勢轉多{C_RESET}"
            signal_tg_name = "★★ [結構轉換] MSS Bullish 確立 (多頭訊號)！"
            has_signal = True
        elif last_bar['mss_bearish']:
            signal_str = f"{C_RED}{C_BOLD}★★ [結構轉換] MSS Bearish 確立！趨勢轉空{C_RESET}"
            signal_tg_name = "★★ [結構轉換] MSS Bearish 確立 (空頭訊號)！"
            has_signal = True
        elif last_bar['cisd_bullish']:
            signal_str = f"{C_GREEN}{C_BOLD}★★★ [價格交付改變] CISD Bullish 爆量突破對立 OB！{C_RESET}"
            signal_tg_name = "★★★ [價格交付改變] CISD Bullish 爆量突破 (多頭訊號)！"
            has_signal = True
        elif last_bar['cisd_bearish']:
            signal_str = f"{C_RED}{C_BOLD}★★★ [價格交付改變] CISD Bearish 爆量跌破對立 OB！{C_RESET}"
            signal_tg_name = "★★★ [價格交付改變] CISD Bearish 爆量跌破 (空頭訊號)！"
            has_signal = True

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

        # 發送 Telegram 通知
        if has_signal:
            dt_str = last_bar['ts'].strftime('%Y-%m-%d %H:%M:%S')
            
            # 動態計算交易建議與進場、停損、二階段停利價格
            is_bullish_signal = "bullish" in signal_tg_name.lower() or "sweep low" in signal_tg_name.lower() or "mss bullish" in signal_tg_name.lower() or "cisd bullish" in signal_tg_name.lower()
            is_bearish_signal = "bearish" in signal_tg_name.lower() or "sweep high" in signal_tg_name.lower() or "mss bearish" in signal_tg_name.lower() or "cisd bearish" in signal_tg_name.lower()
            
            # 檢查過濾條件 (0.9x ATR 波動濾網)
            is_volatile = last_bar.get('is_volatile', True)

            if is_volatile:
                if is_bullish_signal:
                    ob_low = last_bar['bullish_ob_low']
                    ob_high = last_bar['bullish_ob_high']
                    fvg_high = last_bar['bullish_fvg_high']
                    
                    entry_price = ob_high if not np.isnan(ob_high) else fvg_high
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
                        
                        p_int = int(round(price))
                        e_int = int(round(entry_price))
                        sl_int = int(round(sl_price))
                        tp_int = int(round(tp1_price))
                        
                        tg_text = (
                            f"觸發時間：{dt_str}\n"
                            f"最新價格：{p_int}\n"
                            f"大結構趨勢 (5M)：{trend_5m}\n"
                            f"🚨 訊號類型：{signal_tg_name}\n"
                            f"💡 多單限價買：{e_int} | SL：{sl_int}  | TP：{tp_int}"
                        )
                        logger.signal(f"訊號觸發，發送 Telegram 即時監控通知 (ID: {TELEGRAM_SIGNAL_CHAT_ID}): {signal_tg_name}")
                        send_telegram_notification(tg_text, chat_id=TELEGRAM_SIGNAL_CHAT_ID)
                        self._place_simulated_2stage_order("LONG", entry_price, sl_price, tp1_price)
                        
                elif is_bearish_signal:
                    ob_low = last_bar['bearish_ob_low']
                    ob_high = last_bar['bearish_ob_high']
                    fvg_low = last_bar['bearish_fvg_low']
                    
                    entry_price = ob_low if not np.isnan(ob_low) else fvg_low
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
                        
                        p_int = int(round(price))
                        e_int = int(round(entry_price))
                        sl_int = int(round(sl_price))
                        tp_int = int(round(tp1_price))
                        
                        tg_text = (
                            f"觸發時間：{dt_str}\n"
                            f"最新價格：{p_int}\n"
                            f"大結構趨勢 (5M)：{trend_5m}\n"
                            f"🚨 訊號類型：{signal_tg_name}\n"
                            f"💡 空單限價賣：{e_int} | SL：{sl_int}  | TP：{tp_int}"
                        )
                        logger.signal(f"訊號觸發，發送 Telegram 即時監控通知 (ID: {TELEGRAM_SIGNAL_CHAT_ID}): {signal_tg_name}")
                        send_telegram_notification(tg_text, chat_id=TELEGRAM_SIGNAL_CHAT_ID)
                        self._place_simulated_2stage_order("SHORT", entry_price, sl_price, tp1_price)

    def _place_simulated_2stage_order(self, direction: str, entry_price: float, sl_price: float, tp1_price: float):
        """透過 Shioaji 模擬環境自動送出 2 口小台 (TXFR1) 限價進場與二階段分批停利委託"""
        e_int = int(round(entry_price))
        sl_int = int(round(sl_price))
        tp_int = int(round(tp1_price))
        
        logger.info(f"🤖 [Shioaji 模擬下單觸發] 自動發起 2 口小台 {direction} 委託 | 限價: {e_int}, SL: {sl_int}, TP1: {tp_int}")
        
        # 設定即時持倉追蹤 (二階段停利)
        self.active_position = {
            'direction': direction,
            'entry_price': entry_price,
            'sl': sl_price,
            'tp1': tp1_price,
            'stage': 1
        }
        
        if hasattr(self, 'api') and self.api is not None and hasattr(self, 'contract') and self.contract is None:
            try:
                act = sj.constant.Action.Buy if direction == "LONG" else sj.constant.Action.Sell
                order = self.api.Order(
                    action=act,
                    price=e_int,
                    quantity=2,  # 自動下單 2 口小台以利二階段分批停利
                    order_type=sj.constant.OrderType.ROD,
                    price_type=sj.constant.StockPriceType.LMT,
                    market_type=sj.constant.MarketType.Future
                )
                trade = self.api.place_order(self.contract, order)
                logger.signal(f"✅ [Shioaji 模擬帳戶下單成功] 交易單號: {trade.order.id}")
            except Exception as e:
                logger.error(f"❌ [Shioaji 模擬帳戶下單失敗]: {str(e)}")

    def _check_position_settlement(self, last_bar: Dict[str, Any]):
        """即時檢查持倉是否到達 TP1 (3.0x RR)、保本點 SL、或反轉訊號，並自動發送 Telegram 戰報至平倉頻道 (-1004387856503)"""
        pos = self.active_position
        if pos is None:
            return

        price = last_bar['close']
        high = last_bar['high']
        low = last_bar['low']
        dt_str = last_bar['ts'].strftime('%Y-%m-%d %H:%M:%S')
        
        dir_label = "多單 (LONG)" if pos['direction'] == 'LONG' else "空單 (SHORT)"
        dir_mult = 1 if pos['direction'] == 'LONG' else -1
        
        # Stage 1: 尚未到達 TP1 (持有 100% 部位, 2口)
        if pos['stage'] == 1:
            is_sl_hit = (low <= pos['sl']) if pos['direction'] == 'LONG' else (high >= pos['sl'])
            if is_sl_hit:
                sl_p = pos['sl']
                net_pts = (sl_p - pos['entry_price']) * dir_mult - 2.0
                net_pnl = (net_pts * 50.0) - 100.0
                
                msg = (
                    f"🛑 <b>[SMC 模擬單平倉戰報 - 停損出場]</b>\n"
                    f"━━━━━━━━━━━━━━━━━━\n"
                    f"<b>交易標的</b>：台指期近月 (TXFR1)\n"
                    f"<b>交易方向</b>：{dir_label}\n"
                    f"<b>平倉時間</b>：{dt_str}\n"
                    f"<b>平倉原因</b>：🛑 觸及 SL 停損價\n\n"
                    f"<b>進場價格</b>：<code>{int(round(pos['entry_price']))}</code>\n"
                    f"<b>平倉價格</b>：<code>{int(round(sl_p))}</code>\n"
                    f"<b>平倉部位</b>：2 口\n"
                    f"<b>本次盈虧</b>：<code>{net_pnl:+,.0f} NTD</code> ({net_pts:+.1f} 點)"
                )
                logger.signal(f"模擬單觸及 SL 停損，發送平倉戰報至 ID {TELEGRAM_SETTLEMENT_CHAT_ID}: {net_pnl:+,.0f} NTD")
                send_telegram_notification(msg, chat_id=TELEGRAM_SETTLEMENT_CHAT_ID)
                self.active_position = None
                return
                
            is_tp1_hit = (high >= pos['tp1']) if pos['direction'] == 'LONG' else (low <= pos['tp1'])
            if is_tp1_hit:
                tp1_p = pos['tp1']
                net_pts = (tp1_p - pos['entry_price']) * dir_mult - 2.0
                net_pnl_50 = ((net_pts * 50.0) - 100.0) * 0.5
                
                pos['stage'] = 2
                pos['sl'] = pos['entry_price']  # 移動保本
                pos['pnl_stage1'] = net_pnl_50
                
                msg = (
                    f"🎉 <b>[SMC 模擬單平倉戰報 - 部分停利]</b>\n"
                    f"━━━━━━━━━━━━━━━━━━\n"
                    f"<b>交易標的</b>：台指期近月 (TXFR1)\n"
                    f"<b>交易方向</b>：{dir_label}\n"
                    f"<b>平倉時間</b>：{dt_str}\n"
                    f"<b>平倉原因</b>：🎯 到達 TP1 第一目標 (3.0x RR)！\n\n"
                    f"<b>進場價格</b>：<code>{int(round(pos['entry_price']))}</code>\n"
                    f"<b>平倉價格</b>：<code>{int(round(tp1_p))}</code>\n"
                    f"<b>平倉部位</b>：1 口 (50% 部位落袋為安)\n"
                    f"<b>本次盈虧</b>：<code>{net_pnl_50:+,.0f} NTD</code>\n\n"
                    f"🛡️ <b>風控狀態</b>：剩餘 1 口部位停損價已自動拉回保本價 (<code>{int(round(pos['entry_price']))}</code>)！"
                )
                logger.signal(f"模擬單觸及 TP1 平倉 50%，發送部分停利戰報至 ID {TELEGRAM_SETTLEMENT_CHAT_ID}: {net_pnl_50:+,.0f} NTD")
                send_telegram_notification(msg, chat_id=TELEGRAM_SETTLEMENT_CHAT_ID)
                return

        # Stage 2: 已平倉 50% 且保本 (持有剩餘 50% 部位, 1口)
        elif pos['stage'] == 2:
            closed = False
            exit_p = 0.0
            reason = ""
            
            if pos['direction'] == 'LONG':
                if low <= pos['sl']:
                    exit_p, reason, closed = pos['sl'], "🛑 觸及進場保本價", True
                elif last_bar['mss_bearish'] or last_bar['sweep_high']:
                    exit_p, reason, closed = price, "🚨 檢測到反向空頭 SMC 訊號", True
            else: # SHORT
                if high >= pos['sl']:
                    exit_p, reason, closed = pos['sl'], "🛑 觸及進場保本價", True
                elif last_bar['mss_bullish'] or last_bar['sweep_low']:
                    exit_p, reason, closed = price, "🚨 檢測到反向多頭 SMC 訊號", True
                    
            if closed:
                net_pts = (exit_p - pos['entry_price']) * dir_mult - 2.0
                net_pnl_50 = ((net_pts * 50.0) - 100.0) * 0.5
                tot_pnl = pos.get('pnl_stage1', 0.0) + net_pnl_50
                
                msg = (
                    f"🎯 <b>[SMC 模擬單平倉戰報 - 最終結算]</b>\n"
                    f"━━━━━━━━━━━━━━━━━━\n"
                    f"<b>交易標的</b>：台指期近月 (TXFR1)\n"
                    f"<b>交易方向</b>：{dir_label}\n"
                    f"<b>平倉時間</b>：{dt_str}\n"
                    f"<b>平倉原因</b>：{reason}\n\n"
                    f"<b>進場價格</b>：<code>{int(round(pos['entry_price']))}</code>\n"
                    f"<b>平倉價格</b>：<code>{int(round(exit_p))}</code>\n"
                    f"<b>平倉部位</b>：1 口 (剩餘 50% 部位)\n"
                    f"<b>後半段損益</b>：<code>{net_pnl_50:+,.0f} NTD</code>\n\n"
                    f"🏆 <b>該單累積總損益</b>：<code>{tot_pnl:+,.0f} NTD</code> (本單圓滿結束！)"
                )
                logger.signal(f"模擬單最終平倉，發送最終結算戰報至 ID {TELEGRAM_SETTLEMENT_CHAT_ID}: {tot_pnl:+,.0f} NTD")
                send_telegram_notification(msg, chat_id=TELEGRAM_SETTLEMENT_CHAT_ID)
                self.active_position = None
