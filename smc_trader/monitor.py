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
    SLIPPAGE_POINTS, COMMISSION_FEE, ATR_MULT
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
            atr_mult=ATR_MULT,
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
        self.closing_positions: List[Dict[str, Any]] = [] # 暫存等待平倉成交回報的部位


        # 最近一次收到 tick 的時間，供健康檢查（多久沒收到報價）使用
        self.last_tick_time: Optional[datetime.datetime] = None

        # Shioaji API / 合約物件；於 _run_shioaji() 中實際賦值，
        # 供 _place_simulated_2stage_order() 送出模擬單使用
        self.api: Optional[Any] = None
        self.contract: Optional[Any] = None

        # 預先載入一部分歷史數據以讓 Swing Points 能夠在初始時就能被計算
        self._init_balance_file()
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
            if stat == getattr(sj.OrderState, "FuturesDeal", None) or stat == 10:  # 10 = simulated fallback
                action = msg.get('action')
                price = msg.get('price')
                qty = msg.get('quantity')
                print()
                logger.info(f"⚡ [成交回報] Action: {action}, 價格: {price}, 口數: {qty}")
                
                # 若為進場單或加碼單 [FIX 2026-08-05]
                if action in ["Buy", "Sell"]:
                    pos = self.active_position
                    if pos:
                        matched_purpose = None
                        if 'pending_trades' in pos:
                            for t in pos['pending_trades']:
                                if t['purpose'] == 'entry':
                                    matched_purpose = 'entry'
                                    pos['pending_trades'].remove(t)
                                    break
                                elif t['purpose'] == 'add_on':
                                    matched_purpose = 'add_on'
                                    pos['pending_trades'].remove(t)
                                    break
                        else: # [Mock 模擬模式]
                            if 'real_entry_price' not in pos:
                                matched_purpose = 'entry'
                            elif 'pending_close' not in pos:
                                # 已經有建倉均價、且不是平倉單，那一定就是加碼單
                                matched_purpose = 'add_on'

                        if matched_purpose == 'entry' and 'real_entry_price' not in pos:
                            pos['real_entry_price'] = float(price)

                            direction = pos.get('direction', 'UNKNOWN')
                            dir_label = "多單市價買" if direction == "LONG" else "空單市價賣"
                            dt_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            e_int = int(round(float(price)))
                            sl_int = int(round(pos.get('sl', 0)))
                            tp_int = int(round(pos.get('tp1', 0)))
                            lots = pos.get('lots', 2)

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
                            return True

                        elif matched_purpose == 'add_on':
                            direction = pos.get('direction', 'UNKNOWN')
                            dir_label = "多單市價買" if direction == "LONG" else "空單市價賣"
                            dt_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            e_int = int(round(float(price)))
                            total_lots = pos.get('lots', 2)
                            has_sl2 = pos.get('sl2') is not None and not isinstance(pos.get('sl2'), float) or (isinstance(pos.get('sl2'), float) and not math.isnan(pos.get('sl2')))
                            sl1_val = int(round(pos.get('sl', 0)))
                            sl_str = f"SL1: {sl1_val} | SL2: {int(round(pos.get('sl2', 0)))}" if has_sl2 else f"SL: {sl1_val}"
                            tp_int = int(round(pos.get('tp1', 0)))

                            add_msg = (
                                f"🔼 <b>[SMC 加倉成交確認]</b>\n"
                                f"━━━━━━━━━━━━━━━━━━\n"
                                f"<b>成交時間</b>：{dt_str}\n"
                                f"<b>交易方向</b>：{direction} ({dir_label})\n"
                                f"<b>加碼成交價</b>：<code>{e_int}</code>\n"
                                f"<b>更新後持倉</b>：{total_lots} 口 | <b>{sl_str}</b> | <b>TP1</b>: <code>{tp_int}</code>"
                            )
                            send_telegram_notification(add_msg, chat_id=TELEGRAM_SETTLEMENT_CHAT_ID)
                            return False

            # 平倉成交 (可能來自 active_position 或已經移到 closing_positions 的舊部位)
            target_pos = None
            if self.active_position and 'pending_close' in self.active_position:
                target_pos = self.active_position
            else:
                for cp in self.closing_positions:
                    if 'pending_close' in cp:
                        target_pos = cp
                        break
            
            if target_pos:
                close_info = target_pos['pending_close']
                exit_price = float(price)
                close_lots = int(qty)
                
                close_info['filled_lots'] = close_info.get('filled_lots', 0) + close_lots
                close_info['total_value'] = close_info.get('total_value', 0.0) + exit_price * close_lots
                
                # 若已全數成交
                if close_info['filled_lots'] >= close_info['req_lots']:
                    avg_exit = close_info['total_value'] / close_info['filled_lots']
                    self._process_settlement_deal(avg_exit, close_info, target_pos)
                    del target_pos['pending_close']
        except Exception as e:
            logger.error(f"處理成交回報失敗: {e}")

    def _process_settlement_deal(self, exit_price: float, close_info: dict, pos: dict):
        """處理平倉完全成交後的損益計算與 Telegram 通知"""
        if not pos: return
        
        dt_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        dir_label = "多單 (LONG)" if pos['direction'] == 'LONG' else "空單 (SHORT)"
        entry_p = pos.get('real_entry_price', pos['entry_price'])
        req_lots = close_info['req_lots']
        stage_type = close_info['stage_type'] # 'SL', 'SL1', 'TP1', 'FINAL', 'REVERSE'
        reason = close_info['reason']
        
        net_pts, net_pnl = self._calc_pnl(entry_p, exit_price, pos['direction'], req_lots)
        
        # 將本次淨損益累計到本地餘額暫存檔
        self._update_balance(net_pnl)
        balance_str = f"{self._get_balance():+,.0f} NTD"
                
        # 根據階段發送對應戰報
        if stage_type == 'SL' or stage_type == 'SL1':
            stage_name = "停損出場" if stage_type == 'SL' else "分批停損 (SL1)"
            msg = (
                f"🛑 <b>[SMC 真實平倉戰報 - {stage_name}]</b>\n"
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
            logger.signal(f"平倉戰報 ({stage_name}): {net_pnl:+,.0f} NTD")
            send_telegram_notification(msg, chat_id=TELEGRAM_SETTLEMENT_CHAT_ID)
            
            if stage_type == 'SL1':
                pos['pnl_stage1'] = pos.get('pnl_stage1', 0.0) + net_pnl
                pos['remaining_lots'] = pos.get('lots', 2) - req_lots
            else:
                if pos in self.closing_positions:
                    self.closing_positions.remove(pos)
                elif self.active_position == pos:
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

            if stage_type == 'REVERSE' and 'pending_reverse' in pos:
                # [FIX 2026-08-06] 反手交易延遲開倉：平倉戰報已發出，
                # 現在才送出新倉委託，保證 TG 時序「先平倉 → 後開倉」。
                rev = pos.pop('pending_reverse')
                if pos in self.closing_positions:
                    self.closing_positions.remove(pos)
                self.active_position = None
                logger.info(f"🔄 [反手開倉] 平倉結算完成，送出 {rev['direction']} {rev['lots']} 口新倉")
                self._place_simulated_2stage_order(
                    rev['direction'], rev['entry_price'], rev['sl_price'],
                    rev['tp1_price'], signal_level=rev['signal_level'],
                    lots=rev['lots'], sl2_price=rev['sl2_price']
                )
            elif stage_type == 'FINAL' or (stage_type == 'REVERSE' and 'pending_reverse_order' not in pos):
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
        if isinstance(dt, str):
            dt = pd.to_datetime(dt).to_pydatetime()
        elif hasattr(dt, 'to_pydatetime'):
            dt = dt.to_pydatetime()
        elif not isinstance(dt, datetime.datetime):
            dt = datetime.datetime.now()
            
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)

        self.last_tick_time = datetime.datetime.now()

        t = dt.time()
        if (datetime.time(13, 45) <= t < datetime.time(15, 0)) or (datetime.time(5, 0) <= t < datetime.time(8, 45)):
            return

        if self.current_bar_1m is not None and dt < self.current_bar_1m['ts']:
            logger.warning(f"⚠️ 收到時間倒退的 tick（{dt.strftime('%H:%M:%S.%f')} 早於目前 K 線 {self.current_bar_1m['ts'].strftime('%H:%M:%S.%f')}），已略過。")
            return

        time_step = 10 if self.mode == "mock" else 60
        
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
            time_diff = (dt - cb['ts']).total_seconds()
            
            is_new_minute = (dt.minute != cb['ts'].minute) or (time_diff >= time_step)
            
            if not is_new_minute:
                cb['high'] = max(cb['high'], price)
                cb['low'] = min(cb['low'], price)
                cb['close'] = price
                cb['volume'] += vol
                ts_str = dt.strftime('%H:%M:%S')
                print(f"\r⚡ [即時報價] {ts_str} | 現價: {price:.1f} | 單量: {vol} | K線 (高:{cb['high']:.1f} 低:{cb['low']:.1f})", end="", flush=True)
            else:
                print("\n") 
                self.history_1m.append(cb)
                if len(self.history_1m) > 2000:
                    self.history_1m.pop(0)
                
                self.current_bar_1m = {
                    'ts': dt,
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': vol
                }
                
                self._analyze_and_print_state()

    def _analyze_and_print_state(self, trigger_actions: bool = True):
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

        df_5m_proc = self.detector.process_5m_structure(df_5m)
        df_1m_proc = self.detector.process_1m_signals(df_1m, df_5m_proc)

        last_bar = df_1m_proc.iloc[-1]
        ts_str = last_bar['ts'].strftime('%H:%M:%S')
        price = last_bar['close']
        trend_5m = last_bar['trend_5m']

        if trigger_actions:
            self._check_position_settlement(last_bar)

        has_signal = False
        signal_level = 0
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

        bull_ob = f"[{last_bar['bullish_ob_low']} - {last_bar['bullish_ob_high']}]" if not np.isnan(last_bar['bullish_ob_low']) else "無"
        bear_ob = f"[{last_bar['bearish_ob_low']} - {last_bar['bearish_ob_high']}]" if not np.isnan(last_bar['bearish_ob_low']) else "無"
        
        bull_fvg = f"[{last_bar['bullish_fvg_low']} - {last_bar['bullish_fvg_high']}]" if not np.isnan(last_bar['bullish_fvg_low']) else "無"
        bear_fvg = f"[{last_bar['bearish_fvg_low']} - {last_bar['bearish_fvg_high']}]" if not np.isnan(last_bar['bearish_fvg_low']) else "無"

        trend_color = C_GREEN if trend_5m == "BULLISH" else (C_RED if trend_5m == "BEARISH" else C_RESET)
        
        logger.info(f"[{ts_str}] 價格: {C_BOLD}{price}{C_RESET} | 大結構趨勢 (5M): {trend_color}{C_BOLD}{trend_5m}{C_RESET}")
        logger.info(f"       即時信號 : {signal_str}")
        logger.info(f"       多頭 OB  : {C_GREEN}{bull_ob}{C_RESET} | 空頭 OB  : {C_RED}{bear_ob}{C_RESET}")
        logger.info(f"       多頭 FVG : {C_GREEN}{bull_fvg}{C_RESET} | 空頭 FVG : {C_RED}{bear_fvg}{C_RESET}")
        logger.info("-" * 70)

        if has_signal and not trigger_actions:
            logger.info(f"       （初始結構運算偵測到訊號條件，但因使用舊快取資料，已略過通知與下單）")

        if has_signal and trigger_actions:
            dt_str = last_bar['ts'].strftime('%Y-%m-%d %H:%M:%S')
            is_bullish_signal = "bullish" in signal_tg_name.lower() or "sweep low" in signal_tg_name.lower() or "mss bullish" in signal_tg_name.lower() or "cisd bullish" in signal_tg_name.lower()
            is_bearish_signal = "bearish" in signal_tg_name.lower() or "sweep high" in signal_tg_name.lower() or "mss bearish" in signal_tg_name.lower() or "cisd bearish" in signal_tg_name.lower()
            
            is_volatile = last_bar.get('is_volatile', True)

            if is_volatile:
                if is_bullish_signal:
                    entry_price = price
                    ob_low = last_bar['bullish_ob_low']
                    conf_sl = last_bar['leg_low_1m']
                    sl1_price, sl2_price = np.nan, np.nan
                    
                    if not np.isnan(ob_low):
                        sl1_price = ob_low
                        if not np.isnan(conf_sl) and conf_sl < ob_low:
                            sl2_price = conf_sl
                    elif not np.isnan(conf_sl):
                        sl1_price = conf_sl
                    else:
                        sl1_price = last_bar['low'] - 15.0
                    
                    if not np.isnan(entry_price) and not np.isnan(sl1_price) and price > sl1_price:
                        sl_points = entry_price - sl1_price
                        if sl_points <= 5.0:
                            sl1_price = entry_price - 20.0
                            sl_points = 20.0
                            
                        if sl_points > MAX_SL_POINTS:
                            sl1_price = entry_price - MAX_SL_POINTS
                            sl_points = MAX_SL_POINTS
                            sl2_price = np.nan
                        
                        tp1_price = entry_price + sl_points * 3.0
                        
                        logger.signal(f"訊號觸發，發送 Telegram 即時監控通知 (ID: {TELEGRAM_SIGNAL_CHAT_ID}): {signal_tg_name}")
                        self._handle_signal_action("LONG", signal_level, signal_tg_name, entry_price, sl1_price, tp1_price, price, dt_str, trend_5m, sl2_price)
                    else:
                        logger.info(f"       🚫 [訊號過濾] 多頭條件不足: entry_price={entry_price}, sl1_price={sl1_price}, 現價={price} (現價需大於SL)")
                        
                elif is_bearish_signal:
                    entry_price = price
                    ob_high = last_bar['bearish_ob_high']
                    conf_sh = last_bar['leg_high_1m']
                    sl1_price, sl2_price = np.nan, np.nan
                    
                    if not np.isnan(ob_high):
                        sl1_price = ob_high
                        if not np.isnan(conf_sh) and conf_sh > ob_high:
                            sl2_price = conf_sh
                    elif not np.isnan(conf_sh):
                        sl1_price = conf_sh
                    else:
                        sl1_price = last_bar['high'] + 15.0
                    
                    if not np.isnan(entry_price) and not np.isnan(sl1_price) and price < sl1_price:
                        sl_points = sl1_price - entry_price
                        if sl_points <= 5.0:
                            sl1_price = entry_price + 20.0
                            sl_points = 20.0
                            
                        if sl_points > MAX_SL_POINTS:
                            sl1_price = entry_price + MAX_SL_POINTS
                            sl_points = MAX_SL_POINTS
                            sl2_price = np.nan
                        
                        tp1_price = entry_price - sl_points * 3.0
                        
                        logger.signal(f"訊號觸發，發送 Telegram 即時監控通知 (ID: {TELEGRAM_SIGNAL_CHAT_ID}): {signal_tg_name}")
                        self._handle_signal_action("SHORT", signal_level, signal_tg_name, entry_price, sl1_price, tp1_price, price, dt_str, trend_5m, sl2_price)
                    else:
                        logger.info(f"       🚫 [訊號過濾] 空頭條件不足: entry_price={entry_price}, sl1_price={sl1_price}, 現價={price} (現價需小於SL)")
            else:
                logger.info(f"       🚫 [訊號過濾] 波動率不足 (is_volatile=False)，略過本次訊號")

    def _handle_signal_action(self, direction: str, signal_level: int, signal_tg_name: str,
                               entry_price: float, sl_price: float, tp1_price: float,
                               price: float, dt_str: str, trend_5m: str, sl2_price: float = None):
        """依持倉狀態與訊號等級進行開倉、覆蓋、或加碼"""
        pos = self.active_position
        p_int = int(round(price))
        e_int = int(round(entry_price))
        sl_int = int(round(sl_price))
        tp_int = int(round(tp1_price))
        dir_label = "多單市價買" if direction == "LONG" else "空單市價賣"

        pure_signal_text = (
            f"觸發時間：{dt_str}\n"
            f"最新價格：{p_int}\n"
            f"大結構趨勢 (5M)：{trend_5m}\n"
            f"🚨 訊號類型：{signal_tg_name}\n"
        )
        if sl2_price is not None and not np.isnan(sl2_price):
            pure_signal_text += f"💡 {dir_label}：{e_int} | SL1：{sl_int} | SL2：{int(round(sl2_price))} | TP：{tp_int}"
        else:
            pure_signal_text += f"💡 {dir_label}：{e_int} | SL：{sl_int}  | TP：{tp_int}"
        send_telegram_notification(pure_signal_text, chat_id=TELEGRAM_SIGNAL_CHAT_ID)

        if pos is None:
            if abs(price - entry_price) > 300:
                skip_msg = f"⏸️ [極端跳空過濾] 現價 {p_int} 距離理論邊界過遠（>300點），略過本次下單。"
                logger.info(skip_msg)
                send_telegram_notification(skip_msg, chat_id=TELEGRAM_SETTLEMENT_CHAT_ID)
                return
                
            self._place_simulated_2stage_order(direction, entry_price, sl_price, tp1_price, signal_level=signal_level, lots=2, sl2_price=sl2_price)
            return

        if signal_level >= 2:
            if direction == pos['direction']:
                self._add_on_position(direction, signal_tg_name, entry_price, sl_price, tp1_price, dt_str, sl2_price)
            else:
                self._reverse_position(direction, signal_tg_name, entry_price, sl_price, tp1_price, dt_str, price, sl2_price)
            return

        logger.info(f"⏸️ 已持倉中（{pos['direction']}／{pos.get('signal_level', '?')}★開倉），偵測到 {signal_level}★ 新訊號但不符合覆蓋條件，略過本次訊號。")
        skip_text = (
            f"觸發時間：{dt_str}\n"
            f"最新價格：{p_int}\n"
            f"🚨 訊號類型：{signal_tg_name}\n"
            f"⏸️ 已持倉不開新倉"
        )
        send_telegram_notification(skip_text, chat_id=TELEGRAM_SETTLEMENT_CHAT_ID)

    def _add_on_position(self, direction: str, signal_tg_name: str, entry_price: float,
                          sl_price: float, tp1_price: float, dt_str: str, sl2_price: float = None):
        pos = self.active_position
        add_lots = 1

        if pos.get('stage') == 2:
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
        if sl2_price is not None and not np.isnan(sl2_price):
            pos['sl2'] = sl2_price
        else:
            pos.pop('sl2', None)
        pos['sl1_hit'] = False
        pos['tp1'] = tp1_price
        pos['signal_level'] = 2

        sl_str = f"SL1: {sl_int} | SL2: {int(round(sl2_price))}" if sl2_price is not None and not np.isnan(sl2_price) else f"SL: {sl_int}"
        remain_note = f"下次 TP1 將平 {preview_close} 口、留 {preview_remain} 口續抱" if preview_remain > 0 else f"下次 TP1 將全部 {preview_close} 口出場"
        msg = (
            f"🔼 <b>[SMC 加倉通知 - 2★訊號覆蓋1★持倉]</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"<b>觸發時間</b>：{dt_str}\n"
            f"<b>訊號類型</b>：{signal_tg_name}\n"
            f"<b>加碼部位</b>：+{add_lots} 口（{old_lots} → {new_lots} 口）\n"
            f"<b>更新 </b>{sl_str}｜<b>更新 TP1</b>：<code>{tp_int}</code>\n"
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
                
                if trade and trade.order:
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
            msg = {
                'action': 'Buy' if direction == "LONG" else 'Sell',
                'price': entry_price,
                'quantity': add_lots
            }
            logger.info(f"🔧 [模擬模式] 自動觸發加碼成交回報...")
            self._on_order_callback(sj.OrderState.FuturesDeal if hasattr(sj, 'OrderState') else 10, msg)

    def _reverse_position(self, direction: str, signal_tg_name: str, entry_price: float,
                           sl_price: float, tp1_price: float, dt_str: str, price: float, sl2_price: float = None):
        pos = self.active_position

        # [FIX 2026-08-05] 準確計算目前實際持倉口數
        # case A: stage=2（已部分平倉）→ 用 remaining_lots
        # case B: stage=1 但已觸發 SL1→ 實際剩 remaining_lots（lots - half）
        # case C: 一般 stage=1 → 用 lots
        if pos.get('stage') == 2 or pos.get('sl1_hit'):
            lots = pos.get('remaining_lots', pos.get('lots', 2) - math.ceil(pos.get('lots', 2) / 2))
        else:
            lots = pos.get('lots', 2)

        # [FIX 2026-08-05] 新開倉的口數 = 剛才平掉的口數（維持等量）
        reverse_lots = lots

        # [FIX 2026-08-06] 反手交易競態修正：只送平倉單，延遲開倉
        # 之前 _send_close_order + _place_simulated_2stage_order 連續執行，但 Shioaji
        # callback 異步回報不保證順序 → 新倉成交可能先到 → TG 開倉戰報比平倉先到。
        # 修復：把新倉參數暫存到 pos['pending_reverse']，在 _process_settlement_deal
        # 的 REVERSE 分支結算完後才開新倉，保證「平倉先 → 開倉後」。
        pos['pending_reverse'] = {
            'direction': direction,
            'entry_price': entry_price,
            'sl_price': sl_price,
            'tp1_price': tp1_price,
            'signal_level': 2,
            'lots': reverse_lots,
            'sl2_price': sl2_price,
        }
        logger.info(f"🔄 [反手準備] 平倉後將開 {direction} {reverse_lots} 口 | entry={int(round(entry_price))} SL={int(round(sl_price))} TP1={int(round(tp1_price))}")

        self._send_close_order(lots, 'REVERSE', f"🔄 偵測到反方向 2★ 訊號（{signal_tg_name}），強制平倉反手")

    def _place_simulated_2stage_order(self, direction: str, entry_price: float, sl_price: float,
                                       tp1_price: float, signal_level: int = 1, lots: int = 2, sl2_price: float = None):
        e_int = int(round(entry_price))
        sl_int = int(round(sl_price))
        tp_int = int(round(tp1_price))
        
        logger.info(f"🤖 [Shioaji 模擬下單觸發] 自動發起 {lots} 口小台 {direction} 委託 | 限價: {e_int}, SL: {sl_int}, TP1: {tp_int}")
        
        self.active_position = {
            'direction': direction,
            'entry_price': entry_price,
            'sl': sl_price,
            'sl2': sl2_price,
            'sl1_hit': False,
            'tp1': tp1_price,
            'stage': 1,
            'signal_level': signal_level,
            'lots': lots
        }
        
        if self.api is not None and self.contract is not None:
            try:
                act = sj.Action.Buy if direction == "LONG" else sj.Action.Sell
                order = self.api.Order(
                    action=act,
                    price=0,
                    quantity=lots,
                    order_type=sj.OrderType.ROD,
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
            msg = {
                'action': 'Buy' if direction == "LONG" else 'Sell',
                'price': entry_price,
                'quantity': lots
            }
            logger.info(f"🔧 [模擬模式] 自動觸發進場成交回報...")
            self._on_order_callback(sj.OrderState.FuturesDeal if hasattr(sj, 'OrderState') else 10, msg)

    def _calc_pnl(self, entry_price: float, exit_price: float, direction: str, lots: int) -> "tuple[float, float]":
        dir_mult = 1 if direction == 'LONG' else -1
        net_pts = (exit_price - entry_price) * dir_mult - ROUNDTRIP_SLIPPAGE_PTS
        net_pnl = net_pts * MINI_POINT_VALUE * lots - ROUNDTRIP_COMMISSION_PER_LOT * lots
        return net_pts, net_pnl

    def _init_balance_file(self):
        import json
        balance_file = os.path.join(os.path.dirname(__file__), "..", "data", "balance.json")
        try:
            os.makedirs(os.path.dirname(balance_file), exist_ok=True)
            if not os.path.exists(balance_file):
                with open(balance_file, "w") as f:
                    json.dump({"equity": 0, "last_update": ""}, f)
                logger.info("已建立餘額暫存檔，初始值 0")
        except Exception as e:
            logger.warning(f"初始化餘額暫存檔失敗: {e}")

    def _get_balance(self) -> float:
        import json
        balance_file = os.path.join(os.path.dirname(__file__), "..", "data", "balance.json")
        try:
            with open(balance_file, "r") as f:
                return json.load(f).get("equity", 0)
        except:
            return 0

    def _update_balance(self, pnl: float):
        import json
        balance_file = os.path.join(os.path.dirname(__file__), "..", "data", "balance.json")
        current = self._get_balance()
        new_balance = current + pnl
        with open(balance_file, "w") as f:
            json.dump({"equity": new_balance, "last_update": datetime.datetime.now().isoformat()}, f)
        logger.info(f"餘額更新: {current:+,.0f} -> {new_balance:+,.0f} NTD (本次 {pnl:+,.0f})")

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
        
        if stage_type in ['SL', 'FINAL', 'REVERSE']:
            self.closing_positions.append(pos)
            self.active_position = None
        
        if self.api is not None and self.contract is not None:
            try:
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
                sim_price = pos.get('tp1') if stage_type == 'TP1' else pos.get('sl') if stage_type in ['SL', 'SL1'] else pos.get('sl2', pos['entry_price']) if 'SL2' in reason else pos['entry_price']
            else:
                sim_price = pos.get('tp1') if stage_type == 'TP1' else pos.get('sl') if stage_type in ['SL', 'SL1'] else pos.get('sl2', pos['entry_price']) if 'SL2' in reason else pos['entry_price']
            
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
            if not pos.get('sl1_hit', False):
                is_sl_hit = (low <= pos['sl']) if pos['direction'] == 'LONG' else (high >= pos['sl'])
                if is_sl_hit:
                    if pos.get('sl2') and not np.isnan(pos['sl2']):
                        # 有 SL2，執行分批停損 (SL1)
                        close_lots = math.ceil(total_lots / 2)
                        pos['sl1_hit'] = True
                        self._send_close_order(close_lots, 'SL1', "🛑 觸及 SL1 停損價 (分批停損)")
                        return
                    else:
                        # 沒有 SL2，全數停損
                        self._send_close_order(total_lots, 'SL', "🛑 觸及 SL 停損價")
                        return
            else:
                # 已經觸發過 SL1，檢查是否觸發 SL2 (最終停損)
                if pos.get('sl2') and not np.isnan(pos['sl2']):
                    is_sl2_hit = (low <= pos['sl2']) if pos['direction'] == 'LONG' else (high >= pos['sl2'])
                    if is_sl2_hit:
                        remain_lots = pos.get('remaining_lots', total_lots - math.ceil(total_lots / 2))
                        self._send_close_order(remain_lots, 'FINAL', "🛑 觸及 SL2 停損價 (最終停損)")
                        return

            # 不論是否觸發過 SL1，只要仍在 stage 1，就可以檢查 TP1
            is_tp1_hit = (high >= pos['tp1']) if pos['direction'] == 'LONG' else (low <= pos['tp1'])
            if is_tp1_hit:
                # 若已觸發過 SL1，這裡就直接平掉剩餘部位 (以 FINAL 結算)
                if pos.get('sl1_hit', False):
                    remain_lots = pos.get('remaining_lots', total_lots - math.ceil(total_lots / 2))
                    self._send_close_order(remain_lots, 'FINAL', "🎯 觸及 TP1 目標 (剩餘部位全平)")
                    return
                else:
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
            
            is_volatile = last_bar.get('is_volatile', True)
            
            if pos['direction'] == 'LONG':
                if low <= pos['sl']:
                    reason, closed = "🛑 觸及進場保本價", True
                elif (last_bar['mss_bearish'] or last_bar['sweep_high']) and is_volatile:
                    reason, closed = "🚨 檢測到反向空頭 SMC 訊號", True
            else: # SHORT
                if high >= pos['sl']:
                    reason, closed = "🛑 觸及進場保本價", True
                elif (last_bar['mss_bullish'] or last_bar['sweep_low']) and is_volatile:
                    reason, closed = "🚨 檢測到反向多頭 SMC 訊號", True
                    
            if closed:
                remain_lots = pos.get('remaining_lots', total_lots - math.ceil(total_lots / 2))
                self._send_close_order(remain_lots, 'FINAL', reason)