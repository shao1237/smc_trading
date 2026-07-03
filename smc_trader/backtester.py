import pandas as pd
import numpy as np
import datetime
from typing import List, Dict, Any, Optional, Tuple
from smc_trader.config import (
    INITIAL_CAPITAL, FUTURES_POINT_VALUE, SLIPPAGE_POINTS,
    COMMISSION_FEE, DEFAULT_RR, MAX_PENDING_BARS,
    INTRADAY_EXIT_START, INTRADAY_EXIT_END, MAX_SL_POINTS
)

class Backtester:
    """
    SMC+SNR 策略回測引擎。
    逐 K 棒模擬限價掛單、回踩成交、止損停利與資產曲線更新。
    """
    def __init__(self, initial_capital: float = INITIAL_CAPITAL,
                 point_value: float = FUTURES_POINT_VALUE,
                 slippage: float = SLIPPAGE_POINTS,
                 commission: float = COMMISSION_FEE,
                 default_rr: float = DEFAULT_RR,
                 max_pending_bars: int = MAX_PENDING_BARS,
                 max_sl_points: float = MAX_SL_POINTS):
        self.initial_capital = initial_capital
        self.point_value = point_value
        self.slippage = slippage
        self.commission = commission
        self.default_rr = default_rr
        self.max_pending_bars = max_pending_bars
        self.max_sl_points = max_sl_points
        
        # 解析當沖平倉時間
        start_h, start_m = map(int, INTRADAY_EXIT_START.split(':'))
        end_h, end_m = map(int, INTRADAY_EXIT_END.split(':'))
        self.isq_start = datetime.time(start_h, start_m)
        self.isq_end = datetime.time(end_h, end_m)
        
        self.reset()

    def reset(self):
        """重置回測狀態"""
        self.balance = self.initial_capital
        self.equity_curve: List[Dict[str, Any]] = []
        self.trades: List[Dict[str, Any]] = []
        
        # 當前掛單狀態 (Pending Order)
        self.pending_order: Optional[Dict[str, Any]] = None
        
        # 當前持倉狀態 (Open Position)
        self.position: Optional[Dict[str, Any]] = None

    def run(self, df_1m: pd.DataFrame) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
        """
        執行逐 K 線回測。
        """
        self.reset()
        
        df = df_1m.copy()
        df['equity'] = self.initial_capital
        
        n = len(df)
        
        # 為了更方便在迴圈中存取，將 dataframe 轉成 dictionary array
        bars = df.to_dict('records')
        
        for i in range(n):
            bar = bars[i]
            ts = bar['ts']
            close = bar['close']
            high = bar['high']
            low = bar['low']
            trend_5m = bar['trend_5m']
            
            # 1. 更新帳戶淨值 (Equity)
            current_equity = self.balance
            if self.position is not None:
                # 浮動盈虧計算
                pos_dir = 1 if self.position['direction'] == 'LONG' else -1
                floating_profit = (close - self.position['entry_price']) * pos_dir * self.point_value
                current_equity += floating_profit
            
            self.equity_curve.append({
                'ts': ts,
                'equity': current_equity,
                'balance': self.balance
            })
            df.at[i, 'equity'] = current_equity

            # 1.5 檢查當沖時間強平機制 (Intraday Square-off)
            bar_time = ts.time()
            if self.isq_start <= bar_time <= self.isq_end:
                # 若有持倉，強制以收盤價平倉
                if self.position is not None:
                    self._force_close_position(bar, i, "ISQ")
                # 若有掛單，取消掛單
                if self.pending_order is not None:
                    self.pending_order = None
                continue  # 強平期間不建立新委託

            # 2. 處理持倉中的停損停利判定
            if self.position is not None:
                self._check_position_exit(bar, i)
                continue  # 持倉時不建立新委託

            # 3. 處理掛單是否成交或過期判定
            if self.pending_order is not None:
                self._check_pending_order(bar, i)
                # 如果掛單在這一根 K 線成交了，它會轉化為持倉，我們就不在同一根 K 線再建立新委託
                if self.position is not None:
                    continue

            # 4. 尋找交易訊號
            # 限制只能在 09:00 - 11:00 之間建立新掛單委託
            bar_time = ts.time()
            if not (datetime.time(9, 0) <= bar_time <= datetime.time(11, 0)):
                continue

            # 必須符合大趨勢 (5M BOS)
            if trend_5m == "BULLISH":
                # 做多條件：MSS Bullish 且 CISD Bullish 同時出現，或在最近 3 根 K 線內出現
                # 為了避免遺漏，我們檢查當前或前 2 根的信號
                is_signal = bar['mss_bullish'] or bar['cisd_bullish']
                if is_signal:
                    # 決定進場限價與止損
                    # 我們使用最近的 FVG 或 OB 來定價
                    # 優先使用 OB (Order Block) 的上沿，其次為 FVG 上沿
                    entry_price = np.nan
                    ob_high = bar['bullish_ob_high']
                    fvg_high = bar['bullish_fvg_high']
                    ob_low = bar['bullish_ob_low']
                    
                    if not np.isnan(ob_high):
                        entry_price = ob_high
                    elif not np.isnan(fvg_high):
                        entry_price = fvg_high
                    
                    # 停損設在 OB 的下沿
                    sl_price = ob_low if not np.isnan(ob_low) else bar['confirmed_sl_1m']
                    if np.isnan(sl_price):
                        sl_price = low - 15.0  # 緩衝
                        
                    # 緩衝與止損點數合理性檢查
                    sl_points = entry_price - sl_price
                    if sl_points <= 5.0:
                        sl_price = entry_price - 20.0 # 強制設定最小停損為 20 點
                        sl_points = 20.0
                    
                    # 限制單筆最大止損點數上限
                    if sl_points > self.max_sl_points:
                        sl_price = entry_price - self.max_sl_points
                        sl_points = self.max_sl_points
                    
                    # 判定掛單有效性 (當前價格需高於進場限價，表示未來是「回踩」成交)
                    if not np.isnan(entry_price) and close > entry_price:
                        # 計算停利價 (基於 R:R)
                        tp_price = entry_price + sl_points * self.default_rr
                        
                        self.pending_order = {
                            'direction': 'LONG',
                            'entry_price': entry_price,
                            'sl': sl_price,
                            'tp': tp_price,
                            'setup_time': ts,
                            'setup_idx': i,
                            'bars_pending': 0
                        }

            elif trend_5m == "BEARISH":
                # 做空條件
                is_signal = bar['mss_bearish'] or bar['cisd_bearish']
                if is_signal:
                    entry_price = np.nan
                    ob_low = bar['bearish_ob_low']
                    fvg_low = bar['bearish_fvg_low']
                    ob_high = bar['bearish_ob_high']
                    
                    if not np.isnan(ob_low):
                        entry_price = ob_low
                    elif not np.isnan(fvg_low):
                        entry_price = fvg_low
                    
                    # 停損設在 OB 的上沿
                    sl_price = ob_high if not np.isnan(ob_high) else bar['confirmed_sh_1m']
                    if np.isnan(sl_price):
                        sl_price = high + 15.0
                        
                    sl_points = sl_price - entry_price
                    if sl_points <= 5.0:
                        sl_price = entry_price + 20.0
                        sl_points = 20.0
                    
                    # 限制單筆最大止損點數上限
                    if sl_points > self.max_sl_points:
                        sl_price = entry_price + self.max_sl_points
                        sl_points = self.max_sl_points

                    # 判定掛單有效性 (當前價格需低於進場限價，表示未來是「向上回踩」成交)
                    if not np.isnan(entry_price) and close < entry_price:
                        tp_price = entry_price - sl_points * self.default_rr
                        
                        self.pending_order = {
                            'direction': 'SHORT',
                            'entry_price': entry_price,
                            'sl': sl_price,
                            'tp': tp_price,
                            'setup_time': ts,
                            'setup_idx': i,
                            'bars_pending': 0
                        }

        return self.trades, df

    def _check_pending_order(self, bar: Dict[str, Any], current_idx: int):
        """檢查掛單是否成交或過期"""
        po = self.pending_order
        if po is None:
            return
            
        low = bar['low']
        high = bar['high']
        ts = bar['ts']
        
        # 判定是否回踩成交
        triggered = False
        if po['direction'] == 'LONG':
            # 限價做多：價格的低點跌破或觸及限價，代表買入成交
            if low <= po['entry_price']:
                triggered = True
        else:
            # 限價做空：價格的高點升破或觸及限價，代表賣出成交
            if high >= po['entry_price']:
                triggered = True
                
        if triggered:
            # 轉換為持倉
            self.position = {
                'direction': po['direction'],
                'entry_price': po['entry_price'],
                'sl': po['sl'],
                'tp': po['tp'],
                'entry_time': ts,
                'entry_idx': current_idx
            }
            self.pending_order = None
            return

        # 累加掛單等待的 K 線數量
        po['bars_pending'] += 1
        if po['bars_pending'] >= self.max_pending_bars:
            # 掛單過期取消
            self.pending_order = None

    def _check_position_exit(self, bar: Dict[str, Any], current_idx: int):
        """檢查持倉是否到達停損或停利"""
        pos = self.position
        if pos is None:
            return
            
        low = bar['low']
        high = bar['high']
        ts = bar['ts']
        
        exit_price = 0.0
        exit_type = ""
        closed = False
        
        if pos['direction'] == 'LONG':
            # 檢查停損是否被觸發
            if low <= pos['sl']:
                exit_price = pos['sl']
                exit_type = "SL"
                closed = True
            # 檢查停利是否被觸發 (最悲觀狀況：同根 K 線觸發時以 SL 為主)
            elif high >= pos['tp']:
                exit_price = pos['tp']
                exit_type = "TP"
                closed = True
        else:
            # SHORT
            if high >= pos['sl']:
                exit_price = pos['sl']
                exit_type = "SL"
                closed = True
            elif low <= pos['tp']:
                exit_price = pos['tp']
                exit_type = "TP"
                closed = True
                
        if closed:
            # 計算淨盈虧 (納入手續費與滑價)
            # 單邊滑價 = slippage, 來回 = 2 * slippage
            # 單邊手續費 = commission, 來回 = 2 * commission
            dir_mult = 1 if pos['direction'] == 'LONG' else -1
            
            # 滑價的影響：多單被止損/止盈，成交價會更差；
            # 這裡我們將滑價直接扣減在點數盈虧中
            gross_points = (exit_price - pos['entry_price']) * dir_mult
            
            # 納入滑價後的實際點數盈虧
            # 做多：進場買在 entry+slippage，出場賣在 exit-slippage。因此點數減去 2 * slippage
            net_points = gross_points - (2.0 * self.slippage)
            
            # 金額盈虧
            gross_pnl = net_points * self.point_value
            net_pnl = gross_pnl - (2.0 * self.commission)
            
            # 更新餘額
            self.balance += net_pnl
            
            # 記錄交易
            trade_record = {
                'direction': pos['direction'],
                'entry_time': pos['entry_time'],
                'entry_price': pos['entry_price'],
                'exit_time': ts,
                'exit_price': exit_price,
                'exit_type': exit_type,
                'gross_points': gross_points,
                'net_points': net_points,
                'pnl': net_pnl,
                'balance_after': self.balance,
                'rr_achieved': abs(net_points / (pos['entry_price'] - pos['sl'])) if (pos['entry_price'] - pos['sl']) != 0 else 0
            }
            self.trades.append(trade_record)
            self.position = None

    def _force_close_position(self, bar: Dict[str, Any], current_idx: int, exit_type: str):
        """強制以收盤價平倉持倉"""
        pos = self.position
        if pos is None:
            return
            
        close = bar['close']
        ts = bar['ts']
        
        dir_mult = 1 if pos['direction'] == 'LONG' else -1
        gross_points = (close - pos['entry_price']) * dir_mult
        net_points = gross_points - (2.0 * self.slippage)
        gross_pnl = net_points * self.point_value
        net_pnl = gross_pnl - (2.0 * self.commission)
        
        self.balance += net_pnl
        
        trade_record = {
            'direction': pos['direction'],
            'entry_time': pos['entry_time'],
            'entry_price': pos['entry_price'],
            'exit_time': ts,
            'exit_price': close,
            'exit_type': exit_type,
            'gross_points': gross_points,
            'net_points': net_points,
            'pnl': net_pnl,
            'balance_after': self.balance,
            'rr_achieved': abs(net_points / (pos['entry_price'] - pos['sl'])) if (pos['entry_price'] - pos['sl']) != 0 else 0
        }
        self.trades.append(trade_record)
        self.position = None
