import numpy as np
import pandas as pd
import datetime
from typing import List, Dict, Any, Tuple
from smc_trader.backtester import Backtester
from smc_trader.smc_detector import SMCDetector

class StatsValidator:
    """
    SMC 策略高階統計檢定與驗證模組。
    實作 MCPT、Bonferroni 校正、Walk-Forward 分析與 Bootstrap 信賴區間。
    """
    def __init__(self, default_alpha: float = 0.05):
        self.default_alpha = default_alpha

    def run_bootstrap_ci(self, trades: List[Dict[str, Any]], num_bootstrap: int = 2000) -> Dict[str, Any]:
        """
        Bootstrap (拔靴法) 預期收益信賴區間估計。
        重複抽樣以確定 95% 信賴區間下限是否大於 0。
        """
        if not trades:
            return {
                'low_ci': 0.0,
                'high_ci': 0.0,
                'mean_pnl': 0.0,
                'passed': False,
                'distribution': []
            }

        pnls = np.array([t['pnl'] for t in trades])
        n = len(pnls)
        
        # 進行重複隨機抽樣
        # 建立 shape 為 (num_bootstrap, n) 的隨機索引陣列
        rng = np.random.default_rng(42)
        bootstrap_indices = rng.choice(n, size=(num_bootstrap, n), replace=True)
        bootstrap_samples = pnls[bootstrap_indices]
        
        # 計算每次抽樣的平均交易損益
        bootstrap_means = np.mean(bootstrap_samples, axis=1)
        
        # 計算 95% 信賴區間 (2.5% 與 97.5% 分位數)
        low_ci = np.percentile(bootstrap_means, 2.5)
        high_ci = np.percentile(bootstrap_means, 97.5)
        mean_pnl = np.mean(pnls)
        
        # 通過條件：期望值 CI 下限大於 0
        passed = low_ci > 0.0
        
        return {
            'low_ci': round(float(low_ci), 1),
            'high_ci': round(float(high_ci), 1),
            'mean_pnl': round(float(mean_pnl), 1),
            'passed': bool(passed),
            'distribution': bootstrap_means.tolist()
        }

    def run_mcpt(self, trades: List[Dict[str, Any]], num_permutations: int = 1000) -> Dict[str, Any]:
        """
        MCPT (蒙地卡羅排列檢定)。
        隨機打亂交易的方向，重新計算 1000 次隨機總收益，
        以檢驗真實獲利是否具有顯著的統計學意義。
        """
        if not trades:
            return {
                'p_value': 1.0,
                'real_profit': 0.0,
                'passed': False,
                'distribution': []
            }

        # 提取每筆交易的實際點數變化 (不含方向)
        raw_points = np.array([abs(t['gross_points']) for t in trades])
        net_pnls = np.array([t['pnl'] for t in trades])
        real_profit = np.sum(net_pnls)
        
        n_trades = len(trades)
        random_profits = []
        rng = np.random.default_rng(42)
        
        for _ in range(num_permutations):
            random_dirs = rng.choice([-1, 1], size=n_trades)
            rand_gross_points = raw_points * random_dirs
            rand_net_points = rand_gross_points - 2.0
            rand_pnl = (rand_net_points * 50.0) - 100.0
            random_profits.append(np.sum(rand_pnl))
            
        random_profits = np.array(random_profits)
        
        # 計算 p-value: 隨機獲利大於等於真實獲利的比例
        p_value = np.sum(random_profits >= real_profit) / num_permutations
        
        # 通過條件: p-value < 0.05
        passed = p_value < self.default_alpha
        
        return {
            'p_value': round(float(p_value), 4),
            'real_profit': round(float(real_profit), 1),
            'passed': bool(passed),
            'distribution': random_profits.tolist()
        }

    def run_bonferroni(self, p_value: float, num_tests: int = 10) -> Dict[str, Any]:
        """
        Bonferroni (邦費羅尼校正)。
        在最佳化測試了多組參數時，嚴格降低顯著性水準門檻。
        """
        adjusted_alpha = self.default_alpha / num_tests
        passed = p_value < adjusted_alpha
        
        return {
            'original_alpha': self.default_alpha,
            'adjusted_alpha': round(adjusted_alpha, 4),
            'p_value': p_value,
            'num_tests': num_tests,
            'passed': bool(passed)
        }

    def run_walk_forward(self, df_1m: pd.DataFrame, df_5m: pd.DataFrame, 
                          num_folds: int = 3, train_ratio: float = 0.7,
                          exec_type: str = "B", t_start=None, t_end=None) -> Dict[str, Any]:
        """
        Walk-Forward (向前推進分析)。
        將數據滾動劃分，在 IS 上尋找最佳 R:R 參數，在 OOS 上評估真實泛化表現。
        """
        if t_start is None: t_start = datetime.time(0, 0)
        if t_end is None: t_end = datetime.time(23, 59)
        
        n = len(df_1m)
        fold_size = int(n / (num_folds + 1))
        
        is_profits = []
        oos_profits = []
        rr_candidates = [2.5, 3.5, 4.8, 5.5]
        detector = SMCDetector(pullback_buffer_pts=20.0, atr_mult=0.90)
        
        def run_custom_engine(df_proc, rr_val):
            trades = []
            position = None
            pending_order = None
            balance = 1000000.0
            slippage, commission, point_value = 1.0, 50.0, 50.0
            max_sl_pts = 80.0
            m_len = len(df_proc)
            
            for i in range(m_len - 1):
                bar = df_proc.iloc[i].to_dict()
                ts = bar['ts']
                close, high, low, trend_5m = bar['close'], bar['high'], bar['low'], bar['trend_5m']
                bar_time = ts.time()
                
                if position is not None:
                    dir_m = 1 if position['direction'] == 'LONG' else -1
                    closed, exit_p = False, 0.0
                    if position['direction'] == 'LONG':
                        if low <= position['sl']: exit_p, closed = position['sl'], True
                        elif high >= position['tp']: exit_p, closed = position['tp'], True
                    else:
                        if high >= position['sl']: exit_p, closed = position['sl'], True
                        elif low <= position['tp']: exit_p, closed = position['tp'], True
                    if not closed and (datetime.time(13, 30) <= bar_time <= datetime.time(13, 45)):
                        exit_p, closed = close, True
                    if closed:
                        net_pts = (exit_p - position['entry_price']) * dir_m - (2.0 * slippage)
                        net_pnl = net_pts * point_value - (2.0 * commission)
                        balance += net_pnl
                        trades.append({'pnl': net_pnl})
                        position = None
                        
                if exec_type.startswith("A") and pending_order is not None:
                    po = pending_order
                    trig = (low <= po['entry_price']) if po['direction'] == 'LONG' else (high >= po['entry_price'])
                    if trig:
                        position = {'direction': po['direction'], 'entry_price': po['entry_price'], 'sl': po['sl'], 'tp': po['tp']}
                        pending_order = None
                        continue
                    else:
                        po['bars_pending'] += 1
                        if po['bars_pending'] >= 10: pending_order = None
                        
                if position is not None or pending_order is not None: continue
                if not (t_start <= bar_time <= t_end): continue
                if 'is_volatile' in bar and not bar['is_volatile']: continue
                
                if trend_5m == "BULLISH":
                    if bar['mss_bullish'] or bar['cisd_bullish'] or bar['sweep_low']:
                        sl_price = bar['bullish_ob_low'] if not np.isnan(bar['bullish_ob_low']) else (bar['confirmed_sl_1m'] if not np.isnan(bar['confirmed_sl_1m']) else low - 15.0)
                        if exec_type.startswith("A"):
                            e_price = bar['bullish_ob_high'] if not np.isnan(bar['bullish_ob_high']) else bar['bullish_fvg_high']
                            if not np.isnan(e_price) and close > e_price:
                                sl_pts = e_price - sl_price
                                if sl_pts <= 5.0: sl_price, sl_pts = e_price - 20.0, 20.0
                                if sl_pts > max_sl_pts: sl_price, sl_pts = e_price - max_sl_pts, max_sl_pts
                                pending_order = {'direction': 'LONG', 'entry_price': e_price, 'sl': sl_price, 'tp': e_price + sl_pts * rr_val, 'bars_pending': 0}
                        else:
                            next_open = df_proc.iloc[i+1]['open']
                            if next_open > sl_price:
                                sl_pts = next_open - sl_price
                                if sl_pts <= 5.0: sl_price, sl_pts = next_open - 20.0, 20.0
                                if sl_pts > max_sl_pts: sl_price, sl_pts = next_open - max_sl_pts, max_sl_pts
                                position = {'direction': 'LONG', 'entry_price': next_open, 'sl': sl_price, 'tp': next_open + sl_pts * rr_val}
                elif trend_5m == "BEARISH":
                    if bar['mss_bearish'] or bar['cisd_bearish'] or bar['sweep_high']:
                        sl_price = bar['bearish_ob_high'] if not np.isnan(bar['bearish_ob_high']) else (bar['confirmed_sh_1m'] if not np.isnan(bar['confirmed_sh_1m']) else high + 15.0)
                        if exec_type.startswith("A"):
                            e_price = bar['bearish_ob_low'] if not np.isnan(bar['bearish_ob_low']) else bar['bearish_fvg_low']
                            if not np.isnan(e_price) and close < e_price:
                                sl_pts = sl_price - e_price
                                if sl_pts <= 5.0: sl_price, sl_pts = e_price + 20.0, 20.0
                                if sl_pts > max_sl_pts: sl_price, sl_pts = e_price + max_sl_pts, max_sl_pts
                                pending_order = {'direction': 'SHORT', 'entry_price': e_price, 'sl': sl_price, 'tp': e_price - sl_pts * rr_val, 'bars_pending': 0}
                        else:
                            next_open = df_proc.iloc[i+1]['open']
                            if next_open < sl_price:
                                sl_pts = sl_price - next_open
                                if sl_pts <= 5.0: sl_price, sl_pts = next_open + 20.0, 20.0
                                if sl_pts > max_sl_pts: sl_price, sl_pts = next_open + max_sl_pts, max_sl_pts
                                position = {'direction': 'SHORT', 'entry_price': next_open, 'sl': sl_price, 'tp': next_open - sl_pts * rr_val}
                                
            return sum([t['pnl'] for t in trades]) if trades else 0.0

        for fold in range(num_folds):
            start_idx = fold * fold_size
            train_end_idx = start_idx + int(fold_size * 2 * train_ratio)
            test_end_idx = start_idx + fold_size * 2
            if test_end_idx > n: break
            
            df_train_1m = df_1m.iloc[start_idx:train_end_idx].reset_index(drop=True)
            df_test_1m = df_1m.iloc[train_end_idx:test_end_idx].reset_index(drop=True)
            
            train_ts_start, train_ts_end = df_train_1m['ts'].min(), df_train_1m['ts'].max()
            test_ts_start, test_ts_end = df_test_1m['ts'].min(), df_test_1m['ts'].max()
            
            df_train_5m = df_5m[(df_5m['ts'] >= train_ts_start) & (df_5m['ts'] <= train_ts_end)].reset_index(drop=True)
            df_test_5m = df_5m[(df_5m['ts'] >= test_ts_start) & (df_5m['ts'] <= test_ts_end)].reset_index(drop=True)
            
            df_train_proc = detector.process_1m_signals(df_train_1m, detector.process_5m_structure(df_train_5m))
            df_test_proc = detector.process_1m_signals(df_test_1m, detector.process_5m_structure(df_test_5m))
            
            best_rr = 4.8
            best_is_profit = -float('inf')
            
            for rr in rr_candidates:
                pnl = run_custom_engine(df_train_proc, rr)
                if pnl > best_is_profit:
                    best_is_profit = pnl
                    best_rr = rr
                    
            best_oos_profit = run_custom_engine(df_test_proc, best_rr)
            is_profits.append(best_is_profit)
            oos_profits.append(best_oos_profit)
            
        avg_is = np.mean(is_profits) if is_profits else 0.0
        avg_oos = np.mean(oos_profits) if oos_profits else 0.0
        wfe = (avg_oos / avg_is) * 100 if avg_is > 0 else (100.0 if avg_oos > 0 else 0.0)
        passed = wfe >= 50.0
        
        return {
            'avg_is_profit': round(float(avg_is), 1),
            'avg_oos_profit': round(float(avg_oos), 1),
            'wfe': round(wfe, 2),
            'passed': bool(passed)
        }
