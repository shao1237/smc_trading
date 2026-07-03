import numpy as np
import pandas as pd
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
        # 用來模擬隨機買賣方向下的損益
        raw_points = np.array([abs(t['gross_points']) for t in trades])
        # 我們需要扣減的滑價點數 (每筆 2 * slippage) 與手續費
        # 來計算真實點數
        net_pnls = np.array([t['pnl'] for t in trades])
        real_profit = np.sum(net_pnls)
        
        # 為了模擬隨機方向
        # 每次隨機分配 +1 (LONG) 或 -1 (SHORT) 給每筆交易點數，
        # 並扣除手續費及滑價
        n_trades = len(trades)
        random_profits = []
        
        rng = np.random.default_rng(42)
        
        for _ in range(num_permutations):
            # 隨機生成 +1 或 -1
            random_dirs = rng.choice([-1, 1], size=n_trades)
            # 計算隨機交易盈虧點數
            rand_gross_points = raw_points * random_dirs
            # 扣減單邊滑價與手續費的點數等值 (大台 1點=200元，單邊手續費50元=0.25點，單邊滑價2點)
            # 在 backtester 中，手續費是 50 元/單邊，滑價 2 點/單邊
            # 故來回成本 = 4 點滑價 + 100 元 (0.5 點大台價值) = 4.5 點
            rand_net_points = rand_gross_points - 4.0 # 扣減來回滑價
            rand_pnl = (rand_net_points * 200.0) - 100.0 # 扣減來回手續費 100 NTD
            
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
                         num_folds: int = 3, train_ratio: float = 0.7) -> Dict[str, Any]:
        """
        Walk-Forward (向前推進分析)。
        將數據滾動劃分，在 IS 上尋找最佳 R:R 參數，在 OOS 上評估真實泛化表現。
        """
        n = len(df_1m)
        fold_size = int(n / (num_folds + 1))
        
        is_profits = []
        oos_profits = []
        
        # 測試最佳化的 R:R 候選參數
        rr_candidates = [1.5, 2.0, 2.5, 3.0, 3.5]
        
        # 實體化回測引擎
        detector = SMCDetector()
        
        print("開始向前推進分析 (Walk-Forward Analysis)...")
        
        for fold in range(num_folds):
            # 劃分時間範圍
            # 例如 fold 0: 訓練集 [0 to 2*fold_size]，測試集 [2*fold_size to 3*fold_size]
            # fold 1: 訓練集 [fold_size to 3*fold_size]，測試集 [3*fold_size to 4*fold_size]
            start_idx = fold * fold_size
            train_end_idx = start_idx + int(fold_size * 2 * train_ratio)
            test_end_idx = start_idx + fold_size * 2
            
            if test_end_idx > n:
                break
                
            df_train_1m = df_1m.iloc[start_idx:train_end_idx].reset_index(drop=True)
            df_test_1m = df_1m.iloc[train_end_idx:test_end_idx].reset_index(drop=True)
            
            # 對訓練集和測試集提取對齊的 5M 結構
            # 為了簡化，在此直接用 index 進行過濾
            train_ts_start, train_ts_end = df_train_1m['ts'].min(), df_train_1m['ts'].max()
            test_ts_start, test_ts_end = df_test_1m['ts'].min(), df_test_1m['ts'].max()
            
            df_train_5m = df_5m[(df_5m['ts'] >= train_ts_start) & (df_5m['ts'] <= train_ts_end)].reset_index(drop=True)
            df_test_5m = df_5m[(df_5m['ts'] >= test_ts_start) & (df_5m['ts'] <= test_ts_end)].reset_index(drop=True)
            
            # 計算特徵
            df_train_proc = detector.process_1m_signals(df_train_1m, detector.process_5m_structure(df_train_5m))
            df_test_proc = detector.process_1m_signals(df_test_1m, detector.process_5m_structure(df_test_5m))
            
            # 1. 在 In-Sample (IS) 上最佳化參數
            best_rr = 2.5
            best_is_profit = -float('inf')
            
            for rr in rr_candidates:
                backtester = Backtester(default_rr=rr)
                trades, _ = backtester.run(df_train_proc)
                pnl = sum([t['pnl'] for t in trades]) if trades else 0.0
                if pnl > best_is_profit:
                    best_is_profit = pnl
                    best_rr = rr
            
            # 2. 將最佳參數套用到 Out-Of-Sample (OOS)
            backtester_oos = Backtester(default_rr=best_rr)
            trades_oos, _ = backtester_oos.run(df_test_proc)
            best_oos_profit = sum([t['pnl'] for t in trades_oos]) if trades_oos else 0.0
            
            is_profits.append(best_is_profit)
            oos_profits.append(best_oos_profit)
            
            print(f"  Fold {fold+1}: IS 最優 R:R={best_rr} (獲利: {best_is_profit:,.0f} NTD) | OOS 實戰獲利: {best_oos_profit:,.0f} NTD")
            
        avg_is = np.mean(is_profits) if is_profits else 0.0
        avg_oos = np.mean(oos_profits) if oos_profits else 0.0
        
        # 計算向前推進效率 WFE
        # WFE = avg_oos / avg_is
        # 為了避免除以 0，且符合實踐：
        wfe = (avg_oos / avg_is) * 100 if avg_is > 0 else (100.0 if avg_oos > 0 else 0.0)
        passed = wfe >= 50.0
        
        return {
            'avg_is_profit': round(float(avg_is), 1),
            'avg_oos_profit': round(float(avg_oos), 1),
            'wfe': round(wfe, 2),
            'passed': bool(passed),
            'folds_data': {
                'is_profits': is_profits,
                'oos_profits': oos_profits
            }
        }
