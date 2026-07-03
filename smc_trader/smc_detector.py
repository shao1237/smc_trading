import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

class SMCDetector:
    """
    SMC & SNR 策略特徵檢測器。
    負責辨識 5M 結構的 BOS 趨勢，以及 1M 結構中的 Swing High/Low、Liquidity Sweep、MSS、FVG、OB 和 CISD。
    所有特徵的計算都符合「無未來函數 (No Look-ahead Bias)」的原則。
    """
    def __init__(self, swing_window_5m: int = 5, swing_window_1m: int = 3, 
                 volume_ma_period: int = 20, volume_mult: float = 1.2):
        self.swing_window_5m = swing_window_5m
        self.swing_window_1m = swing_window_1m
        self.volume_ma_period = volume_ma_period
        self.volume_mult = volume_mult

    def detect_swings(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """
        滾動檢測 Swing High 與 Swing Low。
        注意：第 i 根 K 棒的 Swing 狀態，只有在第 i + window 根 K 棒收盤時才能被「確認」。
        """
        df = df.copy()
        df['is_swing_high'] = False
        df['is_swing_low'] = False
        df['swing_high_val'] = np.nan
        df['swing_low_val'] = np.nan

        highs = df['high'].values
        lows = df['low'].values
        n = len(df)

        for i in range(window, n - window):
            val_h = highs[i]
            val_l = lows[i]
            
            # Swing High: 左右各 window 根都小於它
            is_sh = True
            for w in range(1, window + 1):
                if highs[i - w] >= val_h or highs[i + w] > val_h:
                    is_sh = False
                    break
            
            # Swing Low: 左右各 window 根都大於它
            is_sl = True
            for w in range(1, window + 1):
                if lows[i - w] <= val_l or lows[i + w] < val_l:
                    is_sl = False
                    break

            if is_sh:
                df.at[i, 'is_swing_high'] = True
                df.at[i, 'swing_high_val'] = val_h
            if is_sl:
                df.at[i, 'is_swing_low'] = True
                df.at[i, 'swing_low_val'] = val_l

        return df

    def process_5m_structure(self, df_5m: pd.DataFrame) -> pd.DataFrame:
        """
        在 5M K 線上辨識 Swing Points 與 BOS (Break of Structure)。
        BOS 定義：實體 K 棒收盤突破前一個「已確認的」Swing High 或 Swing Low。
        """
        df_5m = self.detect_swings(df_5m, self.swing_window_5m)
        
        # 用於追踪目前已確認的 Swing High/Low
        # 因為只有在當前索引 t，我們才能確認 t - swing_window_5m 的 Swing 點
        # 故回測時我們只能在第 t 步，使用索引 <= t - swing_window_5m 已經確認的 Swing 值
        df_5m['confirmed_swing_high'] = np.nan
        df_5m['confirmed_swing_low'] = np.nan
        df_5m['trend_5m'] = "NONE" # NONE, BULLISH, BEARISH

        last_sh = np.nan
        last_sl = np.nan
        current_trend = "NONE"

        highs = df_5m['high'].values
        lows = df_5m['low'].values
        closes = df_5m['close'].values
        is_sh_arr = df_5m['is_swing_high'].values
        is_sl_arr = df_5m['is_swing_low'].values

        confirmed_sh_arr = np.full(len(df_5m), np.nan)
        confirmed_sl_arr = np.full(len(df_5m), np.nan)
        trend_arr = []

        for i in range(len(df_5m)):
            # 檢索是否有新的 Swing 點在 i - swing_window_5m 被確認
            confirm_idx = i - self.swing_window_5m
            if confirm_idx >= 0:
                if is_sh_arr[confirm_idx]:
                    last_sh = highs[confirm_idx]
                if is_sl_arr[confirm_idx]:
                    last_sl = lows[confirm_idx]

            confirmed_sh_arr[i] = last_sh
            confirmed_sl_arr[i] = last_sl

            # 檢測 BOS (實體收盤價突破)
            # 只有當前收盤價與「在此之前已確認」的 last_sh/last_sl 做比較
            if not np.isnan(last_sh) and closes[i] > last_sh:
                current_trend = "BULLISH" # 向上 BOS
            elif not np.isnan(last_sl) and closes[i] < last_sl:
                current_trend = "BEARISH" # 向下 BOS

            trend_arr.append(current_trend)

        df_5m['confirmed_swing_high'] = confirmed_sh_arr
        df_5m['confirmed_swing_low'] = confirmed_sl_arr
        df_5m['trend_5m'] = trend_arr

        return df_5m

    def process_1m_signals(self, df_1m: pd.DataFrame, df_5m_processed: pd.DataFrame) -> pd.DataFrame:
        """
        在 1M K 線上辨識 Liquidity Sweep、MSS、FVG、OB 和 CISD，並融入對齊的 5M 大結構趨勢。
        """
        # 1. 計算 1M Swing High/Low
        df_1m = self.detect_swings(df_1m, self.swing_window_1m)
        
        # 2. 將 5M 大趨勢對齊到 1M
        # 為了嚴格防範 Look-ahead bias，1M 時間為 t 時，只能使用已經收盤的 5M K 線
        # 例如，若 1M 時間為 09:04:00，對應已收盤的 5M K 線是 09:00:00 (包含 09:00~09:04 的 K 棒)
        # 我們先對 5M 資料建立以「結束時間」為 Key 的對照表
        # 通常 Resample 得到的 5M 時間戳如 09:00:00，其代表 [09:00, 09:04] 這段期間
        # 該 5M K 線在 09:05:00 之後的 1M K 線才能被正式存取其狀態
        df_5m_lookup = df_5m_processed.copy()
        # 計算此根 5M K 線收盤的 1M 時間 (5M 標記 + 5分鐘)
        df_5m_lookup['close_1m_ts'] = df_5m_lookup['ts'] + pd.Timedelta(minutes=5)
        
        # 建立時間對應字典
        trend_dict = dict(zip(df_5m_lookup['close_1m_ts'], df_5m_lookup['trend_5m']))
        
        # 對 1M DataFrame 進行前向填充 5M 趨勢
        df_1m['trend_5m'] = "NONE"
        current_5m_trend = "NONE"
        trend_5m_list = []
        for ts in df_1m['ts']:
            # 若此時剛好有 5M K 線收盤，更新大結構狀態
            if ts in trend_dict:
                current_5m_trend = trend_dict[ts]
            trend_5m_list.append(current_5m_trend)
        df_1m['trend_5m'] = trend_5m_list

        # 3. 計算 Volume MA 用於檢測爆量
        df_1m['vol_ma'] = df_1m['volume'].rolling(window=self.volume_ma_period).mean()
        df_1m['is_vol_spike'] = df_1m['volume'] >= (df_1m['vol_ma'] * self.volume_mult)

        # 4. 檢測 1M 結構信號與指標
        df_1m['sweep_low'] = False   # 買方流動性掠奪
        df_1m['sweep_high'] = False  # 賣方流動性掠奪
        df_1m['mss_bullish'] = False  # 結構向上轉換 (CHoCH)
        df_1m['mss_bearish'] = False  # 結構向下轉換 (CHoCH)
        
        # 為了儲存 OB 與 FVG 區間
        df_1m['bullish_ob_low'] = np.nan
        df_1m['bullish_ob_high'] = np.nan
        df_1m['bearish_ob_low'] = np.nan
        df_1m['bearish_ob_high'] = np.nan

        df_1m['bullish_fvg_low'] = np.nan
        df_1m['bullish_fvg_high'] = np.nan
        df_1m['bearish_fvg_low'] = np.nan
        df_1m['bearish_fvg_high'] = np.nan
        
        df_1m['cisd_bullish'] = False  # 價格交付向上改變
        df_1m['cisd_bearish'] = False  # 價格交付向下改變

        # 先計算 1M 在時間 t 時已確認的 Swing High/Low
        df_1m['confirmed_sh_1m'] = np.nan
        df_1m['confirmed_sl_1m'] = np.nan
        
        last_sh_1m = np.nan
        last_sl_1m = np.nan
        
        highs = df_1m['high'].values
        lows = df_1m['low'].values
        closes = df_1m['close'].values
        opens = df_1m['open'].values
        is_sh_1m = df_1m['is_swing_high'].values
        is_sl_1m = df_1m['is_swing_low'].values
        is_vol_spike = df_1m['is_vol_spike'].values

        # 動態保存的結構物件
        # 我們追蹤最後幾個 OB 與 FVG 以便 CISD 突破檢測
        last_bullish_ob: Optional[Tuple[float, float]] = None # (low, high)
        last_bearish_ob: Optional[Tuple[float, float]] = None # (low, high)
        
        # 流動性掠奪觸發標誌，用於尋找隨後的 MSS
        sweep_low_active = False
        sweep_low_idx = -1
        sweep_high_active = False
        sweep_high_idx = -1

        for i in range(len(df_1m)):
            # 1. 檢索是否有新的 1M Swing 點在 i - swing_window_1m 被確認
            confirm_idx = i - self.swing_window_1m
            if confirm_idx >= 0:
                if is_sh_1m[confirm_idx]:
                    last_sh_1m = highs[confirm_idx]
                if is_sl_1m[confirm_idx]:
                    last_sl = lows[confirm_idx] # 修正：應該是 last_sl_1m
                    last_sl_1m = lows[confirm_idx]

            df_1m.at[i, 'confirmed_sh_1m'] = last_sh_1m
            df_1m.at[i, 'confirmed_sl_1m'] = last_sl_1m

            # 2. 檢測 FVG (第 i 根，看 i-2 與 i 之間的缺口)
            # 多頭 FVG: i-2 的 High < i 的 Low
            if i >= 2 and highs[i-2] < lows[i]:
                df_1m.at[i, 'bullish_fvg_low'] = highs[i-2]
                df_1m.at[i, 'bullish_fvg_high'] = lows[i]
            
            # 空頭 FVG: i-2 的 Low > i 的 High
            if i >= 2 and lows[i-2] > highs[i]:
                df_1m.at[i, 'bearish_fvg_low'] = highs[i]
                df_1m.at[i, 'bearish_fvg_high'] = lows[i-2]

            # 3. 檢測 Liquidity Sweep (流動性掠奪)
            # Sweep Low: 最低點跌破已確認的 1M Swing Low，但實體收盤收在 Swing Low 之上
            if not np.isnan(last_sl_1m) and lows[i] < last_sl_1m and closes[i] >= last_sl_1m:
                df_1m.at[i, 'sweep_low'] = True
                sweep_low_active = True
                sweep_low_idx = i
                
            # Sweep High: 最高點漲破已確認的 1M Swing High，但實體收盤收在 Swing High 之下
            if not np.isnan(last_sh_1m) and highs[i] > last_sh_1m and closes[i] <= last_sh_1m:
                df_1m.at[i, 'sweep_high'] = True
                sweep_high_active = True
                sweep_high_idx = i

            # 4. 檢測 MSS / CHoCH (結構轉換)
            # Bullish MSS: 當 sweep_low_active 且實體收盤價突破「在此之前已確認」的 1M Swing High
            if sweep_low_active and not np.isnan(last_sh_1m) and closes[i] > last_sh_1m:
                df_1m.at[i, 'mss_bullish'] = True
                sweep_low_active = False # 重設
                
                # 確定 Bullish OB：在 MSS 突破前（即 sweep_low_idx 到 i 之間）最後一根陰線 (Close < Open)
                # 若無陰線，則取該區段的最低 K 棒
                ob_low, ob_high = np.nan, np.nan
                for idx in range(i, sweep_low_idx - 1, -1):
                    if idx >= 0 and closes[idx] < opens[idx]:
                        ob_low = lows[idx]
                        ob_high = highs[idx]
                        break
                if np.isnan(ob_low): # 如果沒陰線，用最低那根
                    min_idx = sweep_low_idx + np.argmin(lows[sweep_low_idx:i+1])
                    ob_low = lows[min_idx]
                    ob_high = highs[min_idx]
                
                last_bullish_ob = (ob_low, ob_high)

            # Bearish MSS: 當 sweep_high_active 且實體收盤價跌破「在此之前已確認」的 1M Swing Low
            if sweep_high_active and not np.isnan(last_sl_1m) and closes[i] < last_sl_1m:
                df_1m.at[i, 'mss_bearish'] = True
                sweep_high_active = False # 重設
                
                # 確定 Bearish OB：在 MSS 突破前最後一根陽線 (Close > Open)
                ob_low, ob_high = np.nan, np.nan
                for idx in range(i, sweep_high_idx - 1, -1):
                    if idx >= 0 and closes[idx] > opens[idx]:
                        ob_low = lows[idx]
                        ob_high = highs[idx]
                        break
                if np.isnan(ob_low):
                    max_idx = sweep_high_idx + np.argmax(highs[sweep_high_idx:i+1])
                    ob_low = lows[max_idx]
                    ob_high = highs[max_idx]
                
                last_bearish_ob = (ob_low, ob_high)

            # 將最新確定的 OB 記錄到 DataFrame 中，以便回測引擎調用
            if last_bullish_ob is not None:
                df_1m.at[i, 'bullish_ob_low'] = last_bullish_ob[0]
                df_1m.at[i, 'bullish_ob_high'] = last_bullish_ob[1]
            if last_bearish_ob is not None:
                df_1m.at[i, 'bearish_ob_low'] = last_bearish_ob[0]
                df_1m.at[i, 'bearish_ob_high'] = last_bearish_ob[1]

            # 5. 檢測 CISD (價格交付改變)
            # Bullish CISD: 實體收盤突破對立 (Bearish) OB 的最高點，且伴隨成交量爆量 (is_vol_spike) 及當前有 FVG 形成
            # 這裡的對立 OB 是 last_bearish_ob
            if last_bearish_ob is not None and closes[i] > last_bearish_ob[1]:
                # 伴隨爆量與 FVG 形成
                has_bullish_fvg = (i >= 2 and highs[i-2] < lows[i])
                if is_vol_spike[i] and has_bullish_fvg:
                    df_1m.at[i, 'cisd_bullish'] = True
                    # 重新定義新的強勢 Bullish OB：為此次引發 CISD 突破的爆量陽線之前的那根陰線
                    ob_low, ob_high = np.nan, np.nan
                    for idx in range(i, max(-1, i-5), -1):
                        if closes[idx] < opens[idx]:
                            ob_low = lows[idx]
                            ob_high = highs[idx]
                            break
                    if not np.isnan(ob_low):
                        last_bullish_ob = (ob_low, ob_high)
                        df_1m.at[i, 'bullish_ob_low'] = ob_low
                        df_1m.at[i, 'bullish_ob_high'] = ob_high

            # Bearish CISD: 實體收盤跌破對立 (Bullish) OB 的最低點，且伴隨成交量爆量及當前有 FVG 形成
            if last_bullish_ob is not None and closes[i] < last_bullish_ob[0]:
                has_bearish_fvg = (i >= 2 and lows[i-2] > highs[i])
                if is_vol_spike[i] and has_bearish_fvg:
                    df_1m.at[i, 'cisd_bearish'] = True
                    # 重新定義新的強勢 Bearish OB
                    ob_low, ob_high = np.nan, np.nan
                    for idx in range(i, max(-1, i-5), -1):
                        if closes[idx] > opens[idx]:
                            ob_low = lows[idx]
                            ob_high = highs[idx]
                            break
                    if not np.isnan(ob_low):
                        last_bearish_ob = (ob_low, ob_high)
                        df_1m.at[i, 'bearish_ob_low'] = ob_low
                        df_1m.at[i, 'bearish_ob_high'] = ob_high

        return df_1m
