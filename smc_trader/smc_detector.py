import pandas as pd
import numpy as np
import datetime
from typing import Dict, List, Tuple, Optional

class SMCDetector:
    """
    SMC & SNR 策略特徵檢測器。
    負責辨識 5M 結構的 BOS 趨勢，以及 1M 結構中的 Swing High/Low、Liquidity Sweep、MSS、FVG、OB 和 CISD。
    所有特徵的計算都符合「無未來函數 (No Look-ahead Bias)」的原則。
    """
    def __init__(self, swing_window_5m: int = 5, swing_window_1m: int = 3, 
                 volume_ma_period: int = 20, volume_mult: float = 1.3,
                 atr_period: int = 14, atr_ma_period: int = 20, atr_mult: float = 1.0,
                 pullback_buffer_pts: float = 20.0,
                 pullback_buffer_atr_mult: float = 1.0,
                 pullback_buffer_min: float = 10.0,
                 pullback_buffer_max: float = 40.0,
                 atr_5m_period: int = 14):
        self.swing_window_5m = swing_window_5m
        self.swing_window_1m = swing_window_1m
        self.volume_ma_period = volume_ma_period
        self.volume_mult = volume_mult
        self.atr_period = atr_period
        self.atr_ma_period = atr_ma_period
        self.atr_mult = atr_mult
        # [FIX] pullback_buffer_pts 現在只作為「5M ATR 還不足以計算時」的備援固定值
        # （例如剛開盤、歷史資料還不夠 atr_5m_period 根）。正常情況下動態緩衝
        # （pullback_buffer_atr_mult * 5M ATR，夾在 [pullback_buffer_min, pullback_buffer_max]
        # 之間）才是實際生效的緩衝寬度，波動大時自動放寬、波動小時自動收緊。
        self.pullback_buffer_pts = pullback_buffer_pts
        self.pullback_buffer_atr_mult = pullback_buffer_atr_mult
        self.pullback_buffer_min = pullback_buffer_min
        self.pullback_buffer_max = pullback_buffer_max
        self.atr_5m_period = atr_5m_period

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
        BOS 定義：實體 K 棒收盤突破前一個「已確認的」Swing High/Low 一段動態緩衝距離
        （5M ATR-based，波動大時自動放寬、波動小時自動收緊）。
        """
        df_5m = self.detect_swings(df_5m, self.swing_window_5m)
        
        # 用於追踪目前已確認的 Swing High/Low
        # 因為只有在當前索引 t，我們才能確認 t - swing_window_5m 的 Swing 點
        # 故回測時我們只能在第 t 步，使用索引 <= t - swing_window_5m 已經確認的 Swing 值
        df_5m['confirmed_swing_high'] = np.nan
        df_5m['confirmed_swing_low'] = np.nan
        df_5m['trend_5m'] = "NONE" # NONE, BULLISH, BEARISH

        # [NEW] 5M ATR：作為動態緩衝的基礎。用 5M 自己的 high/low/close 算 True Range，
        # 再取 rolling mean。ATR 不足（資料剛開始還不夠 atr_5m_period 根）時為 NaN，
        # 迴圈裡會 fallback 用 self.pullback_buffer_pts 這個固定值。
        prev_close_5m = df_5m['close'].shift(1)
        tr1 = df_5m['high'] - df_5m['low']
        tr2 = (df_5m['high'] - prev_close_5m).abs()
        tr3 = (df_5m['low'] - prev_close_5m).abs()
        tr_5m = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1, skipna=True)
        atr_5m_arr = tr_5m.rolling(window=self.atr_5m_period, min_periods=1).mean().values
        df_5m['atr_5m'] = atr_5m_arr

        last_sh = np.nan
        last_sl = np.nan
        current_trend = "NONE"

        # --- SMC 單次突破消耗與深回檔轉 NONE 機制 ---
        # active_break_level：目前維持 current_trend 的那個「被突破的關鍵價位」。
        #   BULLISH 時代表被向上突破的 swing high；BEARISH 時代表被向下突破的 swing low。
        #   若價格深幅回落，收盤價「跌回」該價位之下 (或漲回之上)，代表原本的推動結構已失效，
        #   趨勢自動降級為 NONE，而不是盲目維持原方向。
        # consumed_sh / consumed_sl：記錄「已經被用來觸發過 BOS 的最高 swing high / 最低 swing low」。
        #   同一個價位不會重複觸發同方向的 BOS；唯有價格再創出「新的、比 consumed 值更極端」的
        #   swing high/low 並且實體突破，才能重新確立同方向趨勢，避免在盤整時反覆假突破同一價位。
        # [FIX] consumed_sh/consumed_sl 原本永久不重置，只要拿去跑跨日、跨月的長資料，
        # 就會被很久以前、跟當下結構已經無關的極值卡死，導致「劣勢方向」的新鮮度門檻越墊越高，
        # 最終幾乎鎖死（實測：連續資料不重置時，多空比可以從健康的1.2~2倍惡化到200倍以上）。
        # 現在改成：一旦趨勢真的降級成 NONE（代表這波結構已經失效），就把對應的 consumed 值
        # 一併歸零，讓下一次哪怕是「同一個舊高/低點」的重新突破也能被視為全新的一次。
        active_break_level = np.nan
        consumed_sh = np.nan
        consumed_sl = np.nan

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

            # [NEW] 動態緩衝：ATR 足夠時用 ATR*mult（夾在 min~max 之間），不足時 fallback 固定值。
            # 同一個 dynamic_buffer 同時用在「進場死區」跟「降級到NONE」兩處，兩端各留一段緩衝，
            # 避免 reset 之後任何一次極小幅度的假突破就重新觸發（見上方討論的死區設計）。
            if not np.isnan(atr_5m_arr[i]):
                dynamic_buffer = float(np.clip(
                    atr_5m_arr[i] * self.pullback_buffer_atr_mult,
                    self.pullback_buffer_min, self.pullback_buffer_max
                ))
            else:
                dynamic_buffer = self.pullback_buffer_pts

            # 檢測「新鮮」BOS：實體收盤突破 last_sh/last_sl 一段死區緩衝，且該價位尚未被消耗過
            # (last_sh/last_sl 比之前 consumed 過的更高/更低，代表是真正的新高/新低)
            fresh_bullish_bos = (
                not np.isnan(last_sh) and closes[i] > (last_sh + dynamic_buffer) and
                (np.isnan(consumed_sh) or last_sh > consumed_sh)
            )
            fresh_bearish_bos = (
                not np.isnan(last_sl) and closes[i] < (last_sl - dynamic_buffer) and
                (np.isnan(consumed_sl) or last_sl < consumed_sl)
            )

            if fresh_bullish_bos:
                # 向上 BOS（新高突破，含反轉或創新高延續兩種情況）
                current_trend = "BULLISH"
                active_break_level = last_sh
                consumed_sh = last_sh
            elif fresh_bearish_bos:
                # 向下 BOS（新低突破，含反轉或創新低延續兩種情況）
                current_trend = "BEARISH"
                active_break_level = last_sl
                consumed_sl = last_sl
            else:
                # 沒有新的 BOS 發生時，檢查是否發生「深幅回檔」，趨勢降級為 NONE
                if current_trend == "BULLISH" and not np.isnan(active_break_level) and closes[i] < (active_break_level - dynamic_buffer):
                    current_trend = "NONE"
                    active_break_level = np.nan
                    consumed_sh = np.nan  # [NEW] 結構已失效，重置，下次同樣的高點也算全新一次
                elif current_trend == "BEARISH" and not np.isnan(active_break_level) and closes[i] > (active_break_level + dynamic_buffer):
                    current_trend = "NONE"
                    active_break_level = np.nan
                    consumed_sl = np.nan  # [NEW]

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
        df_5m_lookup = df_5m_processed[['ts', 'trend_5m']].copy()
        # 動態計算 5M K 線的實際跨度，自適應 Mock 模式 (50秒) 與真實模式 (5分鐘)
        if len(df_5m_lookup) >= 2:
            time_diff = df_5m_lookup['ts'].diff().median()
            # 若 median 計算出合理值則使用，否則 fallback
            close_delta = time_diff if not pd.isnull(time_diff) else pd.Timedelta(minutes=5)
        else:
            close_delta = pd.Timedelta(minutes=5)
            
        # 計算此根 5M K 線收盤的 1M 時間
        df_5m_lookup['close_1m_ts'] = df_5m_lookup['ts'] + close_delta
        df_5m_lookup = df_5m_lookup.sort_values('close_1m_ts').reset_index(drop=True)

        # 注意：不可用「時間戳記完全相等」比對來對齊 (原本用 dict 查表的做法)。
        # 離線回測資料的 1M 時間戳剛好整分對齊，相等比對不會出錯；
        # 但即時監控 (monitor.py) 的 1M K 棒時間戳來自真實 tick 抵達時刻，
        # 帶有不規則秒數/微秒，幾乎不可能與 close_1m_ts 完全相等，
        # 會導致 trend_5m 在對齊失敗後「凍結」在最後一次命中的值。
        # 改用 merge_asof(direction='backward')：為每根 1M K 棒找出「時間 <= 自己」
        # 的最後一根已收盤 5M K 棒，同樣不會用到尚未收盤的 5M 資料，維持無未來函數。
        df_1m = df_1m.sort_values('ts').reset_index(drop=True)
        # 若 df_1m 已帶有先前呼叫留下的 trend_5m 欄位（例如 Walk-Forward 對同一份
        # 已處理過的資料重新切片、再次呼叫本函式），先移除舊欄位，
        # 避免 merge_asof 因兩邊都有 trend_5m 而產生 trend_5m_x / trend_5m_y，
        # 造成後面 df_1m['trend_5m'] 找不到欄位。確保本函式可重複呼叫。
        if 'trend_5m' in df_1m.columns:
            df_1m = df_1m.drop(columns=['trend_5m'])
        df_1m = pd.merge_asof(
            df_1m,
            df_5m_lookup[['close_1m_ts', 'trend_5m']],
            left_on='ts',
            right_on='close_1m_ts',
            direction='backward'
        )
        df_1m['trend_5m'] = df_1m['trend_5m'].fillna('NONE')
        df_1m = df_1m.drop(columns=['close_1m_ts'])

        # 標記日盤與夜盤，隔離跨盤期間的計算
        import datetime as dt_mod
        df_1m['session_type'] = np.where(
            (df_1m['ts'].dt.time >= dt_mod.time(8, 45)) & (df_1m['ts'].dt.time <= dt_mod.time(13, 45)), 
            'Day', 'Night'
        )

        # 3. 計算 Volume MA 用於檢測爆量 (隔離日夜盤計算，避免夜盤開盤誤判)
        df_1m['vol_ma'] = df_1m.groupby('session_type')['volume'].transform(
            lambda x: x.rolling(window=self.volume_ma_period, min_periods=1).mean()
        )
        df_1m['is_vol_spike'] = df_1m['volume'] >= (df_1m['vol_ma'] * self.volume_mult)

        # 3.5 計算 ATR 與波動濾網標記
        # 依 session_type 分組取 shift(1)，避免跨盤跳空缺口污染 TR 計算
        prev_close = df_1m.groupby('session_type')['close'].shift(1)
        tr1 = df_1m['high'] - df_1m['low']
        tr2 = (df_1m['high'] - prev_close).abs()
        tr3 = (df_1m['low'] - prev_close).abs()
        df_1m['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # 隔離日夜盤的 ATR 均線計算，避免早盤高波動影響夜盤初期的波動率判定
        df_1m['atr'] = df_1m.groupby('session_type')['tr'].transform(
            lambda x: x.rolling(window=self.atr_period, min_periods=1).mean()
        )
        df_1m['atr_ma'] = df_1m.groupby('session_type')['atr'].transform(
            lambda x: x.rolling(window=self.atr_ma_period, min_periods=1).mean()
        )
        
        df_1m['is_volatile'] = True
        df_1m.loc[df_1m['atr'].notna() & df_1m['atr_ma'].notna(), 'is_volatile'] = \
            df_1m['atr'] >= (df_1m['atr_ma'] * self.atr_mult)

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
        df_1m['leg_high_1m'] = np.nan
        df_1m['leg_low_1m'] = np.nan
        
        last_sh_1m = np.nan
        last_sl_1m = np.nan
        leg_high_1m = np.nan
        leg_low_1m = np.nan
        
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
                    if np.isnan(leg_high_1m) or last_sh_1m > leg_high_1m:
                        leg_high_1m = last_sh_1m
                if is_sl_1m[confirm_idx]:
                    last_sl_1m = lows[confirm_idx]
                    if np.isnan(leg_low_1m) or last_sl_1m < leg_low_1m:
                        leg_low_1m = last_sl_1m

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
                # [FIX] 鎖定插針本身觸及的最低價(lows[i])，而不是被掃的舊參考值——
                # 插針才是這次流動性掠奪真正探到的深度，SMC邏輯上該保護的位置
                if np.isnan(leg_low_1m) or lows[i] < leg_low_1m:
                    leg_low_1m = lows[i]

            # Sweep High: 最高點漲破已確認的 1M Swing High，但實體收盤收在 Swing High 之下
            if not np.isnan(last_sh_1m) and highs[i] > last_sh_1m and closes[i] <= last_sh_1m:
                df_1m.at[i, 'sweep_high'] = True
                sweep_high_active = True
                sweep_high_idx = i
                if np.isnan(leg_high_1m) or highs[i] > leg_high_1m:
                    leg_high_1m = highs[i]

            # 3.5 錨點失效判定：若收盤價「真的」跌破/漲破這一波追蹤的錨點（不只是插針），
            # 不論當下是否仍在 sweep_low_active/sweep_high_active 期間，都直接作廢重置。
            if not np.isnan(leg_low_1m) and closes[i] < leg_low_1m:
                sweep_low_active = False
                leg_low_1m = np.nan
            if not np.isnan(leg_high_1m) and closes[i] > leg_high_1m:
                sweep_high_active = False
                leg_high_1m = np.nan

            df_1m.at[i, 'confirmed_sh_1m'] = last_sh_1m
            df_1m.at[i, 'confirmed_sl_1m'] = last_sl_1m
            df_1m.at[i, 'leg_high_1m'] = leg_high_1m
            df_1m.at[i, 'leg_low_1m'] = leg_low_1m

            # 3.6 「乾淨突破」歸零：即使沒有正式的 sweep_low_active/sweep_high_active，
            # 只要收盤價站上前高/跌破前低，代表這一段已經翻頁，舊的 leg 錨點對之後
            # 全新一段沒有意義，一併歸零，避免卡住很久以前不相干的舊低/高點。
            # （若當下正是 sweep_active，留給下面的 MSS 判斷處理，這根的欄位已經寫入，不受影響）
            if not sweep_low_active and not np.isnan(last_sh_1m) and closes[i] > last_sh_1m:
                leg_low_1m = np.nan
            if not sweep_high_active and not np.isnan(last_sl_1m) and closes[i] < last_sl_1m:
                leg_high_1m = np.nan

            # 4. 檢測 MSS / CHoCH (結構轉換)
            # Bullish MSS: 當 sweep_low_active 且實體收盤價突破「在此之前已確認」的 1M Swing High
            if sweep_low_active and not np.isnan(last_sh_1m) and closes[i] > last_sh_1m:
                df_1m.at[i, 'mss_bullish'] = True
                sweep_low_active = False # 重設
                
                # [FIX] 只重置多方自己的 leg_low_1m（已轉為用 OB 當主要參考）；
                # 不要連帶重置 leg_high_1m——那是空方獨立的狀態，跟這次多方 MSS 無關。
                # 也不覆寫本根 df_1m.at[i,'leg_low_1m']（上面已寫入正確值，monitor.py
                # 要靠它算 SL2），只重置變數讓下一根開始才是 NaN。
                leg_low_1m = np.nan
                
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
                
                # [FIX] 同上，只重置空方自己的 leg_high_1m，不連帶動到 leg_low_1m
                leg_high_1m = np.nan
                
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

            # 5. OB 失效 (Mitigation) 檢查與更新
            # 如果價格跌破 Bullish OB 的下緣，則該 OB 被「緩解 / 破壞」，宣告失效
            if last_bullish_ob is not None and lows[i] < last_bullish_ob[0]:
                last_bullish_ob = None
                
            # 如果價格突破 Bearish OB 的上緣，則該 OB 被「緩解 / 破壞」，宣告失效
            if last_bearish_ob is not None and highs[i] > last_bearish_ob[1]:
                last_bearish_ob = None

            # 將最新確定且尚未失效的 OB 記錄到 DataFrame 中
            if last_bullish_ob is not None:
                df_1m.at[i, 'bullish_ob_low'] = last_bullish_ob[0]
                df_1m.at[i, 'bullish_ob_high'] = last_bullish_ob[1]
            if last_bearish_ob is not None:
                df_1m.at[i, 'bearish_ob_low'] = last_bearish_ob[0]
                df_1m.at[i, 'bearish_ob_high'] = last_bearish_ob[1]

            # 5. 檢測 CISD (價格交付改變)
            # 取得當前 K 棒的時間，限制只能在 09:00 到 11:00 之間觸發 CISD
            bar_time = df_1m['ts'].iloc[i].time()
            is_cisd_time_valid = (datetime.time(9, 0) <= bar_time <= datetime.time(11, 0))

            # Bullish CISD: 實體收盤突破對立 (Bearish) OB 的最高點，且伴隨成交量爆量 (is_vol_spike) 及當前有 FVG 形成
            # 這裡的對立 OB 是 last_bearish_ob
            if last_bearish_ob is not None and closes[i] > last_bearish_ob[1]:
                # 伴隨爆量與 FVG 形成
                has_bullish_fvg = (i >= 2 and highs[i-2] < lows[i])
                if is_vol_spike[i] and has_bullish_fvg and is_cisd_time_valid:
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
                if is_vol_spike[i] and has_bearish_fvg and is_cisd_time_valid:
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
