import pandas as pd
import numpy as np
from smc_trader.smc_detector import SMCDetector
from smc_trader.backtester import Backtester

def test_smc_detection():
    print("開始驗證 SMC 檢測演算法...")
    
    # 建立一個測試用 1M K 線資料 (共 20 根)
    # 人工構造一個：BULLISH sweep -> MSS -> CISD 結構
    # 初始價格：100.0
    ts = pd.date_range("2026-07-04 09:00:00", periods=20, freq="1min")
    
    # 用於觸發 swing points 的波動
    # 1. 前期高點在第 3 根: High=105.0
    # 2. 前期低點在第 6 根: Low=95.0
    # 3. 第 9 根跌破 95.0 到 94.0，但收盤在 96.0 (Liquidity Sweep)
    # 4. 第 12 根實體收盤漲破 105.0 到 107.0 (MSS)
    # 5. 引發突破的第 12 根伴隨大成交量與 FVG 形成 (CISD)
    
    opens =   [100, 101, 102, 103, 101, 98,  97,  96,  96,  98,  100, 101, 104, 106, 108, 107, 106, 108, 110, 112]
    highs =   [102, 103, 105, 104, 102, 100, 99,  98,  98,  101, 102, 103, 106, 108, 109, 108, 107, 109, 111, 113]
    lows =    [99,  100, 101, 100, 97,  95,  96,  95,  94,  96,  98,  100, 101, 105, 107, 106, 105, 107, 109, 111]
    closes =  [101, 102, 103, 101, 98,  96,  97,  97,  96,  100, 101, 102, 105, 107, 108, 106, 106, 108, 111, 112]
    volumes = [500, 600, 800, 500, 400, 300, 400, 350, 2000, 800, 700, 900, 1500, 3000, 800, 600, 500, 600, 700, 800]
    
    df_1m = pd.DataFrame({
        'ts': ts,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })

    # 人工構造對齊的 5M 大結構（設為 BULLISH，否則 1M 不會交易）
    df_5m_processed = pd.DataFrame({
        'ts': pd.date_range("2026-07-04 09:00:00", periods=4, freq="5min"),
        'trend_5m': ["BULLISH", "BULLISH", "BULLISH", "BULLISH"]
    })

    detector = SMCDetector(swing_window_1m=2, volume_ma_period=5, volume_mult=1.2)
    
    # 進行 1M 特徵標註
    df_1m_processed = detector.process_1m_signals(df_1m, df_5m_processed)

    # 驗證 Swing 檢測是否正常運作
    print("-> 驗證 Swing High & Low:")
    sh_indices = df_1m_processed[df_1m_processed['is_swing_high'] == True].index.tolist()
    sl_indices = df_1m_processed[df_1m_processed['is_swing_low'] == True].index.tolist()
    print(f"   識別的 Swing High 索引: {sh_indices}")
    print(f"   識別的 Swing Low 索引: {sl_indices}")
    
    # 驗證 FVG 是否成功檢測
    # 多頭 FVG: 第 i-2 根的 High < 第 i 根的 Low。
    # 比如第 13 根的 Low 是 105.0，而第 11 根的 High 是 102.0，這會產生 FVG。
    print("-> 驗證 FVG 檢測:")
    fvg_indices = df_1m_processed[df_1m_processed['bullish_fvg_low'].notna()].index.tolist()
    print(f"   檢測到 Bullish FVG 的索引: {fvg_indices}")

    # 驗證 Sweep Low 檢測
    print("-> 驗證 Sweep Low 檢測:")
    sweep_low_indices = df_1m_processed[df_1m_processed['sweep_low'] == True].index.tolist()
    print(f"   檢測到 Sweep Low 的索引: {sweep_low_indices}")

    # 驗證 MSS 檢測
    print("-> 驗證 MSS Bullish 檢測:")
    mss_indices = df_1m_processed[df_1m_processed['mss_bullish'] == True].index.tolist()
    print(f"   檢測到 MSS Bullish 的索引: {mss_indices}")

    # 驗證 CISD 檢測
    print("-> 驗證 CISD Bullish 檢測:")
    cisd_indices = df_1m_processed[df_1m_processed['cisd_bullish'] == True].index.tolist()
    print(f"   檢測到 CISD Bullish 的索引: {cisd_indices}")

    # 執行回測引擎測試
    print("-> 驗證回測狀態機與交易撮合:")
    backtester = Backtester(initial_capital=1000000.0, default_rr=2.0)
    trades, df_res = backtester.run(df_1m_processed)
    print(f"   測試成交交易筆數: {len(trades)}")
    for t in trades:
        print(f"   交易方向: {t['direction']}, 進場時間: {t['entry_time']}, 進場價: {t['entry_price']}, 出場時間: {t['exit_time']}, 出場價: {t['exit_price']}, 盈虧: {t['pnl']:.1f}")

    print("SMC 演算法單元測試完成！\n")

if __name__ == "__main__":
    test_smc_detection()
