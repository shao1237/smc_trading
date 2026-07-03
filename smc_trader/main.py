import os
import argparse
import pandas as pd
from smc_trader.config import (
    SHIOAJI_API_KEY, SHIOAJI_SECRET_KEY, SHIOAJI_SIMULATION, INITIAL_CAPITAL, DEFAULT_RR,
    SWING_WINDOW_5M, SWING_WINDOW_1M, VOLUME_MA_PERIOD, VOLUME_MULT
)
from smc_trader.data_provider import ShioajiDataProvider, MockDataProvider, resample_to_5m
from smc_trader.smc_detector import SMCDetector
from smc_trader.backtester import Backtester
from smc_trader.reporter import Reporter
from smc_trader.stats_validator import StatsValidator

def main():
    parser = argparse.ArgumentParser(description="SMC+SNR 台指期自動化交易回測系統")
    parser.add_argument("--mode", type=str, choices=["mock", "shioaji"], default="mock",
                        help="資料來源模式: mock (模擬數據) 或 shioaji (永豐金 API)")
    parser.add_argument("--start", type=str, default="2026-06-25",
                        help="回測開始日期 (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2026-07-02",
                        help="回測結束日期 (YYYY-MM-DD)")
    parser.add_argument("--rr", type=float, default=DEFAULT_RR,
                        help="風險報酬比 (Risk-Reward Ratio)")
    parser.add_argument("--api-key", type=str, default=SHIOAJI_API_KEY,
                        help="Shioaji API Key")
    parser.add_argument("--secret-key", type=str, default=SHIOAJI_SECRET_KEY,
                        help="Shioaji Secret Key")
    parser.add_argument("--validate", action="store_true",
                        help="是否啟用蒙地卡羅/Bootstrap/WFA 等高階統計檢定")
    parser.add_argument("--num-tests", type=int, default=10,
                        help="Bonferroni 校正所測試的參數組數量")
    parser.add_argument("--vol-mult", type=float, default=VOLUME_MULT,
                        help="CISD 的成交量爆量判定倍數")
    
    args = parser.parse_args()

    print("=" * 60)
    print("           SMC + SNR 台指期交易策略自動化回測系統           ")
    print("=" * 60)
    print(f"運行模式: {args.mode.upper()}")
    print(f"時間區間: {args.start} 至 {args.end}")
    print(f"風險報酬比 (R:R): {args.rr}")
    print("-" * 60)

    # 1. 取得數據
    df_1m = None
    if args.mode == "shioaji":
        # 檢查參數是否提供
        k = args.api_key
        s = args.secret_key
        if not k or not s:
            print("[警告] 未設定 Shioaji API_KEY 或 SECRET_KEY。將自動 Fallback 至 MOCK 模擬數據模式！")
            args.mode = "mock"
        else:
            try:
                provider = ShioajiDataProvider(
                    api_key=k,
                    secret_key=s,
                    simulation=SHIOAJI_SIMULATION
                )
                df_1m = provider.get_1m_data(start_date=args.start, end_date=args.end)
            except Exception as e:
                print(f"[錯誤] Shioaji 資料撈取失敗: {str(e)}")
                print("[提示] 將自動 Fallback 至 MOCK 模擬數據模式以利程式順利運行！")
                args.mode = "mock"

    if args.mode == "mock":
        provider = MockDataProvider()
        df_1m = provider.get_1m_data(start_date=args.start, end_date=args.end)

    if df_1m is None or df_1m.empty:
        print("[嚴重錯誤] 無法獲取 K 線數據，程式終止。")
        return

    print(f"成功加載 1M 原始數據: 共 {len(df_1m)} 根 K 線")

    # 2. 合成 5M K 線
    print("正在將 1M 資料合成為 5M 結構數據...")
    df_5m = resample_to_5m(df_1m)
    print(f"5M 數據合成完畢: 共 {len(df_5m)} 根 K 線")

    # 3. SMC & SNR 特徵辨識
    print("正在進行 SMC 核心結構與特徵檢測 (BOS, MSS, Sweep, FVG, OB, CISD)...")
    detector = SMCDetector(
        swing_window_5m=SWING_WINDOW_5M,
        swing_window_1m=SWING_WINDOW_1M,
        volume_ma_period=VOLUME_MA_PERIOD,
        volume_mult=args.vol_mult
    )
    
    # 3.1 處理 5M 大結構
    df_5m_processed = detector.process_5m_structure(df_5m)
    
    # 3.2 處理 1M 內部結構與信號標註
    df_1m_processed = detector.process_1m_signals(df_1m, df_5m_processed)
    print("特徵識別完成。")

    # 4. 執行回測
    print("開始執行逐 K 棒回測交易模擬...")
    backtester = Backtester(
        initial_capital=INITIAL_CAPITAL,
        default_rr=args.rr
    )
    trades, df_backtested = backtester.run(df_1m_processed)
    print(f"回測執行完畢。總成交交易筆數: {len(trades)} 筆")

    # 5. 生成報告與圖表
    print("正在計算策略指標與生成回測報告...")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports")
    os.makedirs(output_dir, exist_ok=True)
    
    reporter = Reporter(output_dir=output_dir)
    
    # 5.1 執行高階統計檢定 (若啟用了 --validate)
    val_metrics = None
    val_img_path = None
    if args.validate:
        print("正在執行高階統計檢定 (MCPT, Bootstrap, WFA, Bonferroni)...")
        validator = StatsValidator()
        
        # A. Bootstrap CI
        boot_res = validator.run_bootstrap_ci(trades, num_bootstrap=2000)
        
        # B. MCPT
        mcpt_res = validator.run_mcpt(trades, num_permutations=1000)
        
        # C. Bonferroni 校正
        bonf_res = validator.run_bonferroni(mcpt_res['p_value'], num_tests=args.num_tests)
        
        # D. Walk-Forward Analysis
        wfa_res = validator.run_walk_forward(df_1m_processed, df_5m_processed, num_folds=3)
        
        val_metrics = {
            'mcpt': mcpt_res,
            'bootstrap': boot_res,
            'bonferroni': bonf_res,
            'wfa': wfa_res
        }
        
        # 繪製統計檢定直方圖
        val_img_path = reporter.plot_validation_charts(
            mcpt_dist=mcpt_res['distribution'],
            real_profit=mcpt_res['real_profit'],
            p_value=mcpt_res['p_value'],
            bootstrap_dist=boot_res['distribution'],
            low_ci=boot_res['low_ci'],
            high_ci=boot_res['high_ci'],
            filename="validation_charts.png"
        )
        print("高階統計檢定完成，並已生成檢定分佈圖表。")

    # 計算指標
    metrics = reporter.calculate_metrics(trades, backtester.equity_curve, INITIAL_CAPITAL)
    
    # 繪製曲線圖
    equity_img_path = reporter.plot_equity_curve(backtester.equity_curve, filename="equity_curve.png")
    
    # 生成 HTML
    report_html_path = reporter.generate_html_report(
        metrics, 
        trades, 
        equity_img_path, 
        validation_metrics=val_metrics, 
        validation_filepath=val_img_path, 
        filename="report.html"
    )

    # 6. 輸出摘要至終端機
    print("=" * 60)
    print("                       回測結果摘要                       ")
    print("=" * 60)
    print(f"初始資金 : {INITIAL_CAPITAL:,.0f} NTD")
    print(f"期末淨值 : {metrics['ending_equity']:,.0f} NTD")
    print(f"淨損益   : {metrics['net_profit']:+,.0f} NTD")
    print(f"總交易數 : {metrics['total_trades']} 筆 (獲利: {metrics['total_wins']} 筆, 虧損: {metrics['total_losses']} 筆)")
    print(f"勝率     : {metrics['win_rate']}%")
    print(f"平均盈虧比 (R:R) : {metrics['avg_rr']}")
    print(f"獲利因子 (Profit Factor) : {metrics['profit_factor']}")
    print(f"最大回撤 (MDD)          : {metrics['max_drawdown_pct']}%")
    if val_metrics:
        print("-" * 60)
        print("                     高階統計驗證結果                     ")
        print("-" * 60)
        print(f"MCPT p-value       : {val_metrics['mcpt']['p_value']} (通過: {val_metrics['mcpt']['passed']})")
        print(f"Bonferroni 校正門檻 : {val_metrics['bonferroni']['adjusted_alpha']} (通過: {val_metrics['bonferroni']['passed']})")
        print(f"Walk-Forward (WFE) : {val_metrics['wfa']['wfe']}% (通過: {val_metrics['wfa']['passed']})")
        print(f"Bootstrap 95% CI   : [{val_metrics['bootstrap']['low_ci']}, {val_metrics['bootstrap']['high_ci']}] (通過: {val_metrics['bootstrap']['passed']})")
    print("=" * 60)
    print(f"資產曲線圖已儲存至: [equity_curve.png](file:///{equity_img_path.replace(os.sep, '/')})")
    if val_img_path:
        print(f"高階統計圖已儲存至: [validation_charts.png](file:///{val_img_path.replace(os.sep, '/')})")
    print(f"HTML 互動報告已生成至: [report.html](file:///{report_html_path.replace(os.sep, '/')})")
    print("=" * 60)

if __name__ == "__main__":
    main()
