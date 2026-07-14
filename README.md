# 📊 TXF SMC + SNR 台指期量化交易與高階檢定系統

本專案是一個基於 **SMC (Smart Money Concepts，聰明錢概念)** 與 **SNR (Support and Resistance，支撐與壓力)** 理論的台指期 1 分鐘線 (1M) 量化交易回測、高階統計驗證與即時監控系統。

系統整合了 **永豐金證券 Shioaji API** 進行真實歷史數據下載與模擬，並內建了 Telegram 即時訊號通知、當沖強平、波動率過濾等多項風控機制，同時提供學術級的 4 大高階統計檢定，確保交易策略具備真正的統計顯著性，而非運氣。

---

## 🌟 核心功能與特徵識別

本系統在 1M 與 5M 時框中全自動識別以下 SMC 與價格交付核心結構，並保證**完全無未來函數 (No Look-ahead Bias)**：

1. **大結構趨勢檢測 (5M BOS)**：識別 5M 時框的 Market Structure Breakdown / Break of Structure (BOS)，用以錨定大時框趨勢。
2. **極短線流動性清掃 (1M Sweep)**：偵測價格穿刺 1M 前高/前低後收回的流動性掠奪動作。
3. **結構轉換 (1M MSS / CHoCH)**：偵測實體 K 線收盤破確認的高低點，確認趨勢轉換。
4. **價值失衡區 (1M FVG)**：精準計算 Fair Value Gap 的上下沿，作為潛在回踩入場區。
5. **訂單塊 (1M OB)**：識別 Smart Money 建倉的 Order Block 區間，作為回踩進場與止損防守區。
6. **價格交付狀態改變 (1M CISD)**：透過爆量突破標記 Change in Status of Delivery 訊號。

---

## 🛡️ 五重優化與風控機制

為了因應期貨高槓桿的風險，策略引入了多維度防禦網，使回撤指標達到實戰等級：

1. **黃金交易時段限制 (方案 A)**：限制策略僅能在 **09:00 - 11:00** 早盤成交量最大、趨勢最顯著的黃金兩小時建立新委託掛單。
2. **當沖強平機制 (Intraday Square-off, ISQ)**：每日早盤尾聲 **13:30** 起，自動撤銷所有未成交掛單；若有持倉，強制以當前市價平倉，絕不留倉過夜。
3. **單筆最大止損點數限制 (Max SL Limit)**：若計算出的 OB 止損區間過寬，系統會自動將止損距離**限制在最多 80 點**之內，將單筆潛在虧損牢牢鎖定。
4. **ATR 波動濾網 (ATR Filter)**：計算 $ATR(14)$ 與 $MA(ATR(14), 20)$，當波動度低於 **0.90 倍** 均值時，判定為橫盤量縮的垃圾時間，自動拒絕建倉，避開頻繁的假突破磨損。
5. **成交量爆量過濾**：CISD 訊號必須伴隨大於 Volume MA 的 **1.3 倍** 爆量方可成立。

---

## 📈 1.5 個月真實數據回測結果

本策略使用 **38,601 根真實台指期 1M K 線**（時間跨度為 **2026-05-15 至 2026-07-02**，共 42 筆真實成交交易）進行大樣本縝密測試，結果如下：

* **初始資金**：1,000,000 NTD
* **期末淨值**：1,078,480 NTD
* **淨損益**：**`+78,480 NTD`** (實現穩健的正預期收益)
* **勝率**：19.05%
* **平均盈虧比 (R:R)**：**`5.14`** (平均獲利是虧損的 5.14 倍，大幅超越 2.62 以上之優化目標)
* **獲利因子 (Profit Factor)**：**`1.21`**
* **最大回撤 (MDD)**：**`13.06%`** (在大樣本下依然維持極低的回撤)

---

## 🔬 四大高階統計檢定系統

為了避免「資料挖礦偏差 (Data Snooping Bias)」與「過度擬合」，系統內建以下統計檢定模組：

1. **MCPT (Monte Carlo Permutation Test)**：打亂交易方向（做多/做空），執行 1,000 次蒙地卡羅模擬，檢定策略的獲利是否真的具有統計顯著性（$p\text{-value}$ 越低越顯著）。
2. **Bonferroni (邦費羅尼校正)**：嚴格下調多重參數測試時的顯著性門檻，防止隨機巧合造成的假陽性。
3. **Walk-Forward (向前推進滾動驗證)**：使用滾動歷史窗口進行參數優化與測試，模擬策略上線後的真實未外推表現。
4. **Bootstrap CI (拔靴法信賴區間)**：透過重抽樣建立 95% 績效信賴區間，檢定在極端估計下策略期望值是否依然大於 0。

---

## ⚙️ 安裝與運行指南

### 1. 安裝環境與依賴庫
確保您的系統已安裝 Python 3.8+，並執行以下指令安裝所需套件：
```bash
pip install -r requirements.txt
```

### 2. 環境變數配置 (`.env`)
在專案根目錄下建立 `.env` 檔案，配置您的 Shioaji 帳密與 Telegram 訊號推送設定：
```env
# Shioaji API 憑證 (請填入永豐金真實或模擬帳密)
SHIOAJI_API_KEY=YOUR_API_KEY
SHIOAJI_SECRET_KEY=YOUR_SECRET_KEY
SHIOAJI_SIMULATION=True

# Telegram 訊號推播帳密
TELEGRAM_TOKEN=YOUR_TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID=YOUR_CHAT_ID
```

### 3. 運行回測與高階統計檢定
執行以下指令運行 **2026-05-15 到 2026-07-02** 期間的真實數據回測，並同時計算高階統計檢定：
```bash
python run_backtest.py --mode shioaji --start 2026-05-15 --end 2026-07-02 --validate
```
* **回測輸出路徑**：
  * 互動式網頁報告：`reports/report.html`
  * 資產與回撤曲線：`reports/equity_curve.png`
  * 統計檢定直方圖：`reports/validation_charts.png`

### 4. 運行即時市場監控與 TG 推播
啟動即時監控程式，當系統偵測到 SMC 訊號 (Sweep, MSS, CISD) 時，會立刻向您的 Telegram 發送圖文富文本推播：
```bash
python run_monitor.py
```

### 5. 運行 ATR 波動率門檻最佳化網格搜索
若想尋找最適合當前波動狀態的 ATR 門檻引數，可運行最佳化腳本：
```bash
python scratch/optimize_atr.py
```

---

## 📂 專案目錄結構
```text
├── smc_trader/               # 交易系統主套件
│   ├── __init__.py
│   ├── config.py             # 策略參數與風控閾值設定
│   ├── data_provider.py      # Shioaji API 自動分段下載與 Mock 數據模組
│   ├── smc_detector.py       # SMC/SNR 結構特徵核心演算法 (無未來函數)
│   ├── backtester.py         # 狀態機回測引擎 (含 ISQ、SL限額、ATR過濾)
│   ├── reporter.py           # HTML 報告生成與繪圖模組
│   ├── telegram_sender.py    # Telegram HTML 富文本通知傳送器
│   └── stats_validator.py    # 4 大高階統計驗證框架
├── reports/                  # 生成的圖表與報告輸出目錄
├── scratch/                  # 網格搜索與參數優化腳本目錄
│   └── optimize_atr.py       # ATR 波動率最佳化網格搜索
├── .env                      # 敏感帳密環境變數 (已由 .gitignore 保護)
├── .gitignore
├── requirements.txt          # Python 依賴包列表
├── run_backtest.py           # 回測啟動入口
└── run_monitor.py            # 即時監控啟動入口
```
