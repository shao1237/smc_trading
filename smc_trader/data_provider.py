import os
import datetime
import numpy as np
import pandas as pd
import shioaji as sj
from smc_trader.config import DATA_CACHE_DIR

class DataProvider:
    """資料供應器基類"""
    def __init__(self):
        pass

    def get_1m_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """獲取 1 分鐘線資料，欄位需包含 ['ts', 'open', 'high', 'low', 'close', 'volume']"""
        raise NotImplementedError


class ShioajiDataProvider(DataProvider):
    """永豐金證券 Shioaji API 資料供應器"""
    def __init__(self, api_key: str, secret_key: str, simulation: bool = True):
        super().__init__()
        self.api_key = api_key
        self.secret_key = secret_key
        self.simulation = simulation
        self.api = None

    def _login(self):
        """登入 API"""
        if self.api is not None:
            return True
        
        if not self.api_key or not self.secret_key:
            raise ValueError("Shioaji 登入失敗：未提供 api_key 或 secret_key")
            
        print(f"正在登入 Shioaji API (模擬模式: {self.simulation})...")
        self.api = sj.Shioaji(simulation=self.simulation)
        try:
            self.api.login(api_key=self.api_key, secret_key=self.secret_key)
            print("Shioaji 登入成功！")
            return True
        except Exception as e:
            self.api = None
            raise ConnectionError(f"Shioaji 登入過程中發生異常: {str(e)}")


    def get_1m_data(self, start_date: str, end_date: str, symbol: str = "TXFR1") -> pd.DataFrame:
        """
        從 Shioaji API 獲取指定期間的台指期 1M 歷史 K 線
        參數格式: start_date='2026-06-01', end_date='2026-06-30'
        """
        cache_file = os.path.join(DATA_CACHE_DIR, f"shioaji_{symbol}_{start_date}_{end_date}.csv")
        if os.path.exists(cache_file):
            print(f"從快取載入歷史數據: {cache_file}")
            df = pd.read_csv(cache_file)
            df['ts'] = pd.to_datetime(df['ts'])
            return df

        self._login()
        
        # 尋找期貨合約
        print(f"尋找台指期合約: {symbol}...")
        try:
            # 取得台指期合約物件，通常是 api.Contracts.Futures.TXF
            # 我們可以藉由合約清單來配對近月商品
            # 若為連續月 symbol (如 TXFR1)，通常可以用特定的方式獲取，或者獲取當前的主力合約
            # 這裡我們支援輸入 TXFR1 (近一) 或具體合約如 TXF202607
            
            futures = self.api.Contracts.Futures.TXF
            target_symbol = "TXFR1" if symbol in ["TXFR1", "TXF"] else symbol
            
            try:
                contract = futures[target_symbol]
            except KeyError:
                contract = getattr(futures, target_symbol, None)
                
            if contract is None:
                raise ValueError(f"找不到指定的期貨合約: {target_symbol}")
                
            print(f"成功取得合約: {contract.code} - {contract.name}")
            
            print(f"開始下載 {contract.code} 歷史 K 線 ({start_date} 至 {end_date})...")
            kbars = self.api.kbars(
                contract=contract,
                start=start_date,
                end=end_date
            )
            
            if not kbars or len(kbars.ts) == 0:
                raise ValueError("Shioaji API 未回傳任何 K 線數據，請確認日期範圍與開市時間")

            # 轉換為 DataFrame
            df = pd.DataFrame({**kbars})
            df['ts'] = pd.to_datetime(df['ts'])
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            # 只保留需要的欄位
            df = df[['ts', 'open', 'high', 'low', 'close', 'volume']]
            
            # 存入快取
            df.to_csv(cache_file, index=False)
            print(f"歷史數據已下載並存入快取: {cache_file} (共 {len(df)} 筆)")
            return df
            
        except Exception as e:
            raise RuntimeError(f"獲取歷史數據失敗: {str(e)}")


class MockDataProvider(DataProvider):
    """高品質台指期模擬數據生成器，用於測試與展示"""
    def __init__(self, seed: int = 42):
        super().__init__()
        self.seed = seed

    def get_1m_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        生成具備 SMC 特徵的模擬 1M K 線數據。
        時間跨度包含多個交易日，每天 08:45 到 13:45 (台指期一般交易時段，共 300 分鐘)。
        """
        np.random.seed(self.seed)
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # 取得所有交易日 (排除週六週日)
        days = pd.date_range(start=start_dt, end=end_dt, freq='B')
        
        all_dfs = []
        base_price = 20000.0 # 台指期起始點數
        
        print(f"正在生成 {len(days)} 個交易日的模擬 K 線數據...")
        
        for day in days:
            # 每天生成 300 根 1M K 線 (08:45 到 13:45)
            # 建立時間序列
            start_time = datetime.datetime.combine(day.date(), datetime.time(8, 45))
            ts_list = [start_time + datetime.timedelta(minutes=i) for i in range(300)]
            
            # 設計一個有趨勢的日內價格波形（例如：開盤走高 -> 盤整 -> 列取流動性 -> 大反轉 -> BOS 延續）
            # 用幾個不同週期的正弦波，加上隨機雜訊，創造結構
            t = np.linspace(0, 4 * np.pi, 300)
            
            # 日內基本波形：前段走高，中段洗盤跌破，後段大漲 (模擬 Liquidity Sweep + MSS + CISD + BOS)
            trend = 100 * np.sin(t) - 50 * np.sin(2.5 * t) + 150 * (t / (4 * np.pi))
            
            # 加入隨機漫步
            noise = np.cumsum(np.random.normal(0, 3.0, 300))
            
            prices = base_price + trend + noise
            
            opens = []
            highs = []
            lows = []
            closes = []
            volumes = []
            
            for i in range(300):
                p = prices[i]
                prev_c = prices[i-1] if i > 0 else p
                
                # 決定這根 K 棒的開高低收
                o = prev_c + np.random.normal(0, 1.0)
                c = p
                
                # 依開收盤價加入隨機的高低點
                h = max(o, c) + max(0, np.random.exponential(2.0))
                l = min(o, c) - max(0, np.random.exponential(2.0))
                
                # 特殊特徵注入：在特定時間點製造「流動性清掃」與「爆量突破 (CISD)」
                # 假設在第 120 根 K 棒左右（大約 10:45），製造一個向下清掃前低，隨後在第 125 根進行爆量大陽線突破的結構
                if i == 120:  # 流動性清掃：影子極低但實體收高
                    l = min(o, c) - 45.0
                    c = o + 5.0
                    h = c + 2.0
                    vol = np.random.randint(1500, 2500)
                elif i in [121, 122, 123]:  # 快速拉升，形成 FVG
                    o = closes[-1]
                    c = o + 20.0 + np.random.normal(0, 2.0)
                    h = c + np.random.exponential(1.0)
                    l = o - np.random.exponential(1.0)
                    vol = np.random.randint(2000, 3500)  # 爆量
                elif i == 124: # MSS / CISD 確立大陽線
                    o = closes[-1]
                    c = o + 35.0
                    h = c + 1.0
                    l = o - 1.0
                    vol = np.random.randint(4000, 6000)  # 主力爆量推進
                else:
                    # 一般成交量
                    vol = np.random.randint(200, 800)
                    if abs(c - o) > 15:  # 大波動伴隨稍大成交量
                        vol += np.random.randint(500, 1000)
                
                opens.append(round(o, 1))
                highs.append(round(h, 1))
                lows.append(round(l, 1))
                closes.append(round(c, 1))
                volumes.append(vol)
                
            day_df = pd.DataFrame({
                'ts': ts_list,
                'open': opens,
                'high': highs,
                'low': lows,
                'close': closes,
                'volume': volumes
            })
            
            all_dfs.append(day_df)
            # 隔天開盤價繼承前一天收盤價並帶有些微跳空
            base_price = closes[-1] + np.random.normal(0, 15.0)
            
        df_all = pd.concat(all_dfs, ignore_index=True)
        return df_all


def resample_to_5m(df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    將 1 分鐘 K 線資料合成 5 分鐘 K 線資料。
    """
    df_5m = df_1m.resample('5min', on='ts').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna().reset_index()
    return df_5m
