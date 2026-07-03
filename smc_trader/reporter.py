import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from jinja2 import Template
from typing import List, Dict, Any, Tuple

class Reporter:
    """
    SMC+SNR 回測報告生成器。
    負責計算各項策略指標 (勝率, R:R, Profit Factor, MDD)，繪製資產曲線圖，並生成精美的 HTML 報告。
    """
    def __init__(self, output_dir: str = "."):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def calculate_metrics(self, trades: List[Dict[str, Any]], equity_curve: List[Dict[str, Any]], initial_capital: float) -> Dict[str, Any]:
        """
        計算回測指標
        """
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_rr': 0.0,
                'profit_factor': 0.0,
                'max_drawdown_pct': 0.0,
                'max_drawdown_points': 0.0,
                'net_profit': 0.0,
                'ending_equity': initial_capital,
                'total_wins': 0,
                'total_losses': 0
            }

        df_trades = pd.DataFrame(trades)
        df_equity = pd.DataFrame(equity_curve)

        # 1. 交易次數、勝敗筆數
        total_trades = len(df_trades)
        wins = df_trades[df_trades['pnl'] > 0]
        losses = df_trades[df_trades['pnl'] <= 0]
        total_wins = len(wins)
        total_losses = len(losses)

        # 2. 勝率 (Win Rate)
        win_rate = (total_wins / total_trades) * 100 if total_trades > 0 else 0.0

        # 3. 獲利因子 (Profit Factor)
        total_gain = wins['pnl'].sum() if total_wins > 0 else 0.0
        total_loss = abs(losses['pnl'].sum()) if total_losses > 0 else 0.0
        profit_factor = total_gain / total_loss if total_loss > 0 else (float('inf') if total_gain > 0 else 1.0)

        # 4. 平均盈虧比 (Avg Risk-Reward Ratio)
        # 用於衡量實際達到的平均獲利與平均虧損比例
        avg_gain = wins['pnl'].mean() if total_wins > 0 else 0.0
        avg_loss = abs(losses['pnl'].mean()) if total_losses > 0 else 0.0
        avg_rr = avg_gain / avg_loss if avg_loss > 0 else (float('inf') if avg_gain > 0 else 0.0)

        # 5. 最大回撤 (Maximum Drawdown)
        # 計算百分比回撤與絕對值點數回撤
        df_equity['peak'] = df_equity['equity'].cummax()
        df_equity['dd_pct'] = (df_equity['peak'] - df_equity['equity']) / df_equity['peak'] * 100
        df_equity['dd_val'] = df_equity['peak'] - df_equity['equity']
        
        max_drawdown_pct = df_equity['dd_pct'].max()
        max_drawdown_val = df_equity['dd_val'].max()

        # 6. 淨利與期末權益
        ending_equity = df_equity['equity'].iloc[-1]
        net_profit = ending_equity - initial_capital

        return {
            'total_trades': total_trades,
            'win_rate': round(win_rate, 2),
            'avg_rr': round(avg_rr, 2) if avg_rr != float('inf') else "Infinity",
            'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else "Infinity",
            'max_drawdown_pct': round(max_drawdown_pct, 2),
            'max_drawdown_val': round(max_drawdown_val, 2),
            'net_profit': round(net_profit, 2),
            'ending_equity': round(ending_equity, 2),
            'total_wins': total_wins,
            'total_losses': total_losses
        }

    def plot_equity_curve(self, equity_curve: List[Dict[str, Any]], filename: str = "equity_curve.png") -> str:
        """
        繪製資產曲線圖與回撤圖，並儲存為圖片。
        """
        df_equity = pd.DataFrame(equity_curve)
        df_equity['peak'] = df_equity['equity'].cummax()
        df_equity['drawdown'] = (df_equity['equity'] - df_equity['peak'])

        # 建立畫布
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        
        # 使用現代暗黑配色
        plt.style.use('dark_background')
        fig.patch.set_facecolor('#0f172a')
        ax1.set_facecolor('#1e293b')
        ax2.set_facecolor('#1e293b')

        # 繪製資產曲線
        ax1.plot(df_equity['ts'], df_equity['equity'], color='#10b981', linewidth=2, label='Equity Net Value')
        ax1.fill_between(df_equity['ts'], df_equity['equity'], df_equity['equity'].min(), color='#10b981', alpha=0.1)
        ax1.set_title('SMC Strategy Backtest - Equity Curve', fontsize=14, color='#f8fafc', fontweight='bold', pad=15)
        ax1.set_ylabel('Equity (NTD)', fontsize=12, color='#94a3b8')
        ax1.grid(True, color='#334155', linestyle='--', alpha=0.5)
        ax1.tick_params(colors='#94a3b8')

        # 繪製回撤圖
        ax2.fill_between(df_equity['ts'], df_equity['drawdown'], 0, color='#f43f5e', alpha=0.3, label='Drawdown')
        ax2.plot(df_equity['ts'], df_equity['drawdown'], color='#f43f5e', linewidth=1)
        ax2.set_ylabel('Drawdown (NTD)', fontsize=12, color='#94a3b8')
        ax2.set_xlabel('Date Time', fontsize=12, color='#94a3b8')
        ax2.grid(True, color='#334155', linestyle='--', alpha=0.5)
        ax2.tick_params(colors='#94a3b8')
        
        # 自動旋轉 X 軸日期標籤
        plt.xticks(rotation=15)
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, facecolor=fig.get_facecolor(), edgecolor='none', dpi=150)
        plt.close()
        
        return filepath

    def generate_html_report(self, metrics: Dict[str, Any], trades: List[Dict[str, Any]], 
                             equity_filepath: str, filename: str = "report.html") -> str:
        """
        使用 HTML 與現代化 CSS 生成回測報告。
        """
        # 轉換圖片路徑為相鄰的相對路徑以利在瀏覽器中顯示
        img_name = os.path.basename(equity_filepath)

        # 準備交易列表的顯示數據
        formatted_trades = []
        for t in trades:
            formatted_trades.append({
                'direction': '多單 (LONG)' if t['direction'] == 'LONG' else '空單 (SHORT)',
                'entry_time': t['entry_time'].strftime('%Y-%m-%d %H:%M') if isinstance(t['entry_time'], pd.Timestamp) else str(t['entry_time']),
                'entry_price': round(t['entry_price'], 1),
                'exit_time': t['exit_time'].strftime('%Y-%m-%d %H:%M') if isinstance(t['exit_time'], pd.Timestamp) else str(t['exit_time']),
                'exit_price': round(t['exit_price'], 1),
                'exit_type': '停利 (TP)' if t['exit_type'] == 'TP' else '停損 (SL)',
                'net_points': round(t['net_points'], 1),
                'pnl': round(t['pnl'], 0),
                'balance_after': round(t['balance_after'], 0),
                'pnl_class': 'text-emerald' if t['pnl'] > 0 else 'text-rose'
            })

        template_str = """
        <!DOCTYPE html>
        <html lang="zh-TW">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>SMC + SNR 交易策略回測報告</title>
            <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Noto+Sans+TC:wght@300;400;700&display=swap" rel="stylesheet">
            <style>
                :root {
                    --bg-main: #0f172a;
                    --bg-card: #1e293b;
                    --bg-input: #334155;
                    --text-main: #f8fafc;
                    --text-muted: #94a3b8;
                    --primary: #10b981;
                    --primary-hover: #34d399;
                    --accent-rose: #f43f5e;
                    --accent-cyan: #06b6d4;
                    --border-color: #334155;
                }
                * {
                    box-sizing: border-box;
                    margin: 0;
                    padding: 0;
                }
                body {
                    font-family: 'Outfit', 'Noto Sans TC', sans-serif;
                    background-color: var(--bg-main);
                    color: var(--text-main);
                    line-height: 1.6;
                    padding: 2rem;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                }
                header {
                    text-align: center;
                    margin-bottom: 2.5rem;
                    border-bottom: 1px solid var(--border-color);
                    padding-bottom: 1.5rem;
                }
                header h1 {
                    font-size: 2.5rem;
                    font-weight: 800;
                    background: linear-gradient(135deg, var(--primary), var(--accent-cyan));
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    margin-bottom: 0.5rem;
                }
                header p {
                    color: var(--text-muted);
                    font-size: 1.1rem;
                }
                .grid-metrics {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                    gap: 1.5rem;
                    margin-bottom: 2.5rem;
                }
                .card {
                    background-color: var(--bg-card);
                    border: 1px solid var(--border-color);
                    border-radius: 16px;
                    padding: 1.5rem;
                    text-align: center;
                    transition: transform 0.3s ease, border-color 0.3s ease;
                    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
                }
                .card:hover {
                    transform: translateY(-5px);
                    border-color: var(--primary);
                }
                .card h3 {
                    font-size: 0.95rem;
                    color: var(--text-muted);
                    text-transform: uppercase;
                    letter-spacing: 0.05em;
                    margin-bottom: 0.75rem;
                }
                .card .value {
                    font-size: 2rem;
                    font-weight: 700;
                    color: var(--text-main);
                }
                .card .value.positive {
                    color: var(--primary);
                }
                .card .value.negative {
                    color: var(--accent-rose);
                }
                .content-section {
                    background-color: var(--bg-card);
                    border: 1px solid var(--border-color);
                    border-radius: 16px;
                    padding: 2rem;
                    margin-bottom: 2.5rem;
                    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
                }
                .content-section h2 {
                    font-size: 1.5rem;
                    margin-bottom: 1.5rem;
                    border-left: 4px solid var(--primary);
                    padding-left: 0.75rem;
                }
                .chart-container {
                    text-align: center;
                    margin: 1rem 0;
                }
                .chart-container img {
                    max-width: 100%;
                    height: auto;
                    border-radius: 12px;
                    border: 1px solid var(--border-color);
                }
                .table-responsive {
                    overflow-x: auto;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    text-align: left;
                    font-size: 0.95rem;
                }
                th {
                    background-color: #1e293b;
                    padding: 1rem;
                    color: var(--text-muted);
                    font-weight: 600;
                    border-bottom: 2px solid var(--border-color);
                }
                td {
                    padding: 1rem;
                    border-bottom: 1px solid var(--border-color);
                    color: var(--text-main);
                }
                tr:hover td {
                    background-color: rgba(255, 255, 255, 0.03);
                }
                .text-emerald {
                    color: var(--primary);
                    font-weight: 600;
                }
                .text-rose {
                    color: var(--accent-rose);
                    font-weight: 600;
                }
                .badge {
                    display: inline-block;
                    padding: 0.25rem 0.6rem;
                    border-radius: 50px;
                    font-size: 0.8rem;
                    font-weight: 600;
                }
                .badge-tp {
                    background-color: rgba(16, 185, 129, 0.15);
                    color: var(--primary);
                }
                .badge-sl {
                    background-color: rgba(244, 63, 94, 0.15);
                    color: var(--accent-rose);
                }
                footer {
                    text-align: center;
                    padding: 2rem 0;
                    color: var(--text-muted);
                    font-size: 0.9rem;
                    border-top: 1px solid var(--border-color);
                }
            </style>
        </head>
        <body>
            <div class="container">
                <header>
                    <h1>SMC + SNR 智能策略回測報告</h1>
                    <p>標的：台股指數期貨 (TXF) | 雙時框分析 (5M 大趨勢 + 1M 內部結構進場)</p>
                </header>

                <!-- 指標網格 -->
                <div class="grid-metrics">
                    <div class="card">
                        <h3>總交易筆數</h3>
                        <div class="value">{{ metrics.total_trades }}</div>
                    </div>
                    <div class="card">
                        <h3>勝率</h3>
                        <div class="value {% if metrics.win_rate >= 50 %}positive{% else %}negative{% endif %}">{{ metrics.win_rate }}%</div>
                    </div>
                    <div class="card">
                        <h3>獲利因子 (PF)</h3>
                        <div class="value {% if metrics.profit_factor != 'Infinity' and metrics.profit_factor >= 1.5 %}positive{% endif %}">{{ metrics.profit_factor }}</div>
                    </div>
                    <div class="card">
                        <h3>平均盈虧比 (R:R)</h3>
                        <div class="value">{{ metrics.avg_rr }}</div>
                    </div>
                    <div class="card">
                        <h3>最大回撤 (MDD)</h3>
                        <div class="value negative">{{ metrics.max_drawdown_pct }}%</div>
                    </div>
                    <div class="card">
                        <h3>淨損益 (NTD)</h3>
                        <div class="value {% if metrics.net_profit >= 0 %}positive{% else %}negative{% endif %}">
                            {{ "${:,.0f}".format(metrics.net_profit) }}
                        </div>
                    </div>
                </div>

                <!-- 資產曲線圖區塊 -->
                <div class="content-section">
                    <h2>資產淨值與回撤曲線 (Equity Curve)</h2>
                    <div class="chart-container">
                        <img src="{{ img_name }}" alt="Equity Curve Chart">
                    </div>
                </div>

                <!-- 交易明細區塊 -->
                <div class="content-section">
                    <h2>交易歷史明細</h2>
                    <div class="table-responsive">
                        <table>
                            <thead>
                                <tr>
                                    <th>編號</th>
                                    <th>交易方向</th>
                                    <th>進場時間</th>
                                    <th>進場價格</th>
                                    <th>出場時間</th>
                                    <th>出場價格</th>
                                    <th>出場類型</th>
                                    <th>淨點數盈虧</th>
                                    <th>淨金額收益</th>
                                    <th>交易後餘額</th>
                                </tr>
                            </thead>
                            <tbody>
                                {% for trade in trades %}
                                <tr>
                                    <td>{{ loop.index }}</td>
                                    <td>{{ trade.direction }}</td>
                                    <td>{{ trade.entry_time }}</td>
                                    <td>{{ trade.entry_price }}</td>
                                    <td>{{ trade.exit_time }}</td>
                                    <td>{{ trade.exit_price }}</td>
                                    <td>
                                        <span class="badge {% if trade.exit_type == '停利 (TP)' %}badge-tp{% else %}badge-sl{% endif %}">
                                            {{ trade.exit_type }}
                                        </span>
                                    </td>
                                    <td class="{{ trade.pnl_class }}">{{ trade.net_points }}</td>
                                    <td class="{{ trade.pnl_class }}">{{ "${:,.0f}".format(trade.pnl) }}</td>
                                    <td>{{ "${:,.0f}".format(trade.balance_after) }}</td>
                                </tr>
                                {% else %}
                                <tr>
                                    <td colspan="10" style="text-align: center; color: var(--text-muted);">本回測區間內無交易成交記錄</td>
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                </div>

                <footer>
                    <p>由 Antigravity 智能回測引擎自動生成 &copy; 2026</p>
                </footer>
            </div>
        </body>
        </html>
        """

        t = Template(template_str)
        rendered_html = t.render(metrics=metrics, trades=formatted_trades, img_name=img_name)
        
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(rendered_html)
            
        return filepath
